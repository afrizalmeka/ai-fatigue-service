import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import cv2
from ultralytics import YOLO
import numpy as np
import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from PIL import Image
import random
from datetime import datetime

# Hardcoded variables
TEST_FOLDER = "TEST ALL"
EXCEL_FILE = "Result Ai Testing Yawning.xlsx"
MODEL_PATH = os.path.join("model", "yawning_pytorch_model_best2.pth")
OUTPUT_EXCEL = "AI_Yawning_Test_Results.xlsx"  # New output file
FRAMES_TO_SAMPLE = 20  # Same as training
CONFIDENCE_THRESHOLD = 0.25
FACE_SIZE = 448
TARGET_FRAMES = 40  # Must match training script
YOLO_WEIGHTS_PATH = os.path.join("model", "yawning_best.pt") # Path to YOLO weights
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize YOLO model
yolo_model = YOLO(YOLO_WEIGHTS_PATH)

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(ResNetFeatureExtractor, self).__init__()
        # Load ResNet50 with ImageNet weights
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layers (avgpool and fc)
        # Keep up to layer4 to get spatial features
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Optionally freeze ResNet parameters
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            print("ResNet backbone is frozen")
        else:
            print("ResNet backbone is trainable")
            
    def forward(self, x):
        features = self.features(x)
        # Global average pooling to get [batch_size, 2048]
        features = torch.mean(features, dim=[2, 3])
        return features

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim=2048):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class LogisticRegressionModelLogits(nn.Module):
    """Logistic regression model that outputs logits (for BCEWithLogitsLoss)"""
    def __init__(self, input_dim=2048):
        super(LogisticRegressionModelLogits, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        x = self.linear(x)
        return x

def load_ground_truth():
    """Load ground truth from Excel file and apply the specified logic"""
    print("Loading ground truth from Excel file...")
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        
        # Clean the data - keep only rows with valid data
        df_clean = df[['File Name', 'Answer', 'Result']].dropna()
        
        print(f"Loaded {len(df_clean)} test samples from Excel")
        
        # Apply the ground truth logic:
        # if answer false and result salah -> true (yawning detection was correct)
        # if answer false and result benar -> false (yawning detection was wrong)
        # if answer true and result benar -> true (yawning detection was correct)
        # if answer true and result salah -> false (yawning detection was wrong)
        
        ground_truth = {}
        for _, row in df_clean.iterrows():
            filename = row['File Name']
            answer = row['Answer']  # True/False
            result = row['Result']  # BENAR/SALAH
            
            # Apply the logic
            if answer == False and result == 'SALAH':
                true_label = 1  # Yawning detection was correct
            elif answer == False and result == 'BENAR':
                true_label = 0  # Yawning detection was wrong
            elif answer == True and result == 'BENAR':
                true_label = 1  # Yawning detection was correct
            elif answer == True and result == 'SALAH':
                true_label = 0  # Yawning detection was wrong
            else:
                print(f"Warning: Unexpected combination for {filename}: Answer={answer}, Result={result}")
                continue
            
            ground_truth[filename] = true_label
        
        print(f"Ground truth mapping created for {len(ground_truth)} videos")
        print(f"Yawning (1): {sum(ground_truth.values())}")
        print(f"Not Yawning (0): {len(ground_truth) - sum(ground_truth.values())}")
        
        return ground_truth
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return {}

def get_uniform_frame_indices(total_frames, num_samples):
    """Get uniformly distributed frame indices"""
    if total_frames <= num_samples:
        return list(range(total_frames))
    
    step = total_frames / num_samples
    indices = [int(i * step) for i in range(num_samples)]
    indices = list(set([min(idx, total_frames - 1) for idx in indices]))
    indices.sort()
    
    return indices

def select_best_face(boxes):
    """Select the best face based on size and confidence"""
    if boxes is None or len(boxes) == 0:
        return None
    
    best_box = None
    best_score = 0
    
    for box in boxes:
        # Get bounding box coordinates and confidence
        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
        confidence = box.conf[0].cpu().numpy()
        
        # Calculate face area
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Combined score: confidence * area (normalized)
        # This prioritizes both confidence and size
        score = confidence * (area / 10000)  # Normalize area to reasonable range
        
        if score > best_score:
            best_score = score
            best_box = box
    
    return best_box

def crop_face_from_bbox(frame, bbox, target_size=448):
    """Crop face from frame based on YOLO bounding box with padding"""
    h, w, _ = frame.shape
    
    # YOLO bbox format: [x1, y1, x2, y2] in pixel coordinates
    x1, y1, x2, y2 = map(int, bbox)
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Add padding to make it more square and include more context
    padding_factor = 0.3  # 30% padding
    pad_x = int(width * padding_factor)
    pad_y = int(height * padding_factor)
    
    # Calculate new coordinates with padding
    x1_padded = max(0, x1 - pad_x)
    y1_padded = max(0, y1 - pad_y)
    x2_padded = min(w, x2 + pad_x)
    y2_padded = min(h, y2 + pad_y)
    
    # Crop the face region
    face_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
    
    if face_crop.size == 0:
        return None
    
    # Resize to target size
    face_resized = cv2.resize(face_crop, (target_size, target_size))
    
    return face_resized

def extract_faces_from_video(video_path):
    """Extract face crops from video using uniform sampling with YOLO detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get uniform frame indices
    frame_indices = get_uniform_frame_indices(total_frames, FRAMES_TO_SAMPLE)
    
    face_crops = []
    last_bbox = None  # Store the last detected bounding box
    
    for frame_idx in frame_indices:
        # Set video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Run YOLO inference on the frame
        results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Process detected faces - select only the best one
        current_bbox = None
        face_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Select the best face (largest with highest confidence)
                best_box = select_best_face(boxes)
                
                if best_box is not None:
                    # Get bounding box coordinates
                    current_bbox = best_box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    face_detected = True
                    last_bbox = current_bbox.copy()  # Update last known bbox
                    break
        
        # Use current detection or fall back to last known bbox
        bbox_to_use = current_bbox if face_detected else last_bbox
        
        if bbox_to_use is not None:
            # Crop face from the frame using the selected bbox
            face_crop = crop_face_from_bbox(frame, bbox_to_use, FACE_SIZE)
            
            if face_crop is not None:
                face_crops.append(face_crop)
    
    cap.release()
    return face_crops

def pad_or_sample_frames(face_images, target_count):
    """Pad or sample frames to get exactly target_count frames"""
    current_count = len(face_images)
    
    if current_count == target_count:
        # Perfect match, return as is
        return face_images
    elif current_count < target_count:
        # Need to pad - duplicate existing frames
        padded_images = face_images.copy()
        
        # Calculate how many frames we need to add
        frames_to_add = target_count - current_count
        
        # Duplicate frames cyclically to reach target count
        for i in range(frames_to_add):
            # Use modulo to cycle through existing frames
            duplicate_idx = i % current_count
            padded_images.append(face_images[duplicate_idx])
        
        return padded_images
    else:
        # Need to sample - randomly select target_count frames
        # Use random sampling without replacement
        sampled_indices = random.sample(range(current_count), target_count)
        sampled_indices.sort()  # Keep temporal order
        
        sampled_images = [face_images[i] for i in sampled_indices]
        return sampled_images

def preprocess_faces_for_resnet(face_crops):
    """Preprocess face crops for ResNet feature extraction with padding/sampling"""
    if not face_crops:
        return None
    
    # Apply padding/sampling to get exactly TARGET_FRAMES
    face_crops = pad_or_sample_frames(face_crops, TARGET_FRAMES)
    
    # Define transforms for ResNet (ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    processed_faces = []
    for face in face_crops:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # Apply transforms
        face_tensor = transform(face_rgb)
        processed_faces.append(face_tensor)
    
    if processed_faces:
        # Stack and take mean across faces
        face_tensor = torch.stack(processed_faces).mean(dim=0)
        return face_tensor.unsqueeze(0)  # Add batch dimension
    else:
        return None

def load_pytorch_model():
    """Load the trained PyTorch model"""
    global TARGET_FRAMES  # Declare global at the beginning
    
    print("Loading PyTorch yawning model...")
    
    # Load checkpoint first to check model type
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Initialize models
    backbone_trainable = checkpoint.get('backbone_trainable', False)
    feature_extractor = ResNetFeatureExtractor(freeze_backbone=not backbone_trainable)
    
    # Check if model uses weighted loss (logits) or regular loss (sigmoid)
    use_weighted_loss = checkpoint.get('use_weighted_loss', False)
    use_focal_loss = checkpoint.get('use_focal_loss', False)
    
    if use_weighted_loss and not use_focal_loss:
        classifier = LogisticRegressionModelLogits()
        print("Loaded model with weighted BCE loss (logits output)")
    else:
        classifier = LogisticRegressionModel()
        print("Loaded model with standard BCE or focal loss (sigmoid output)")
    
    # Load state dictionaries
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Move to device and set to eval mode
    feature_extractor.to(DEVICE)
    classifier.to(DEVICE)
    feature_extractor.eval()
    classifier.eval()
    
    # print(f"Model loaded successfully! Best validation F1-score: {checkpoint.get('best_val_f1', 'N/A')}")
    print(f"Backbone trainable: {backbone_trainable}")
    
    # Check if target frames match
    if 'target_frames' in checkpoint:
        model_target_frames = checkpoint['target_frames']
        if model_target_frames != TARGET_FRAMES:
            print(f"Warning: Model was trained with {model_target_frames} frames, but test script uses {TARGET_FRAMES}")
            print(f"Updating TARGET_FRAMES to match model: {model_target_frames}")
            TARGET_FRAMES = model_target_frames
    
    return feature_extractor, classifier, use_weighted_loss, use_focal_loss

def create_results_excel(predictions, true_labels, prediction_probabilities, test_accuracy, test_cm, classification_rep):
    """Create Excel file with test results in multiple sheets"""
    print(f"\nCreating results Excel file: {OUTPUT_EXCEL}")
    
    # Create Excel writer
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        
        # Sheet 1: Detailed Results
        results_data = []
        for filename in predictions.keys():
            ai_answer = predictions[filename]
            ground_truth = true_labels[filename]
            confidence = prediction_probabilities[filename][ai_answer]
            
            # Convert to readable labels
            ai_answer_text = "Yawning" if ai_answer == 1 else "Not Yawning"
            ground_truth_text = "Yawning" if ground_truth == 1 else "Not Yawning"
            correct = "✓" if ai_answer == ground_truth else "✗"
            
            results_data.append({
                'Filename': filename,
                'AI Answer': ai_answer_text,
                'AI Answer (Numeric)': ai_answer,
                'Ground Truth': ground_truth_text,
                'Ground Truth (Numeric)': ground_truth,
                'Confidence': round(confidence, 4),
                'Correct': correct,
                'Not Yawning Probability': round(prediction_probabilities[filename][0], 4),
                'Yawning Probability': round(prediction_probabilities[filename][1], 4)
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Sheet 2: Accuracy and Confusion Matrix
        accuracy_data = []
        
        # Overall accuracy
        accuracy_data.append(['Overall Accuracy', f'{test_accuracy:.4f}', f'{test_accuracy*100:.2f}%'])
        accuracy_data.append(['', '', ''])  # Empty row
        
        # Confusion Matrix
        accuracy_data.append(['Confusion Matrix', '', ''])
        accuracy_data.append(['', 'Predicted Not Yawning', 'Predicted Yawning'])
        accuracy_data.append(['Actual Not Yawning', test_cm[0,0], test_cm[0,1]])
        accuracy_data.append(['Actual Yawning', test_cm[1,0], test_cm[1,1]])
        accuracy_data.append(['', '', ''])  # Empty row
        
        # Performance metrics
        tn, fp, fn, tp = test_cm.ravel()
        precision_not_yawning = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_not_yawning = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_not_yawning = 2 * (precision_not_yawning * recall_not_yawning) / (precision_not_yawning + recall_not_yawning) if (precision_not_yawning + recall_not_yawning) > 0 else 0
        
        precision_yawning = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_yawning = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_yawning = 2 * (precision_yawning * recall_yawning) / (precision_yawning + recall_yawning) if (precision_yawning + recall_yawning) > 0 else 0
        
        accuracy_data.append(['Performance Metrics', '', ''])
        accuracy_data.append(['', 'Not Yawning', 'Yawning'])
        accuracy_data.append(['Precision', f'{precision_not_yawning:.4f}', f'{precision_yawning:.4f}'])
        accuracy_data.append(['Recall', f'{recall_not_yawning:.4f}', f'{recall_yawning:.4f}'])
        accuracy_data.append(['F1-Score', f'{f1_not_yawning:.4f}', f'{f1_yawning:.4f}'])
        accuracy_data.append(['', '', ''])  # Empty row
        
        # Summary statistics
        total_samples = len(predictions)
        correct_predictions = sum(1 for name in predictions if predictions[name] == true_labels[name])
        incorrect_predictions = total_samples - correct_predictions
        
        accuracy_data.append(['Summary Statistics', '', ''])
        accuracy_data.append(['Total Test Samples', total_samples, ''])
        accuracy_data.append(['Correct Predictions', correct_predictions, ''])
        accuracy_data.append(['Incorrect Predictions', incorrect_predictions, ''])
        accuracy_data.append(['', '', ''])  # Empty row
        
        # Test configuration
        accuracy_data.append(['Test Configuration', '', ''])
        accuracy_data.append(['Test Folder', TEST_FOLDER, ''])
        accuracy_data.append(['Model Path', MODEL_PATH, ''])
        accuracy_data.append(['Target Frames', TARGET_FRAMES, ''])
        accuracy_data.append(['Device Used', str(DEVICE), ''])
        accuracy_data.append(['Test Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ''])
        
        accuracy_df = pd.DataFrame(accuracy_data, columns=['Metric', 'Value', 'Percentage'])
        accuracy_df.to_excel(writer, sheet_name='Accuracy & Metrics', index=False)
        
        # Sheet 3: Classification Report
        # Parse classification report into a more readable format
        report_lines = classification_rep.split('\n')
        report_data = []
        
        report_data.append(['Classification Report', '', '', '', ''])
        report_data.append(['', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        for line in report_lines:
            line = line.strip()
            if line and not line.startswith('accuracy') and 'avg' not in line.lower():
                parts = line.split()
                if len(parts) >= 5:
                    class_name = ' '.join(parts[:-4])
                    if class_name:
                        report_data.append([class_name, parts[-4], parts[-3], parts[-2], parts[-1]])
        
        # Add accuracy and averages
        for line in report_lines:
            line = line.strip()
            if 'accuracy' in line or 'avg' in line.lower():
                parts = line.split()
                if len(parts) >= 4:
                    if 'accuracy' in line:
                        report_data.append(['Accuracy', '', '', parts[1], parts[2]])
                    else:
                        avg_type = ' '.join(parts[:-3])
                        report_data.append([avg_type, parts[-3], parts[-2], parts[-1], ''])
        
        report_df = pd.DataFrame(report_data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        report_df.to_excel(writer, sheet_name='Classification Report', index=False)
    
    print(f"Results saved to: {OUTPUT_EXCEL}")
    print("Excel file contains 3 sheets:")
    print("  1. 'Detailed Results' - Individual predictions for each video")
    print("  2. 'Accuracy & Metrics' - Overall performance metrics and confusion matrix")
    print("  3. 'Classification Report' - Detailed classification metrics")

def test_model():
    """Test the trained PyTorch yawning model on test videos"""
    print("=== TESTING PYTORCH YAWNING DETECTION MODEL (YOLO) ===")
    print(f"Test folder: {TEST_FOLDER}")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"YOLO weights: {YOLO_WEIGHTS_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Target frames per video: {TARGET_FRAMES}")
    print(f"Face size: {FACE_SIZE}x{FACE_SIZE}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("-" * 50)
    
    # Set random seed for reproducible sampling
    random.seed(42)
    
    # Load ground truth
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("Error: Could not load ground truth!")
        return
    
    # Load trained model
    try:
        feature_extractor, classifier, use_weighted_loss, use_focal_loss = load_pytorch_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run the PyTorch yawning training script first!")
        return
    
    # Get all test videos
    video_files = glob.glob(os.path.join(TEST_FOLDER, "*.mp4"))
    video_files.sort()
    
    print(f"Found {len(video_files)} test videos")
    
    # Process each video
    predictions = {}
    true_labels = {}
    prediction_probabilities = {}
    frame_count_stats = []
    
    print("\nProcessing test videos...")
    for i, video_path in enumerate(tqdm(video_files, desc="Testing videos")):
        video_name = os.path.basename(video_path).replace('.mp4', '')
        
        # Check if we have ground truth for this video
        if video_name not in ground_truth:
            print(f"  Warning: No ground truth found for {video_name}")
            continue
        
        try:
            # Extract faces from video
            face_crops = extract_faces_from_video(video_path)
            original_frame_count = len(face_crops)
            frame_count_stats.append(original_frame_count)
            
            if not face_crops:
                # Default prediction when no faces detected
                predictions[video_name] = 0  # Assume not yawning
                prediction_probabilities[video_name] = [1.0, 0.0]  # [prob_not_yawning, prob_yawning]
                true_labels[video_name] = ground_truth[video_name]
                continue
            
            # Preprocess faces (includes padding/sampling)
            face_tensor = preprocess_faces_for_resnet(face_crops)
            
            if face_tensor is None:
                predictions[video_name] = 0
                prediction_probabilities[video_name] = [1.0, 0.0]
                true_labels[video_name] = ground_truth[video_name]
                continue
            
            # Move to device
            face_tensor = face_tensor.to(DEVICE)
            
            # Extract features and make prediction
            with torch.no_grad():
                features = feature_extractor(face_tensor)
                output = classifier(features).squeeze()
                
                # Get prediction and probability based on model type
                if use_weighted_loss and not use_focal_loss:
                    # Model outputs logits, apply sigmoid
                    prediction_prob = torch.sigmoid(output).item()
                else:
                    # Model outputs probabilities directly
                    prediction_prob = output.item()
                
                prediction = 1 if prediction_prob > 0.5 else 0
                
                predictions[video_name] = prediction
                prediction_probabilities[video_name] = [1-prediction_prob, prediction_prob]
                true_labels[video_name] = ground_truth[video_name]
            
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            predictions[video_name] = 0
            prediction_probabilities[video_name] = [1.0, 0.0]
            true_labels[video_name] = ground_truth[video_name]
    
    # Show frame count statistics
    if frame_count_stats:
        print(f"\nFrame Count Statistics:")
        print(f"Original frames - Min: {min(frame_count_stats)}, Max: {max(frame_count_stats)}, Avg: {np.mean(frame_count_stats):.1f}")
        print(f"After padding/sampling: All videos have {TARGET_FRAMES} frames")
        
        need_padding = sum(1 for count in frame_count_stats if count < TARGET_FRAMES)
        need_sampling = sum(1 for count in frame_count_stats if count > TARGET_FRAMES)
        exact_match = sum(1 for count in frame_count_stats if count == TARGET_FRAMES)
        
        print(f"Videos needing padding: {need_padding}")
        print(f"Videos needing sampling: {need_sampling}")
        print(f"Videos with exact frame count: {exact_match}")
    
    # Calculate test results
    if predictions:
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        
        y_true = [true_labels[name] for name in predictions.keys()]
        y_pred = [predictions[name] for name in predictions.keys()]
        
        test_accuracy = accuracy_score(y_true, y_pred)
        test_cm = confusion_matrix(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred, target_names=['Not Yawning', 'Yawning'])
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nTest Confusion Matrix:")
        print("    Predicted")
        print("    0    1")
        print(f"0  {test_cm[0,0]:3d}  {test_cm[0,1]:3d}  Actual")
        print(f"1  {test_cm[1,0]:3d}  {test_cm[1,1]:3d}")
        
        print("\nTest Classification Report:")
        print(classification_rep)
        
        # Detailed results
        print(f"\nDetailed Results:")
        print(f"Total test samples: {len(predictions)}")
        print(f"Correct predictions: {sum(1 for name in predictions if predictions[name] == true_labels[name])}")
        print(f"Incorrect predictions: {sum(1 for name in predictions if predictions[name] != true_labels[name])}")
        
        # Show some examples of correct and incorrect predictions
        correct_examples = [(name, predictions[name], true_labels[name], prediction_probabilities[name]) 
                          for name in predictions if predictions[name] == true_labels[name]]
        incorrect_examples = [(name, predictions[name], true_labels[name], prediction_probabilities[name]) 
                            for name in predictions if predictions[name] != true_labels[name]]
        
        print(f"\nCorrect Predictions (showing first 5):")
        for name, pred, true, probs in correct_examples[:5]:
            confidence = probs[pred]
            print(f"  {name}: Predicted={pred} ({'Yawning' if pred==1 else 'Not Yawning'}), Actual={true}, Confidence={confidence:.3f}")
        
        print(f"\nIncorrect Predictions (showing all):")
        for name, pred, true, probs in incorrect_examples:
            confidence = probs[pred]
            print(f"  {name}: Predicted={pred} ({'Yawning' if pred==1 else 'Not Yawning'}), Actual={true}, Confidence={confidence:.3f}")
        
        # Create Excel file with results
        create_results_excel(predictions, true_labels, prediction_probabilities, 
                           test_accuracy, test_cm, classification_rep)
    
    else:
        print("No predictions made!")

def main():
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Test folder '{TEST_FOLDER}' not found!")
        return
    
    if not os.path.exists(EXCEL_FILE):
        print(f"Error: Excel file '{EXCEL_FILE}' not found!")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please run the PyTorch yawning training script first!")
        return
    
    if not os.path.exists(YOLO_WEIGHTS_PATH):
        print(f"Error: YOLO weights file '{YOLO_WEIGHTS_PATH}' not found!")
        return
    
    test_model()

if __name__ == "__main__":
    main() 