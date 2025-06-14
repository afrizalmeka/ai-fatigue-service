import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import cv2
import mediapipe as mp
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
EXCEL_FILE = "Result Ai Testing Eye Close.xlsx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "eyes_closed_pytorch_model_best1.pth")
OUTPUT_EXCEL = "AI_Test_Results.xlsx"  # New output file
FRAMES_TO_SAMPLE = 20  # Same as training
CONFIDENCE_THRESHOLD = 0.4
FACE_SIZE = 256
TARGET_FRAMES = 40  # Must match training script
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        # Load ResNet50 with ImageNet weights
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layers (avgpool and fc)
        # Keep up to layer4 to get spatial features
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze ResNet parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
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

def load_ground_truth():
    """Load ground truth from Excel file and apply the specified logic"""
    print("Loading ground truth from Excel file...")
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        
        # Clean the data - keep only rows with valid data
        df_clean = df[['File Name', 'Answer', 'Result']].dropna()
        
        print(f"Loaded {len(df_clean)} test samples from Excel")
        
        # Apply the ground truth logic:
        # if answer false and result salah -> true
        # if answer false and result benar -> false
        # if answer true and result benar -> true  
        # if answer true and result salah -> false
        
        ground_truth = {}
        for _, row in df_clean.iterrows():
            filename = row['File Name']
            answer = row['Answer']  # True/False
            result = row['Result']  # BENAR/SALAH
            
            # Apply the logic
            if answer == False and result == 'SALAH':
                true_label = 1  # Eyes closed detection was correct
            elif answer == False and result == 'BENAR':
                true_label = 0  # Eyes closed detection was wrong
            elif answer == True and result == 'BENAR':
                true_label = 1  # Eyes closed detection was correct
            elif answer == True and result == 'SALAH':
                true_label = 0  # Eyes closed detection was wrong
            else:
                print(f"Warning: Unexpected combination for {filename}: Answer={answer}, Result={result}")
                continue
            
            ground_truth[filename] = true_label
        
        print(f"Ground truth mapping created for {len(ground_truth)} videos")
        print(f"Eyes Closed (1): {sum(ground_truth.values())}")
        print(f"Eyes Open (0): {len(ground_truth) - sum(ground_truth.values())}")
        
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

def crop_face_from_detection(frame, detection, target_size=256):
    """Crop face from frame based on detection with padding"""
    h, w, _ = frame.shape
    
    # Get bounding box
    bbox = detection.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    width = int(bbox.width * w)
    height = int(bbox.height * h)
    
    # Add padding
    padding_factor = 0.3
    pad_x = int(width * padding_factor)
    pad_y = int(height * padding_factor)
    
    # Calculate new coordinates with padding
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + width + pad_x)
    y2 = min(h, y + height + pad_y)
    
    # Crop the face region
    face_crop = frame[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return None
    
    # Resize to target size
    face_resized = cv2.resize(face_crop, (target_size, target_size))
    
    return face_resized

def extract_faces_from_video(video_path):
    """Extract face crops from video using uniform sampling"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get uniform frame indices
    frame_indices = get_uniform_frame_indices(total_frames, FRAMES_TO_SAMPLE)
    
    face_crops = []
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(
        model_selection=0, 
        min_detection_confidence=CONFIDENCE_THRESHOLD
    ) as face_detection:
        
        for frame_idx in frame_indices:
            # Set video position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame for face detection
            results = face_detection.process(rgb_frame)
            
            # Process detected faces (take first face only)
            if results.detections:
                for detection in results.detections:
                    face_crop = crop_face_from_detection(frame, detection, FACE_SIZE)
                    if face_crop is not None:
                        face_crops.append(face_crop)
                        break  # Only take first face
    
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
    
    print("Loading PyTorch model...")
    
    # Initialize models
    feature_extractor = ResNetFeatureExtractor()
    classifier = LogisticRegressionModel()
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    # Move to device and set to eval mode
    feature_extractor.to(DEVICE)
    classifier.to(DEVICE)
    feature_extractor.eval()
    classifier.eval()
    
    # print(f"Model loaded successfully! Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Check if target frames match
    if 'target_frames' in checkpoint:
        model_target_frames = checkpoint['target_frames']
        if model_target_frames != TARGET_FRAMES:
            print(f"Warning: Model was trained with {model_target_frames} frames, but test script uses {TARGET_FRAMES}")
            print(f"Updating TARGET_FRAMES to match model: {model_target_frames}")
            TARGET_FRAMES = model_target_frames
    
    return feature_extractor, classifier

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
            ai_answer_text = "Eyes Closed" if ai_answer == 1 else "Eyes Open"
            ground_truth_text = "Eyes Closed" if ground_truth == 1 else "Eyes Open"
            correct = "✓" if ai_answer == ground_truth else "✗"
            
            results_data.append({
                'Filename': filename,
                'AI Answer': ai_answer_text,
                'AI Answer (Numeric)': ai_answer,
                'Ground Truth': ground_truth_text,
                'Ground Truth (Numeric)': ground_truth,
                'Confidence': round(confidence, 4),
                'Correct': correct,
                'Eyes Open Probability': round(prediction_probabilities[filename][0], 4),
                'Eyes Closed Probability': round(prediction_probabilities[filename][1], 4)
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
        accuracy_data.append(['', 'Predicted Eyes Open', 'Predicted Eyes Closed'])
        accuracy_data.append(['Actual Eyes Open', test_cm[0,0], test_cm[0,1]])
        accuracy_data.append(['Actual Eyes Closed', test_cm[1,0], test_cm[1,1]])
        accuracy_data.append(['', '', ''])  # Empty row
        
        # Performance metrics
        tn, fp, fn, tp = test_cm.ravel()
        precision_open = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_open = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_open = 2 * (precision_open * recall_open) / (precision_open + recall_open) if (precision_open + recall_open) > 0 else 0
        
        precision_closed = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_closed = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_closed = 2 * (precision_closed * recall_closed) / (precision_closed + recall_closed) if (precision_closed + recall_closed) > 0 else 0
        
        accuracy_data.append(['Performance Metrics', '', ''])
        accuracy_data.append(['', 'Eyes Open', 'Eyes Closed'])
        accuracy_data.append(['Precision', f'{precision_open:.4f}', f'{precision_closed:.4f}'])
        accuracy_data.append(['Recall', f'{recall_open:.4f}', f'{recall_closed:.4f}'])
        accuracy_data.append(['F1-Score', f'{f1_open:.4f}', f'{f1_closed:.4f}'])
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
    """Test the trained PyTorch model on test videos"""
    print("=== TESTING PYTORCH EYE CLOSED DETECTION MODEL ===")
    print(f"Test folder: {TEST_FOLDER}")
    print(f"Excel file: {EXCEL_FILE}")
    print(f"Device: {DEVICE}")
    print(f"Target frames per video: {TARGET_FRAMES}")
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
        feature_extractor, classifier = load_pytorch_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run the PyTorch training script first!")
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
                predictions[video_name] = 0  # Assume eyes open
                prediction_probabilities[video_name] = [1.0, 0.0]  # [prob_open, prob_closed]
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
                
                # Get prediction and probability
                prediction_prob = output.item()
                prediction = 1 if prediction_prob > 0.33 else 0
                
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
        classification_rep = classification_report(y_true, y_pred, target_names=['Eyes Open', 'Eyes Closed'])
        
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
            print(f"  {name}: Predicted={pred} ({'Closed' if pred==1 else 'Open'}), Actual={true}, Confidence={confidence:.3f}")
        
        print(f"\nIncorrect Predictions (showing all):")
        for name, pred, true, probs in incorrect_examples:
            confidence = probs[pred]
            print(f"  {name}: Predicted={pred} ({'Closed' if pred==1 else 'Open'}), Actual={true}, Confidence={confidence:.3f}")
        
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
        print("Please run the PyTorch training script first!")
        return
    
    test_model()

if __name__ == "__main__":
    main() 