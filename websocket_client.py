import os
import json
import websockets
import asyncio
import httpx
import torch
import requests
import traceback
from api_client import send_result_to_be
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from typing import Tuple

from model.eyes_closed_test_pytorch import (
    ResNetFeatureExtractor,
    LogisticRegressionModel,
    extract_faces_from_video,
    preprocess_faces_for_resnet,
    DEVICE,
    load_pytorch_model
)

from model.yawning_test_pytorch import (
    ResNetFeatureExtractor as YawningExtractor,
    LogisticRegressionModel as YawningClassifier,
    extract_faces_from_video as extract_faces_yawn,
    preprocess_faces_for_resnet as preprocess_faces_yawn,
    DEVICE as DEVICE_YAWN,
    load_pytorch_model as load_yawning_model
)

WS_URL = "wss://vsscustom.report.api.lacak.io/vss-ticket-websocket?email=ciptakridatama.kim@gmail.com"
AUTH_URL = "https://vsscustom.report.api.lacak.io/auth/login"
USERNAME = "ckkim"
PASSWORD = "Ck-kim24"

# Hanya load .env jika tidak dijalankan di GitHub Actions
if not os.getenv("GITHUB_ACTIONS"):
    from dotenv import load_dotenv
    load_dotenv()

# Load once saat file dijalankan
feature_extractor_eye, classifier_eye = load_pytorch_model()
feature_extractor_yawn, classifier_yawn, _, _ = load_yawning_model()

def get_model_by_type(det_type):
    if det_type == "65":
        return feature_extractor_eye, classifier_eye
    
    elif det_type == "66":
        return feature_extractor_yawn, classifier_yawn
    
    else:
        return None, None

async def predict_fatigue(data: dict) -> Tuple[bool, float]:
    det_type = str(data.get("det_tp", ""))
    directory = data.get("directory")
    video_file_name = data.get("video_file_name")

    model, processor = get_model_by_type(det_type)
    if not model or not processor:
        print("❌ det_type tidak dikenali:", det_type)
        return False

    video_url = f"{directory}/{video_file_name}"
    print(f"🔍 Proses video: {video_url}")

    if det_type == "66":
        face_crops = extract_faces_yawn(video_url)
        if not face_crops:
            print("⚠️ Tidak ada wajah terdeteksi (yawning)")
            return False

        face_tensor = preprocess_faces_yawn(face_crops)
        if face_tensor is None:
            print("⚠️ Preprocessing gagal (yawning)")
            return False

        face_tensor = face_tensor.to(DEVICE_YAWN)

        with torch.no_grad():
            features = model(face_tensor)
            output = processor(features).squeeze()
            confidence = output.item() 
            print(f"Confidence yawning: {confidence:.4f}")

        return confidence > 0.5, confidence

    elif det_type == "65":
        face_crops = extract_faces_from_video(video_url)
        if not face_crops:
            print("⚠️ Tidak ada wajah terdeteksi (eyes closed)")
            return False

        face_tensor = preprocess_faces_for_resnet(face_crops)
        if face_tensor is None:
            print("⚠️ Preprocessing gagal (eyes closed)")
            return False

        face_tensor = face_tensor.to(DEVICE)

        with torch.no_grad():
            features = model(face_tensor)
            output = processor(features).squeeze()
            confidence = output.item()
            print(f"Confidence eyes closed: {confidence:.4f}")

        return confidence > 0.33, confidence

    else:
        print("⏭️ Deteksi tidak relevan.")
        return False


async def handle_message(message: str):
    print("📩 Raw WebSocket message:", repr(message), flush=True)

    if not message.strip():
        print("⚠️ Received empty message. Skipping...")
        return

    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error: Message is not valid JSON. Skipped. Content: {repr(message)}")
        return

    if msg.get("messageType") != "FATIGUE":
        print("⏭️ Bukan pesan FATIGUE, dilewati.")
        return

    data = msg.get("data", {})
    det_type = str(data.get("det_tp", ""))

    if det_type not in ["65", "66"]:
        print(f"⏭️ det_type {det_type} bukan 65/66, dilewati.")
        return

    result, confidence = await predict_fatigue(data)
    if result:
        alarm_id = data.get("alarm_id")
        device_no = data.get("device_no")
        print(f"📤 Mengirim hasil ke BE: det_type={det_type}, confidence={confidence:.4f}")
        await send_result_to_be(alarm_id, device_no, confidence)


async def get_token():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            AUTH_URL,
            json={"username": USERNAME, "password": PASSWORD},
            headers={"Content-Type": "application/json"}
        )
        token = response.json().get("token")
        print("Bearer token retrieved")
        return token

async def start_websocket_listener():
    while True:
        try:
            token = await get_token()
            headers = [("Authorization", f"Bearer {token}")]

            async with websockets.connect(
                WS_URL,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=60
            ) as websocket:
                print("✅ Connected to WebSocket")
                while True:
                    msg = await websocket.recv()
                    await handle_message(msg)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"🛑 WebSocket connection closed: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"⚠️ WebSocket error: {e}. Retrying in 5s...")
            traceback.print_exc()
            await asyncio.sleep(5)
