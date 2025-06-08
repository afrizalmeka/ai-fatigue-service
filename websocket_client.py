import os
import json
import websockets
import asyncio
import httpx
from api_client import send_result_to_be
from transformers import AutoModelForVideoClassification, AutoImageProcessor
from PIL import Image
import torch
import requests
from io import BytesIO
from huggingface_hub import login
from dotenv import load_dotenv
import os
import traceback

WS_URL = "wss://vsscustom.report.api.lacak.io/vss-ticket-websocket?email=ckbmb@test.com"
AUTH_URL = "https://vsscustom.report.api.lacak.io/auth/login"
USERNAME = "ckkim"
PASSWORD = "Ck-kim24"

# Hanya load .env jika tidak dijalankan di GitHub Actions
if not os.getenv("GITHUB_ACTIONS"):
    from dotenv import load_dotenv
    load_dotenv()

hf_token = os.getenv("HF_TOKEN_R_MODEL")

if not hf_token:
    raise EnvironmentError("❌ HF_TOKEN_R_MODEL is not set.")

login(token=hf_token)

model_yawn = AutoModelForVideoClassification.from_pretrained(
    "afrizalmeka/yawning-model", token=hf_token
)
proc_yawn = AutoImageProcessor.from_pretrained(
    "afrizalmeka/yawning-model", token=hf_token
)

model_eye = AutoModelForVideoClassification.from_pretrained(
    "afrizalmeka/eyes-closed-model", token=hf_token
)
proc_eye = AutoImageProcessor.from_pretrained(
    "afrizalmeka/eyes-closed-model", token=hf_token
)



def get_model_by_type(det_type):
    if det_type == "66":
        return model_yawn, proc_yawn
    elif det_type == "65":
        return model_eye, proc_eye
    else:
        return None, None

async def predict_fatigue(data: dict) -> bool:
    det_type = str(data.get("det_tp", ""))  # <-- perubahan di sini
    directory = data.get("directory")
    video_file_name = data.get("video_file_name")

    model, processor = get_model_by_type(det_type)
    if not model or not processor:
        print("det_type tidak dikenali:", det_type)
        return False

    # Ambil thumbnail/frame dari video untuk testing (sementara pakai gambar .jpg jika tersedia)
    image_url = f"{directory}/{video_file_name.replace('.mp4', '.jpg')}"
    print("Ambil image dari:", image_url)

    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print("Gagal download atau baca gambar:", e)
        return False

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    predicted = outputs.logits.argmax(-1).item() == 1
    print(f"Hasil prediksi (det_type {det_type}):", predicted)
    return predicted

async def handle_message(message: str):
    print("📩 Raw WebSocket message:", repr(message), flush=True)  # tampilkan isi asli pesan apa adanya

    if not message.strip():
        print("⚠️ Received empty message. Skipping...")
        return

    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error: Message is not valid JSON. Skipped. Content: {repr(message)}")
        return

    # Hanya proses pesan bertipe FATIGUE
    if msg.get("messageType") != "FATIGUE":
        print("⏭️ Bukan pesan FATIGUE, dilewati.")
        return

    data = msg.get("data", {})
    det_type = str(data.get("det_tp", ""))  # sebelumnya: det_type

    if det_type not in ["65", "66"]:
        print(f"⏭️ det_type {det_type} bukan 65/66, dilewati.")
        return

    if await predict_fatigue(data):
        alarm_id = data.get("alarm_id")
        device_no = data.get("device_no")
        await send_result_to_be(alarm_id, device_no)


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

            async with websockets.connect(WS_URL, extra_headers=headers) as websocket:
                print("Connected to WebSocket")
                while True:
                    msg = await websocket.recv()
                    await handle_message(msg)
        except Exception as e:
            print(f"WebSocket error: {e}. Retrying in 5s...")
            traceback.print_exc()  # Cetak traceback lengkap
            await asyncio.sleep(5)
