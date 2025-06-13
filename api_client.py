import httpx
from auth import get_token  # import token login

API_URL = "https://vsscustom.report.api.lacak.io/incident-ticket"

async def send_result_to_be(alarm_id: str, device_no: str, confidence: float):
    payload = {
        "alarm_id": alarm_id,
        "device_no": device_no,
        "alarm_type": "FATIGUE_ALERT",
        "ai_confidence_level": round(confidence * 100, 2)
    }

    token = await get_token()  # get fresh token every call

    print(f"📦 Payload yang dikirim ke BE:\n{payload}")
    print(f"🔐 Bearer Token: {token[:10]}...")  # Hanya tampilkan sebagian token untuk keamanan

    async with httpx.AsyncClient() as client:
        response = await client.post(
            API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
        )
        print(f"Sent result to BE: {response.status_code} | {response.text}")