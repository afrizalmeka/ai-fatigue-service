import httpx

API_URL = "https://vsscustom.report.api.lacak.io/incident-ticket"

async def send_result_to_be(alarm_id: str, device_no: str):
    payload = {
        "alarm_id": alarm_id,
        "device_no": device_no
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json=payload)
        print(f"Sent result to BE: {response.status_code} | {response.text}")
