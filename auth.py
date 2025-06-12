import httpx

AUTH_URL = "https://vsscustom.report.api.lacak.io/auth/login"
USERNAME = "ckkim"
PASSWORD = "Ck-kim24"

async def get_token():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            AUTH_URL,
            json={"username": USERNAME, "password": PASSWORD},
            headers={"Content-Type": "application/json"}
        )
        token = response.json().get("token")
        return token