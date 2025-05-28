import asyncio
from fastapi import FastAPI
from websocket_client import start_websocket_listener

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_websocket_listener())

@app.get("/")
def status():
    return {"status": "AI Fatigue Service is running"}
