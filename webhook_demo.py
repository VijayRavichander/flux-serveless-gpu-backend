import httpx
import asyncio
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

web_app = FastAPI()

def process_and_send_image(callback_url: str, text: str, lora_url: str, lora_name: str):
    try:
        download_models.remote(SharedConfig())
        lora_path = None

        if lora_url and lora_name:
            lora_path = download_lora_weights.remote(lora_url, lora_name)

        img_bytes = Model().inference.remote(text, lora_path, config)

        if not img_bytes:
            raise Exception("Image generation failed")

        async def send_image():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    callback_url,
                    files={"file": (f"{text.replace(' ', '_')}.png", img_bytes, "image/png")}
                )
                if response.status_code != 200:
                    raise Exception("Failed to send image to callback URL")

        asyncio.run(send_image())
    except Exception as e:
        print(f"Error processing image: {e}")

@web_app.post("/webhook/infer")
async def infer_webhook(
    background_tasks: BackgroundTasks,
    callback_url: str = Query(..., description="URL to send the generated image to"),
    text: str = Query("Render a dynamic male image of in a futuristic, neon-lit cityscape with bold contrasts and cinematic lighting, evoking energy, creativity, and modern sophistication looking opposite to the camera"),
    lora_url: str = Query(None, description="URL to download LoRA weights from"),
    lora_name: str = Query(None, description="Name to save the LoRA weights as")
):
    background_tasks.add_task(process_and_send_image, callback_url, text, lora_url, lora_name)
    return JSONResponse(content={"message": "Request received, processing in background"})