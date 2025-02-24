from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

import modal

web_app = FastAPI()
app = modal.App()

image = modal.Image.debian_slim().pip_install("boto3", "fastapi[standard]")


@web_app.post("/foo")
async def foo(request: Request):
    body = await request.json()
    return body


@web_app.get("/bar")
async def bar(arg="world"):
    return HTMLResponse(f"<h1>Hello Fast {arg}!</h1>")


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app