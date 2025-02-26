from dataclasses import dataclass
from pathlib import Path
import modal
from modal import Volume, Image, Mount
from config import SharedConfig, AppConfig, TrainConfig
from utils import load_images, upload_model_to_r2, upload_image_to_r2

#### ------------------------- Setup & Image ------------------------- 
app = modal.App(name="example-dreambooth-flux")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "accelerate==0.31.0",
    "datasets~=2.13.0",
    "fastapi[standard]==0.115.4",
    "ftfy~=6.1.0",
    "huggingface-hub==0.26.2",
    "hf_transfer==0.1.8",
    "numpy<2",
    "peft==0.11.1",
    "pydantic==2.9.2",
    "sentencepiece>=0.1.91,!=0.1.92",
    "smart_open~=6.4.0",
    "starlette==0.41.2",
    "transformers~=4.41.2",
    "torch~=2.2.0",
    "torchvision~=0.16",
    "triton~=2.2.0",
    "wandb==0.17.6",
    "boto3==1.33.6"
)

GIT_SHA = (
    "e649678bf55aeaa4b60bd1f68b1ee726278c0304"  # specify the commit to fetch
)

image = (
    image.apt_install("git")
    # Perform a shallow fetch of just the target `diffusers` commit, checking out
    # the commit in the container's home directory, /root. Then install `diffusers`
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add origin https://github.com/huggingface/diffusers",
        f"cd /root && git fetch --depth=1 origin {GIT_SHA} && git checkout {GIT_SHA}",
        "cd /root && pip install -e .",
    )
)
image = image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"}  # turn on faster downloads from HF
)

image = image.add_local_python_source("config").add_local_python_source("utils")

web_image = image

#### ------------------------- x ------------------------- 

### ------------------------- Secrets & Volumes -------------------------

# Add R2 secret
r2_secret = modal.Secret.from_name(
    "r2-secret", required_keys=["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "R2_BUCKET_NAME"]
)

huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)

volume = modal.Volume.from_name(
    "flux-models", create_if_missing=True
)


MODEL_DIR = "/model"
LORA_DIR = "/lora_adapters"
MODEL_ID = SharedConfig().model_name

### ------------------------- x -------------------------

###  ------------------------- Train -------------------------
@app.function(
    image=image,
    gpu="H100", # fine-tuning is VRAM-heavy and requires a high-VRAM GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=1200,  # 20 minutes
    secrets=[huggingface_secret, r2_secret],
)
def train(images_data_url, steps, trigger_word, uuid):
    from pathlib import Path
    import subprocess
    import os
    from accelerate.utils import write_basic_config

    try:
        # Train Config
        config = TrainConfig()

        # set up hugging face accelerate library for fast training
        write_basic_config(mixed_precision="bf16")

        #load data locally
        img_path = load_images(images_data_url)
        
        if not img_path:
            raise Exception("Invalid Image URL")

        instance_phrase = f"{trigger_word} the person"
        prompt = f"{config.prefix} {instance_phrase} {config.postfix}".strip()
        
        save_path = Path(f'{MODEL_DIR}/trained_models/{uuid}')
        save_path.mkdir(parents=True, exist_ok=True)

        def _exec_subprocess(cmd: list[str]):
            """Executes subprocess and prints log to terminal while subprocess is running."""
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            with process.stdout as pipe:
                for line in iter(pipe.readline, b""):
                    line_str = line.decode()
                    print(f"{line_str}", end="")

            if exitcode := process.wait() != 0:
                raise subprocess.CalledProcessError(exitcode, "\n".join(cmd))
        
        
        print("launching dreambooth training script")
        _exec_subprocess(
            [
                "accelerate",
                "launch",
                "examples/dreambooth/train_dreambooth_lora_flux.py",
                "--mixed_precision=bf16",  # half-precision floats most of the time for faster training
                f"--pretrained_model_name_or_path={config.model_name}",
                f"--instance_data_dir={img_path}",
                f"--output_dir={save_path}",
                f"--instance_prompt={prompt}",
                f"--resolution={config.resolution}",
                f"--train_batch_size={config.train_batch_size}",
                f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
                f"--learning_rate={config.learning_rate}",
                f"--lr_scheduler={config.lr_scheduler}",
                f"--lr_warmup_steps={config.lr_warmup_steps}",
                f"--max_train_steps={steps}",
                f"--checkpointing_steps={config.checkpointing_steps}",
                f"--seed={config.seed}",  # increased reproducibility by seeding the RNG 
                f"--hub_model_id=flux_lora_modal_test"
            ])
        
        # Upload model weights to R2 after training
        print("Uploading model weights to R2 storage...")
        r2_url = upload_model_to_r2(save_path, str(uuid))
        print(f"Model weights uploaded to R2. Access at: {r2_url}")
        print("ALL DONE!!!!!!!!!!!!!!!!!")
        return r2_url

    except Exception as e:
        return None
#  ------------------------- x -------------------------

#  ------------------------- Download Weights -------------------------

# Function to download Flux weights - runs on CPU 
@app.function(
    volumes={MODEL_DIR: volume},
    image=image,
    secrets=[huggingface_secret],
    timeout=600,  # 10 minutes
)
def download_models(config):
    import torch
    from diffusers import DiffusionPipeline
    from huggingface_hub import snapshot_download
    from pathlib import Path
    import os
    import time
    start = time.time()
    print("Downloading the Flux Model")
    snapshot_download(
        config.model_name,
        local_dir=f"{MODEL_DIR}/.cache/huggingface/",
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )
    end = time.time()
    print(f"Model Downloaded in {end - start}")
    # DiffusionPipeline.from_pretrained(
    #         "black-forest-labs/FLUX.1-dev",
    #         torch_dtype=torch.bfloat16,
    #         cache_dir = f"{MODEL_DIR}/.cache/"
    # )

    volume.commit()

#  ------------------------- x -------------------------


#  ------------------------- LoRA Weights Download -------------------------

# Function to download LoRA weights - runs on CPU web server
@app.function(
    image=web_image,
    volumes={MODEL_DIR: volume}
)
def download_lora_weights(lora_url, lora_name):
    """Download LoRA weights from URL and save to the shared volume"""
    import requests
    import os
    
    os.makedirs(f"{MODEL_DIR}{LORA_DIR}", exist_ok=True)
    lora_path = os.path.join(f"{MODEL_DIR}{LORA_DIR}", f"{lora_name}.safetensors")
    
    # Check if we already have the file
    if os.path.exists(lora_path):
        return lora_path
    
    # Download the file
    response = requests.get(lora_url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Save the file
    with open(lora_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return lora_path


#  ------------------------- x -------------------------

@app.cls(image=image, gpu="A100", secrets=[huggingface_secret], volumes={MODEL_DIR: volume})
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline
        import time

        start = time.time()
        print("Loading Flux Model")

        # self.pipe = DiffusionPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev",
        #     torch_dtype=torch.bfloat16,
        #     cache_dir = f"{MODEL_DIR}/.cache/"
        # ).to("cuda")

        self.pipe = DiffusionPipeline.from_pretrained(
            f"{MODEL_DIR}/.cache/huggingface",
            cache_dir = f"{MODEL_DIR}/.cache/huggingface",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        end = time.time()
        print(f"Loaded Flux Model in {end - start} seconds")


        volume.commit()

    @modal.exit()
    def del_dir(self):
        volume.remove_file(f'/.cache/huggingface/', recursive=True)
        volume.remove_file(f'/lora_adapters/', recursive=True)
        volume.remove_file(f'/trained_models/', recursive=True)
        print("Shutting Down")

    @modal.method()
    def inference(self, prompt, lora_path, config):
        import io
        import os   

        volume.reload()

        if lora_path:
            lora_path = Path(lora_path)
            # Add the LoRA Adapters
            print("Loggging: Trying to download LoRA Weights")
            self.pipe.load_lora_weights(lora_path)
            print("Loggging: Downloaded LoRA Weights")


        print("Running Inference")
        image = self.pipe(
            prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]
        print("Inference Successful")

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr

@app.function(
    image=web_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    secrets=[r2_secret],
    timeout= 60 * 40
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Response, Query, BackgroundTasks, Request
    from fastapi.responses import JSONResponse
    import httpx
    import asyncio
    import uuid
    import os
    from pathlib import Path
    from pydantic import BaseModel, HttpUrl
    import time

    class TrainModelSyncRequest(BaseModel):
        steps: int = 500
        trigger_word: str
        images_data_url: HttpUrl
    
    class TrainModelAsyncRequest(TrainModelSyncRequest):
        callback_url: HttpUrl

    class InferenceSyncRequest(BaseModel):
        prompt: str
        lora_url: HttpUrl

    class InferenceAsyncRequest(InferenceSyncRequest):
        callback_url: HttpUrl


    web_app = FastAPI()
    config = AppConfig()
    train_config = TrainConfig()

    def generate_image_background(callback_url: str, text: str, lora_url: str, lora_name: str):
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


    def train_model_background(callback_url: str, images_data_url, steps: int, trigger_word: str, id: str):
        try:
            model_url = train.remote(images_data_url, steps, trigger_word, id)

            if not model_url:
                raise Exception("Model Trained Failed")
            
            response = {
                "config": {
                    "id": str(id),
                    "steps": steps,
                    "rank": train_config.max_train_steps
                }, 
                "diffusers_lora_file": {
                    "url": model_url
                }
            }

            async def send_model(response):
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        callback_url,
                        json = response
                    )
                    if response.status_code != 200:
                        raise Exception("Failed to send image to callback URL")
            
            asyncio.run(send_model(response))

        except Exception as e:
            return HTTPException(status_code=500, detail=str(e))
        
    @web_app.get("/health")
    async def health():
        return "Healthy"

    @web_app.post("/train_model_sync")
    async def train_model_sync(request: TrainModelSyncRequest):
        try:
            res_start = time.time()
            steps = request.steps
            trigger_word = request.trigger_word
            images_data_url = request.images_data_url

            if not images_data_url or not trigger_word:
                raise HTTPException(status_code=400, detail="Missing zipped_url parameter")
            
            id = uuid.uuid4()
            gpu_start = time.time()
            model_url = train.remote(images_data_url, steps, trigger_word, id)
            gpu_end = time.time()

            print(f"Time Taken to Train {steps}: {gpu_end - gpu_start} seconds")

            if not model_url:
                raise Exception("Model Training Failed")
            
            response = {
                "config": {
                    "id": str(id),
                    "steps": steps,
                    "rank": train_config.rank
                }, 
                "diffusers_lora_file": {
                    "url": model_url
                }
            }
            res_end = time.time()
            print(f"Time Take to the entire Request: {res_end - res_start} seconds")
            return JSONResponse(content=response)
        
        except Exception as e:
            return HTTPException(status_code=500, detail=str(e))

    @web_app.post("/train_model_async")
    async def train_model_async(background_tasks: BackgroundTasks,
        request: TrainModelAsyncRequest):
        try:
            steps = request.steps
            trigger_word = request.trigger_word
            images_data_url = request.images_data_url
            callback_url = request.callback_url

            if not images_data_url or not trigger_word:
                raise HTTPException(status_code=400, detail="Missing zipped_url parameter")
            
            id = uuid.uuid4()
            
            background_tasks.add_task(train_model_background, callback_url, images_data_url, steps, trigger_word, id)

            response = {"id": id}
            return JSONResponse(content=response)
        
        except Exception as e:
            return HTTPException(status_code=500, detail=str(e))
    
    @web_app.post("/infer_async")
    async def infer_async(background_tasks: BackgroundTasks, request: InferenceAsyncRequest):
        
        try:
            # Validate the Query
            lora_url = request.lora_url
            prompt = request.prompt
            callback_url = request.callback_url

            id = uuid.uuid4()

            # Add to Background Task 
            background_tasks.add_task(generate_image_background, callback_url, prompt, lora_url, id)

            response = {"id": id}
            return JSONResponse(content=response)

                
        except Exception as e:
            return HTTPException(status_code=500, detail=str(e))

    @web_app.post("/infer_sync")
    async def infer_async(request: InferenceSyncRequest):
        
        try:
            lora_url = request.lora_url
            prompt = request.prompt

            download_models.remote(SharedConfig())
            
            id = uuid.uuid4()
            if lora_url:
                lora_path = download_lora_weights.remote(lora_url, id)
            
            img_bytes = Model().inference.remote(prompt, lora_path, config) 
            
            # Push the Image to Object Store
            image_url = upload_image_to_r2(img_bytes, id)

            if not img_bytes:
                raise HTTPException(status_code=404, detail="Image generation failed")
            
            response = {
                "images": [
                    {
                         "url": image_url,
                         "content_type": "image/jpeg"
                    }
                ],
                "prompt": prompt
            }
            return JSONResponse(content= response)
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app



# - `modal run dreambooth_app.py` will train the model. Change the `instance_example_urls_file` to point to your own pet's images.
# - `modal serve dreambooth_app.py` will [serve](https://modal.com/docs/guide/webhooks#developing-with-modal-serve) the Gradio interface at a temporary location. Great for iterating on code!
# - `modal shell dreambooth_app.py` is a convenient helper to open a bash [shell](https://modal.com/docs/guide/developing-debugging#interactive-shell) in our image. Great for debugging environment issues.


# @app.local_entrypoint()
# def run(  # add more config params here to make training configurable
#     max_train_steps: int = 250,
# ):
#     print("🎨 loading model")
#     download_models.remote(SharedConfig())
#     print("🎨 setting up training")
#     config = TrainConfig(max_train_steps=max_train_steps)
#     instance_example_urls = (
#         Path(TrainConfig.instance_example_urls_file).read_text().splitlines()
#     )
#     train.remote(instance_example_urls, config)
#     print("🎨 training finished")
