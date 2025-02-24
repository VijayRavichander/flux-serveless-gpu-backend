from dataclasses import dataclass
from pathlib import Path
import modal


# Setup

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

web_image = image


@dataclass
class SharedConfig:
    """Configuration information shared across project components."""
    # identifier for pretrained models on Hugging Face
    model_name: str = "black-forest-labs/FLUX.1-dev"


@dataclass
class AppConfig(SharedConfig):
    """Configuration information for inference."""
    num_inference_steps: int = 5
    guidance_scale: float = 6

@dataclass
class TrainConfig(SharedConfig):
    """Configuration for the finetuning step."""

    # training prompt looks like `{PREFIX} {INSTANCE_NAME} the {CLASS_NAME} {POSTFIX}`
    prefix: str = "a photo of"
    postfix: str = ""

    # locator for plaintext file with urls for images of target instance
    instance_example_urls_file: str = str(
        Path(__file__).parent / "instance_example_urls.txt"
    )

    # Hyperparameters/constants from the huggingface training example
    resolution: int = 512
    train_batch_size: int = 3
    rank: int = 16  # lora rank
    gradient_accumulation_steps: int = 1
    learning_rate: float = 4e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_steps: int = 500
    checkpointing_steps: int = 1000
    seed: int = 117


huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)


@app.function(
    image=image,
    gpu="A100-80GB",  # fine-tuning is VRAM-heavy and requires a high-VRAM GPU
    volumes={MODEL_DIR: volume},  # stores fine-tuned model
    timeout=1800,  # 30 minutes
    secrets=[huggingface_secret]
    + (
        [
            modal.Secret.from_name(
                "wandb-secret", required_keys=["WANDB_API_KEY"]
            )
        ]
        if USE_WANDB
        else []
    ),
)
def train(instance_example_urls, config):
    volume.commit()

@app.cls(image=image, gpu="A100", secrets=[huggingface_secret])
class Model:
    def __init__(self, config):
        self.config = config


    @modal.enter()
    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import snapshot_download


        snapshot_download(
        self.config.model_name,
        # local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        )

        print("Model Donwloaded....")

        # set up a hugging face inference pipeline using our model
        pipe = DiffusionPipeline.from_pretrained(
            # MODEL_DIR,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        # pipe.load_lora_weights(MODEL_DIR)
        self.pipe = pipe

    @modal.method()
    def inference(self, text, config):
        import io

        image = self.pipe(
            text,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        ).images[0]

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr


@app.function(
    image=web_image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Response

    web_app = FastAPI()
    config = AppConfig()
    

    @web_app.get("/health")
    async def health():
        return "Healthy"

    @web_app.get("/train")
    async def train_model(image_zip, trigger_word):
        


    @web_app.get("/infer")
    async def infer(text: str = "Mount Everest"):
        try:
            model = Model(config)
            img_bytes = model.inference.remote(text, config)
            
            if not img_bytes:
                raise HTTPException(status_code=404, detail="Image generation failed")
            
            return Response(
                content=img_bytes,
                media_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Content-Disposition": f"inline; filename={text.replace(' ', '_')}.png"
                }
            )
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
