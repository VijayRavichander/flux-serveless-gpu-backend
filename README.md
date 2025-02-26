# Serverless AutoScaling GPU Backend Application using Modal

This repository contains a Modal application for fine-tuning FLUX image generation models using Dreambooth with LoRA in an serverless GPU Infrastructure. The application provides both synchronous and asynchronous API endpoints for training custom models and generating images. Training a model is 75% cheaper than using [Fal.ai](https://fal.ai/) to train your FLUX models.

## Features

- **Fine-tune FLUX models** with custom images using Dreambooth and LoRA adaptation
- **Synchronous and asynchronous API endpoints** for model training and inference
- **GPU acceleration** using GPUs like H100, A100
- **R2 integration** for model and image storage
- **Callback support** for asynchronous operations

## Prerequisites

- [Modal](https://modal.com/) account (Free $30 Credits per Month)
- R2 storage (compatible with S3) for model and image storage (Ample Free Tier Available)
- Hugging Face account with access token


# Useful Resources and Docs
- [Modal Volume Documentation](https://modal.com/docs/reference/modal.Volume)
- [Modal Secrets Guide](https://modal.com/docs/guide/secrets)
- [Modal Pricing](https://modal.com/pricing)
- [Modal Webinar YouTube Playlist](https://www.youtube.com/playlist?list=PL8YtkOmmxBn_E0Fmp_Cf-PVA8EIHqRIUO)
- [Huggingface Diffusion LoRA Fine Tune](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux.py#L231)


## Setup

1. Make sure you have the Modal CLI installed:
   ```
   pip install modal
   ```

2. Set up the required secrets in Modal:
   - `r2-secret` with the following keys:
     - `R2_ACCESS_KEY_ID`
     - `R2_SECRET_ACCESS_KEY`
     - `R2_ENDPOINT_URL`
     - `R2_BUCKET_NAME`
     - `R2-PUBLIC-URL`
   - `huggingface-secret` with:
     - `HF_TOKEN`

## Configuration

The application uses several configuration files:

- `SharedConfig`: Contains shared configuration like model name
- `AppConfig`: Contains application configuration for inference
- `TrainConfig`: Contains training configuration

## Usage

### Deploying the Application

Deploy the application to Modal:

```
modal deploy flux_inference.py
```

### API Endpoints

#### Health Check
```
GET /health
```

#### Synchronous Model Training
```
POST /train_model_sync
```
Request body:
```json
{
  "steps": 500,
  "trigger_word": "person",
  "images_data_url": "https://example.com/images.zip"
}
```

#### Asynchronous Model Training
```
POST /train_model_async
```
Request body:
```json
{
  "steps": 500,
  "trigger_word": "person",
  "images_data_url": "https://example.com/images.zip",
  "callback_url": "https://your-callback-server.com/webhook"
}
```

#### Synchronous Image Generation
```
POST /infer_sync
```
Request body:
```json
{
  "prompt": "Render a dynamic male image in a futuristic, neon-lit cityscape",
  "lora_url": "https://example.com/model.safetensors"
}
```

#### Asynchronous Image Generation
```
POST /infer_async
```
Request body:
```json
{
  "prompt": "Render a dynamic male image in a futuristic, neon-lit cityscape",
  "lora_url": "https://example.com/model.safetensors", 
  "callback_url": "https://your-callback-server.com/webhook"
}
```


### Local Development

- To run the model training process locally:
  ```
  modal run flux_inference.py
  ```

- To serve the application locally for development:
  ```
  modal serve flux_inference.py
  ```

- To debug the environment:
  ```
  modal shell flux_inference.py
  ```

## Technical Details

### Model Training

The application uses Hugging Face's Diffusers library for Dreambooth training with LoRA adaptations. The training process:

1. Downloads the base FLUX model
2. Loads custom images from the provided URL
3. Trains a LoRA adapter with the specified parameters
4. Uploads the trained model to R2 storage
5. Returns the model URL (or calls the callback URL with the information)

### Image Generation

The inference process:

1. Loads the base FLUX model
2. Downloads and applies LoRA weights if specified
3. Generates images based on the provided prompt
4. Uploads the image to R2 storage
5. Returns the image URL (or calls the callback URL with the information)

## Architecture

- `train()`: Function for training models
- `download_models()`: Function for downloading the base FLUX model
- `download_lora_weights()`: Function for downloading LoRA weights
- `Model` class: Handles loading the model and running inference
- `fastapi_app()`: Creates the FastAPI application with all endpoints

## Dependencies

- diffusers (custom commit: e649678bf55aeaa4b60bd1f68b1ee726278c0304)
- transformers
- torch
- accelerate
- fastapi
- boto3 (for R2/S3 storage)