from pathlib import Path


def print_curr_dir(path):
    import os
    from pathlib import Path

    print(f"Listing all files in '{path}':")
    print(Path.cwd())
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # Print the full path relative to the starting directory
            rel_path = os.path.relpath(file_path, path)
            print(Path.cwd())
            print(f"- {rel_path}")

def load_images(zipped_images_url: str) -> Path:
    import zipfile
    import requests
    from io import BytesIO

    img_path = Path("/img")
    img_path.mkdir(parents=True, exist_ok=True)

    # Download the zip file
    response = requests.get(zipped_images_url)
    response.raise_for_status()  # Ensure we successfully downloaded the file

    # Extract images from the zip
    with zipfile.ZipFile(BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(img_path)

    print(f"Extracted {len(zip_ref.namelist())} images to {img_path}")
    full_path = Path.cwd()
    print(full_path)
    print_curr_dir(img_path)
    return img_path

def upload_to_r2(local_model_path, model_id):
    import boto3
    import os
    from pathlib import Path
    
    # Get R2 credentials from the environment
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    endpoint_url = os.environ["R2_ENDPOINT_URL"]
    bucket_name = os.environ["R2_BUCKET_NAME"]
    
    # Create S3 client (R2 is S3-compatible)
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    # Upload all files from the local model path
    for file_path in Path(local_model_path).rglob("*"):
        if file_path.is_file():
            # Create relative path for S3 key
            relative_path = file_path.relative_to(local_model_path)
            s3_key = f"models/{model_id}/{relative_path}"
            
            print(f"Uploading {file_path} to {s3_key}")
            s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key
            )
    
    print(f"Model successfully uploaded to R2 bucket: {bucket_name}/models/{model_id}")
    return f"{endpoint_url}/{bucket_name}/trained_models/{model_id}"