# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
from loadModel import loadModel, MODEL_IDS
from utils import Storage
import subprocess
from pathlib import Path
from convert_to_diffusers import main as convert_to_diffusers
from download_checkpoint import main as download_checkpoint
from status import status
import asyncio

USE_DREAMBOOTH = os.environ.get("USE_DREAMBOOTH")
HF_AUTH_TOKEN = os.environ.get("HF_AUTH_TOKEN")
RUNTIME_DOWNLOADS = os.environ.get("RUNTIME_DOWNLOADS")

HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".cache", "diffusers-api")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)


# i.e. don't run during build
async def send_status_update(process_name: str, status: str, payload: dict = {}, options: dict = {}):
    if RUNTIME_DOWNLOADS:
        from send import send_status_update as _send_status_update

        await _send_status_update(process_name, status, payload, options)


def get_model_filename(model_id: str, model_revision):
    filename = "models--" + model_id.replace("/", "--")
    if model_revision:
        filename += "--" + model_revision
    return filename

# Download model from Hugging Face


async def download_model(
    model_url=None,
    model_id=None,
    model_revision=None,
    checkpoint_url=None,
    checkpoint_config_url=None,
    hf_model_id=None,
    model_precision=None,
    status_update_options={},
    pipeline_class=None,
):
    # Print the function parameters
    print(
        "download_model",
        {
            "model_url": model_url,  # URL for a Diffusers .tar.zst model not on HuggingFace
            "model_id": model_id,  # Can also be a Hugging Face model ID, e.g. "CompVis/stable-diffusion-v1-4"
            "model_revision": model_revision,
            "hf_model_id": hf_model_id,  # Hugging Face model ID, e.g. "CompVis/stable-diffusion-v1-4"
            "checkpoint_url": checkpoint_url,  # URL for a checkpoint file
            "checkpoint_config_url": checkpoint_config_url,
        },
    )

    # If provided a URL for a .tar.zst model, download and extract it
    if model_url:
        # Normalize the model ID
        model_filename = get_model_filename(model_id, model_revision)
        print({"model_filename": model_filename})

        # Get the filename from the model URL
        filename = model_url.split("/").pop()
        if not filename:
            filename = model_filename + ".tar.zst"

        # Define the path where the model will be saved
        model_file = os.path.join(MODELS_DIR, filename)

        # Create an appropriate Storage object for the model, depending on whether the URL is for S3 or HTTP
        storage = Storage(
            model_url, default_path=model_filename + ".tar.zst", status=status
        )

        # If the model exists in S3, download and extract it
        exists_in_s3 = storage.file_exists()
        if exists_in_s3:
            model_dir = os.path.join(MODELS_DIR, model_filename)
            print("model_dir", model_dir)
            await asyncio.to_thread(storage.download_and_extract, model_file, model_dir)
        else:
            print("Model not found in S3")
            raise Exception(f"Model with URL {model_url} not found in S3")

    # If provided a non-Diffusers checkpoint URL, download it and convert it to a Diffusers model
    elif checkpoint_url:
        path = download_checkpoint(checkpoint_url)
        convert_to_diffusers(
            model_id=model_id,
            checkpoint_url=checkpoint_url,
            checkpoint_config_url=checkpoint_config_url,
            path=path,
        )

    # If provided a Hugging Face model ID, download it (really won't because we already downloaded it at build time)
    else:
        # If no Hugging Face model ID is provided, use the model_id (presuming it's Hugging Face compatible)
        hf_model_id = hf_model_id or model_id

        # Do a dry run of loading the huggingface model, which will have already downloaded weights at build time
        loadModel(
            model_id=hf_model_id,
            load=False,
            precision=model_precision,
            revision=model_revision,
            pipeline_class=pipeline_class,
        )


if __name__ == "__main__":
    asyncio.run(
        download_model(
            model_url=os.environ.get("MODEL_URL"),
            model_id=os.environ.get("MODEL_ID"),
            hf_model_id=os.environ.get("HF_MODEL_ID"),
            model_revision=os.environ.get("MODEL_REVISION"),
            model_precision=os.environ.get("MODEL_PRECISION"),
            checkpoint_url=os.environ.get("CHECKPOINT_URL"),
            checkpoint_config_url=os.environ.get("CHECKPOINT_CONFIG_URL"),
        )
    )
