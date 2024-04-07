import os
import requests
import subprocess
import torch
import json
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionInpaintPipeline,
)
from utils import Storage
from device import device_id

MODEL_ID = os.environ.get("MODEL_ID", None)
CHECKPOINT_DIR = "/root/.cache/checkpoints"
CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)
CHECKPOINT_CONFIG_URL = os.environ.get("CHECKPOINT_CONFIG_URL", None)
CHECKPOINT_ARGS = os.environ.get("CHECKPOINT_ARGS", None)


def main(
    model_id: str,
    checkpoint_url: str,
    checkpoint_config_url: str,
    checkpoint_args: dict = {},
    path=None,
):
    if not path:
        fname = checkpoint_url.split("/").pop()
        path = os.path.join(CHECKPOINT_DIR, fname)

    if checkpoint_config_url and checkpoint_config_url != "":
        storage = Storage(checkpoint_config_url)
        configPath = CHECKPOINT_DIR + "/" + path + "_config.yaml"
        print(f"Downloading {checkpoint_config_url} to {configPath}...")
        storage.download_file(configPath)

    print("Converting " + path + " to diffusers model " + model_id + "...", flush=True)

    # diffusers defaults
    args = {
        "scheduler_type": "pndm",
    }

    # our defaults
    args.update(
        {
            "checkpoint_path_or_dict": path,
            "original_config_file": configPath if checkpoint_config_url else None,
            "device": device_id,
            "extract_ema": True,
            "from_safetensors": "safetensor" in path.lower(),
        }
    )

    if "inpaint" in path or "Inpaint" in path:
        args.update({"pipeline_class": StableDiffusionInpaintPipeline})

    # user overrides
    args.update(checkpoint_args)

    pipe = download_from_original_stable_diffusion_ckpt(**args)
    pipe.save_pretrained(model_id, safe_serialization=True)


if __name__ == "__main__":
    if CHECKPOINT_URL and CHECKPOINT_URL != "":
        checkpoint_args = json.loads(CHECKPOINT_ARGS) if CHECKPOINT_ARGS else {}
        main(
            MODEL_ID,
            CHECKPOINT_URL,
            CHECKPOINT_CONFIG_URL,
            checkpoint_args=checkpoint_args,
        )
