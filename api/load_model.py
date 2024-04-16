import os
from diffusers import pipelines as diffusers_pipelines, AutoPipelineForText2Image
from get_scheduler import get_scheduler, DEFAULT_SCHEDULER
from precision import torch_dtype_from_precision
from device import device
import time

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
PIPELINE = os.getenv("PIPELINE")
USE_DREAMBOOTH = True if os.getenv("USE_DREAMBOOTH") == "1" else False
HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".cache", "diffusers-api")


MODEL_IDS = [
    "CompVis/stable-diffusion-v1-4",
    "hakurei/waifu-diffusion",
    "runwayml/stable-diffusion-inpainting",
    "runwayml/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2"
    "stabilityai/stable-diffusion-2-base"
    "stabilityai/stable-diffusion-2-inpainting",
]

# Loads a model from the cache or downloads it from Hugging Face.
# Author designed it so that downloading can be done in a separate call from the loading to GPU,
# so that they can be independently timed.


def load_model(
    model_id: str,
    load=True,  # If false, will just download the model (if not already cached), and not load it to GPU
    precision=None,
    revision=None,
    status_update_options={},
    pipeline_class=None,
):
    # Convert precision to torch dtype
    torch_dtype = torch_dtype_from_precision(precision)

    # If revision is an empty string, set it to None
    if revision == "":
        revision = None

    # Print the function parameters
    print(
        "load_model",
        {
            "model_id": model_id,
            "load": load,
            "precision": precision,
            "revision": revision,
            "pipeline_class": pipeline_class,
        },
    )

    # If no pipeline class is provided, use the default AutoPipelineForText2Image
    if not pipeline_class:
        pipeline_class = AutoPipelineForText2Image

    # Get the pipeline class, either from the provided pipeline_class or by retrieving it from diffusers_pipelines
    pipeline = pipeline_class if PIPELINE == "ALL" else getattr(diffusers_pipelines, PIPELINE)
    print("pipeline", pipeline_class)

    # Print the model that's being loaded or downloaded
    print(
        ("Loading" if load else "Downloading")
        + " model: "
        + model_id
        + (f" ({revision})" if revision else "")
    )

    # Get the scheduler for the model
    scheduler = get_scheduler(model_id, DEFAULT_SCHEDULER, not load)

    # Get the directory for the model
    model_dir = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(model_dir):
        model_dir = None

    # Start timing the model downloading and loading
    from_pretrained = time.time()

    # Download the model or retrieve from cache
    model = pipeline.from_pretrained(
        model_dir or model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        token=HF_AUTH_TOKEN,
        scheduler=scheduler,
        local_files_only=load,
    )
    from_pretrained = round((time.time() - from_pretrained) * 1000)

    # If the model should be loaded, move it to the GPU
    if load:
        to_gpu = time.time()
        model.to(device)
        to_gpu = round((time.time() - to_gpu) * 1000)
        print(f"Loaded from disk in {from_pretrained} ms, to gpu in {to_gpu} ms")
    else:
        print(f"Downloaded in {from_pretrained} ms")

    # Return the model if it was loaded, otherwise return None
    return model if load else None
