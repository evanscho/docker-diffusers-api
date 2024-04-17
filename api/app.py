import asyncio
import torch

from diffusers import __version__
import base64
from io import BytesIO
import PIL
import json
from load_model import load_model
from event_timings_tracker import EventTimingsTracker
import os
import numpy as np
import skimage
import skimage.measure
from get_scheduler import get_scheduler, SCHEDULERS
from get_pipeline import (
    get_pipeline_class,
    get_pipeline_for_model,
    list_available_pipelines,
    clear_pipelines,
)

import requests
from download import download_model, get_model_filename
import traceback
from precision import MODEL_REVISION, MODEL_PRECISION
from device import device, device_id, device_name
from utils import Storage
from hashlib import sha256
from threading import Timer
import extras
from file_transfer_progress import perform_while_tracking_progress

# from torch import autocast
# import re

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)

from lib.textual_inversions import handle_textual_inversions
from lib.prompts import prepare_prompts
from lib.vars import (
    RUNTIME_DOWNLOADS,
    USE_DREAMBOOTH,
    MODEL_ID,
    PIPELINE,
    MODELS_DIR,
)

if USE_DREAMBOOTH:
    from train_dreambooth import TrainDreamBooth
print(os.environ.get("USE_PATCHMATCH"))
if os.environ.get("USE_PATCHMATCH") == "1":
    from PyPatchMatch import patch_match

# Disable gradient computation in PyTorch by default since it's memory-intensive; only enable it when training
torch.set_grad_enabled(False)


async def init():
    """Init is run on server startup"""
    # Load your model to GPU as a global variable here using the variable name "model"
    global model  # needed for banana optimizations; TODO ESS: remove this line and subsequent uses until its re-instantiation in inference()

    # Send a status update indicating that the initialization has started

    await EventTimingsTracker().send_event_update(
        "init",
        "start",
        {
            "device": device_name,
            # HOSTNAME is set automatically in Linux; in a container it's the container's ID
            "hostname": os.getenv("HOSTNAME"),
            "model_id": MODEL_ID,
            "diffusers": __version__,
        },
    )

    global last_model_filename
    last_model_filename = None

    # If runtime downloads are not enabled, load the model at init time
    if not RUNTIME_DOWNLOADS:
        model = load_model(
            model_id=MODEL_ID,
            load=True,
            precision=MODEL_PRECISION,
            revision=MODEL_REVISION,
        )
    else:
        model = None

    # Send a status update indicating that the initialization is done
    await EventTimingsTracker().send_event_update("init", "done")


def decodeBase64Image(imageStr: str, name: str) -> PIL.Image:
    """Function to decode a base64-encoded image"""
    image = PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image


def getFromUrl(url: str, name: str) -> PIL.Image:
    """Function to download an image from a URL"""
    response = requests.get(url)
    image = PIL.Image.open(BytesIO(response.content))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image


def truncateInputs(inputs: dict):
    """Function to truncate inputs to a manageable size for logging or debugging"""
    inputs_copy = inputs.copy()
    if "modelInputs" in inputs_copy:
        # Create a shallow copy of the modelInputs dictionary since even with the shalow copy of 'inputs',
        # the original modelInputs dictionary is still a reference to the original modelInputs dictionary.
        model_inputs = inputs_copy["modelInputs"] = inputs_copy["modelInputs"].copy()

        # Truncate all image inputs to the first 6 characters
        for item in ["init_image", "mask_image", "image", "input_image"]:
            if item in model_inputs:
                model_inputs[item] = model_inputs[item][0:6] + "..."
        if "instance_images" in model_inputs:
            model_inputs["instance_images"] = list(
                map(lambda str: str[0:6] + "...", model_inputs["instance_images"])
            )
    return inputs_copy


def calculate_memory_usage():
    # TODO: move to device.py
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    return 0


async def prepare_status_update_options(pipeline_run, call_inputs, response, status_instance=None):
    status_update_options = {}

    if call_inputs.get("SEND_URL", None):
        status_update_options.update({"SEND_URL": call_inputs.get("SEND_URL")})
    if call_inputs.get("SIGNING_KEY", None):
        status_update_options.update({"SIGNING_KEY": call_inputs.get("SIGNING_KEY")})

    # Add options necessary for streaming updates
    if response and status_instance:
        status_update_options.update({"response": response})
        status_update_options.update({"status_instance": status_instance})

    # Necessary for event timings tracking
    status_update_options.update({"pipeline_run": pipeline_run})

    # Needed to cancel the PercentageCompleteStatusSender from another thread, and to run coroutines
    status_update_options.update({"event_loop": asyncio.get_event_loop()})

    return status_update_options


def env_variables_masked():
    """Masks sensitive environmental variables"""
    env_vars = os.environ.copy()
    for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_AUTH_TOKEN"]:
        if var in env_vars:
            env_vars[var] = "***"
    return env_vars


def print_inputs_and_env_variables(inputs):
    print()  # Add a newline before the inputs for spacing
    print("Environment variables:")
    print(env_variables_masked())
    print()
    print("Inputs:")
    print(json.dumps(truncateInputs(inputs), indent=2))


last_attn_procs = None
last_lora_weights = None
cross_attention_kwargs = None


# This function is used to perform inference (or training!) on the model with the provided inputs.
# It's run with every server call.
# It's an asynchronous function, meaning it's designed to handle multiple requests at the same time.
async def inference(all_inputs: dict, response, status_instance=None) -> dict:
    global model
    global last_model_filename
    global last_attn_procs
    global last_lora_weights
    global cross_attention_kwargs

    # Start new pipeline run (for timing each step + debug data) since starting a new inference
    pipeline_run = EventTimingsTracker().start_new_run()
    print_inputs_and_env_variables(all_inputs)

    # Extract model inputs and call inputs from the all_inputs dictionary
    model_inputs = all_inputs.get("modelInputs", None)
    call_inputs = all_inputs.get("callInputs", None)

    result = {"$meta": {}}  # Initialize the result dictionary

    # Prepare options for status update, including the URL to send the status updates to
    status_update_options = await prepare_status_update_options(pipeline_run, call_inputs, response, status_instance)
    status_instance = status_update_options.get("status_instance", None)

    # If either model inputs or call inputs are missing, return an error
    if model_inputs == None or call_inputs == None:
        return {
            "$error": {
                "code": "INVALID_INPUTS",
                "message": "Expecting on object like { modelInputs: {}, callInputs: {} } but got "
                + json.dumps(all_inputs),
            }
        }

    # Extract the start request ID from the call inputs
    startRequestId = call_inputs.get("startRequestId", None)

    # If the call inputs specify an 'extra' coroutine to use, call that extra asynchronously
    use_extra = call_inputs.get("use_extra", None)
    if use_extra:
        extra = getattr(extras, use_extra, None)
        if not extra:
            return {
                "$error": {
                    "code": "NO_SUCH_EXTRA",
                    "message": f'Requested "{use_extra}", available: "{", ".join(extras.keys())}"',
                }
            }
        return await extra(
            model_inputs,
            call_inputs,
            status_update_options=status_update_options,
            startRequestId=startRequestId,
        )

    # Extract the model ID from the call inputs or fall back to the MODEL_ID env var
    model_id = call_inputs.get("MODEL_ID", None)
    if not model_id:
        if not MODEL_ID:
            return {
                "$error": {
                    "code": "NO_MODEL_ID",
                    "message": "No callInputs.MODEL_ID specified, nor was MODEL_ID env var set.",
                }
            }
        model_id = MODEL_ID
        # Update metadata with the new model ID
        result["$meta"].update({"MODEL_ID": MODEL_ID})

    # Create the filename for the model
    model_filename = get_model_filename(model_id, MODEL_REVISION)

    # If runtime downloads are enabled, download the model and related files
    if RUNTIME_DOWNLOADS:
        # Extract various parameters from the call inputs
        hf_model_id = call_inputs.get("HF_MODEL_ID", None)
        model_revision = call_inputs.get("MODEL_REVISION", None)
        model_precision = call_inputs.get("MODEL_PRECISION", None)
        checkpoint_url = call_inputs.get("CHECKPOINT_URL", None)
        checkpoint_config_url = call_inputs.get("CHECKPOINT_CONFIG_URL", None)

        # Get the path for the model (where it may already be cached, or where it will be downloaded to)
        model_dir = os.path.join(MODELS_DIR, model_filename)

        # Extract the pipeline name from the call inputs, and get the corresponding pipeline class
        pipeline_name = call_inputs.get("PIPELINE", None)
        if pipeline_name:
            pipeline_class = get_pipeline_class(pipeline_name)

        print("model_filename", model_filename, "last_model_filename", last_model_filename)
        print("model_dir", model_dir, "os.path.isdir(model_dir)", os.path.isdir(model_dir))
        # If the new model filename is not the same as the last one, and it's not already on disk, download and load the new model
        if model_filename != last_model_filename and not os.path.isdir(model_dir):
            model_url = call_inputs.get("MODEL_URL", None)
            await download_model(
                model_id=model_id,
                model_url=model_url,
                model_revision=model_revision,
                checkpoint_url=checkpoint_url,
                checkpoint_config_url=checkpoint_config_url,
                hf_model_id=hf_model_id,
                model_precision=model_precision,
                status_update_options=status_update_options,
                pipeline_class=pipeline_class if pipeline_name else None,
            )

            # Clear the pipeline cache when changing the loaded model, as pipelines include references to the model and would
            # therefore prevent memory being reclaimed after unloading the previous model.
            clear_pipelines()

            # Reset the cross attention arguments
            cross_attention_kwargs = None

            # If a model is already loaded, move it to the CPU to avoid a memory leak
            if model:
                model.to("cpu")

            await pipeline_run.send_event_update("load_model", "start", {"startRequestId": startRequestId}, status_update_options)

            # Load the model
            model = perform_while_tracking_progress(load_model, {
                "model_id": model_id,
                "load": True,
                "precision": model_precision,
                "revision": model_revision,
                "status_update_options": status_update_options,
                "pipeline_class": pipeline_class if pipeline_name else None,
            }, "load_model", status_instance)

            await pipeline_run.send_event_update("load_model", "done", {"startRequestId": startRequestId}, status_update_options)
            last_model_filename = model_filename
            last_attn_procs = None
            last_lora_weights = None

    if MODEL_ID == "ALL":
        if model_filename != last_model_filename:
            clear_pipelines()
            cross_attention_kwargs = None
            model = load_model(
                model_id,
                load=True,
                precision=model_precision,
                revision=model_revision,
                status_update_options=status_update_options)
            last_model_filename = model_filename
    else:
        # Make sure the requested model is the one we have (if runtime downloads are not enabled)
        if model_id != MODEL_ID and not RUNTIME_DOWNLOADS:
            return {
                "$error": {
                    "code": "MODEL_MISMATCH",
                    "message": f'Model "{model_id}" not available on this container which hosts "{MODEL_ID}"',
                    "requested": model_id,
                    "available": MODEL_ID,
                }
            }

    if PIPELINE == "ALL":
        pipeline_name = call_inputs.get("PIPELINE", None)

        # Extract the pipline name from the call inputs, use the default AutoPipelineForText2Image
        if not pipeline_name:
            pipeline_name = "AutoPipelineForText2Image"
            result["$meta"].update({"PIPELINE": pipeline_name})

        # Get the pipeline for the model
        pipeline = get_pipeline_for_model(
            pipeline_name,
            model,
            model_id,
            model_revision=model_revision if RUNTIME_DOWNLOADS else MODEL_REVISION,
            model_precision=model_precision if RUNTIME_DOWNLOADS else MODEL_PRECISION,
        )
        if not pipeline:
            return {
                "$error": {
                    "code": "NO_SUCH_PIPELINE",
                    "message": f'"{pipeline_name}" is not an official nor community Diffusers pipelines',
                    "requested": pipeline_name,
                    "available": list_available_pipelines(),
                }
            }
    else:
        # If the pipeline is not set to "ALL", use the model as the pipeline
        pipeline = model

    # Extract the scheduler name from the call inputs, or use the default DPMSolverMultistepScheduler
    scheduler_name = call_inputs.get("SCHEDULER", None)
    if not scheduler_name:
        scheduler_name = "DPMSolverMultistepScheduler"
        result["$meta"].update({"SCHEDULER": scheduler_name})

    # Get the scheduler for the model
    pipeline.scheduler = get_scheduler(model_id, scheduler_name)
    if pipeline.scheduler == None:
        return {
            "$error": {
                "code": "INVALID_SCHEDULER",
                "message": "",
                "requeted": call_inputs.get("SCHEDULER", None),
                "available": ", ".join(SCHEDULERS),
            }
        }

    # Use the model's safety checker if it has one, and if the safety checker isn't disabled
    safety_checker = call_inputs.get("safety_checker", True)
    pipeline.safety_checker = (
        model.safety_checker if safety_checker and hasattr(model, "safety_checker") else None
    )

    # Extract the is_url flag from the call inputs, defaulting to False
    is_url = call_inputs.get("is_url", False)

    # Select the image decoder based on whether the input is an image Url or a base64-encoded image
    image_decoder = getFromUrl if is_url else decodeBase64Image

    # Extract the textual inversions from the call inputs and load them if necessary
    textual_inversions = call_inputs.get("textual_inversions", [])
    await handle_textual_inversions(textual_inversions, model, status=status_instance)

    # TODO: Currently we only support a single string, but we should allow
    # an array too in anticipation of multi-LoRA support in diffusers
    # tracked at https://github.com/huggingface/diffusers/issues/2613.

    # Get the LoRA weights from the inputs and convert the weights to a JSON string
    lora_weights = call_inputs.get("lora_weights", None)
    lora_weights_joined = json.dumps(lora_weights)

    # If the current weights are different from the last ones, process them
    if last_lora_weights != lora_weights_joined:
        # If there were previous weights, unload them
        if last_lora_weights != None and last_lora_weights != "[]":
            print("Unloading previous LoRA weights")
            pipeline.unload_lora_weights()

        # Update the last weights to the current ones and clear the old cross attention arguments
        last_lora_weights = lora_weights_joined
        cross_attention_kwargs = {}

        # Convert the weights to a list if necessary
        if type(lora_weights) is not list:
            lora_weights = [lora_weights] if lora_weights else []

        # Load each LoRA weight into the cross attention arguments and the pipline's LoRA scale
        if len(lora_weights) > 0:
            for weights in lora_weights:
                # Create a storage object for the weights
                storage = Storage(weights, no_raise=True, status=status_instance)
                if storage:
                    # Get the filename and scale from the storage query
                    storage_query_fname = storage.query.get("fname")
                    storage_query_scale = (
                        float(storage.query.get("scale")[0])
                        if storage.query.get("scale")
                        else 1
                    )
                    # Update the cross attention arguments with the LoRA weight's scale
                    cross_attention_kwargs.update({"scale": storage_query_scale})

                    # Update the pipeline's LoRA scale with the LoRA weight's scale
                    # https://github.com/damian0815/compel/issues/42#issuecomment-1656989385
                    pipeline._lora_scale = storage_query_scale

                    # Create a standardized filename for the weights
                    if storage_query_fname:
                        fname = storage_query_fname[0]
                    else:
                        # If there is no existing filename, generate a hash of the weights and use that to create a filename
                        hash = sha256(weights.encode("utf-8")).hexdigest()
                        fname = "url_" + hash[:7] + "--" + storage.url.split("/").pop()
                    cache_fname = "lora_weights--" + fname
                    path = os.path.join(MODELS_DIR, cache_fname)

                    # If the cached weights do not exist, download them
                    if not os.path.exists(path):
                        await asyncio.to_thread(storage.download_file, path)
                    print("Load lora_weights `" + weights + "` from `" + path + "`")

                    # Load the LoRA weights into the pipeline
                    pipeline.load_lora_weights(
                        MODELS_DIR, weight_name=cache_fname, local_files_only=True
                    )
                else:
                    print("Loading from huggingface not supported yet: " + weights)
                    # maybe something like sayakpaul/civitai-light-shadow-lora#lora=l_a_s.s9s?
                    # lora_model_id = "sayakpaul/civitai-light-shadow-lora"
                    # lora_filename = "light_and_shadow.safetensors"
                    # pipeline.load_lora_weights(lora_model_id, weight_name=lora_filename)
    else:
        print("No changes to LoRAs since last call")

    # TODO, generalize
    # Convert the input cross attention arguments to a dictionary object, and combine them with the existing cross attention arguments
    mi_cross_attention_kwargs = model_inputs.get("cross_attention_kwargs", None)
    if mi_cross_attention_kwargs:
        # Remove the existing cross attention arguments from the model inputs
        model_inputs.pop("cross_attention_kwargs")

        # Instantiate a new cross attention arguments dictionary if it doesn't exist
        if not cross_attention_kwargs:
            cross_attention_kwargs = {}

        # If the arguments from the model inputs are a json string, convert them to a dictionary
        if isinstance(mi_cross_attention_kwargs, str):
            cross_attention_kwargs.update(json.loads(mi_cross_attention_kwargs))

        # If the arguments from the model inputs are a dictionary, just copy them to the new dictionary
        elif type(mi_cross_attention_kwargs) == dict:
            cross_attention_kwargs.update(mi_cross_attention_kwargs)
        else:
            # If the arguments are not a string or a dictionary, return an error
            return {
                "$error": {
                    "code": "INVALID_CROSS_ATTENTION_KWARGS",
                    "message": "`cross_attention_kwargs` should be a dict or json string",
                }
            }
    print({"cross_attention_kwargs": cross_attention_kwargs})

    # If there are cross attention arguments (either from the model inputs or otherwise), update the model's cross attention arguments
    if cross_attention_kwargs:
        model_inputs.update({"cross_attention_kwargs": cross_attention_kwargs})

    # If there are images in the model inputs, decode them
    for image_type in ["init_image", "image", "mask_image", "input_image"]:  # ESS added "input_image" here
        if image_type in model_inputs:
            model_inputs[image_type] = image_decoder(model_inputs.get(image_type), image_type)

    # If there are instance images in the model inputs, decode them and organize in a list (unlike the other image types, there can be multiple instance images)
    if "instance_images" in model_inputs:
        model_inputs["instance_images"] = list(
            map(
                lambda str: image_decoder(str, "instance_image"),
                model_inputs["instance_images"],
            )
        )

    # Run patchmatch for inpainting
    if call_inputs.get("FILL_MODE", None) == "patchmatch":
        sel_buffer = np.array(model_inputs.get("init_image"))
        img = sel_buffer[:, :, 0:3]
        mask = sel_buffer[:, :, -1]
        img = patch_match.inpaint(img, mask=255 - mask, patch_size=3)
        model_inputs["init_image"] = PIL.Image.fromarray(img)
        mask = 255 - mask
        mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
        mask = mask.repeat(8, axis=0).repeat(8, axis=1)
        model_inputs["mask_image"] = PIL.Image.fromarray(mask)

    # Run the dreambooth training if specified in the call inputs and the environment variable USE_DREAMBOOTH is set
    if call_inputs.get("train", None) == "dreambooth":
        if not USE_DREAMBOOTH:
            return {
                "$error": {
                    "code": "TRAIN_DREAMBOOTH_NOT_AVAILABLE",
                    "message": 'Called with callInput { train: "dreambooth" } but built with USE_DREAMBOOTH=0',
                }
            }

        # Enable gradient computation in PyTorch for training (it's disabled by default for inference since it's memory-intensive)
        torch.set_grad_enabled(True)

        # Run the TrainDreamBooth function in a separate thread and merge its result with the existing result dictionary

        result = result | await asyncio.to_thread(
            TrainDreamBooth,
            model_id,
            pipeline,
            model_inputs,
            call_inputs,
            status_update_options=status_update_options,
            revision=model_revision,
            variant=model_precision,
        )

        # Disable gradient computation in PyTorch after training
        torch.set_grad_enabled(False)

        result = await finalize_run(result, status_update_options)

        # Return the result now, since the training is done and we won't be doing inference in this same run
        return result

    await pipeline_run.send_event_update("inference", "start", {"startRequestId": startRequestId}, status_update_options)

    # Get the seed from the model inputs if it exists, and create a generator with it or with a random seed
    # Do this after dreambooth as dreambooth accepts a seed int directly.
    seed = model_inputs.get("seed", None)
    if seed == None:
        generator = torch.Generator(device=device)
        generator.seed()
    else:
        # If there is a seed, use it to create a generator
        generator = torch.Generator(device=device).manual_seed(seed)
        del model_inputs["seed"]

    # Add the new generator to the model inputs
    model_inputs.update({"generator": generator})

    callback = None
    # If the user wants a progress message for each step, create a callback function to do so
    if model_inputs.get("callback_steps", None):
        def callback(step: int, timestep: int, callback_kwargs: dict):
            coroutine = pipeline_run.send_event_update(
                "inference",
                "progress",
                {"startRequestId": startRequestId, "step": step},
                status_update_options,
            )
            asyncio.run_coroutine_threadsafe(coroutine, asyncio.get_event_loop())
    else:
        # Otherwise, have the callback function just update the status for each step
        vae = pipeline.vae
        scaling_factor = vae.config.scaling_factor
        image_processor = pipeline.image_processor

        def callback(step: int, timestep: int, callback_kwargs: dict):
            if status_instance:
                status_instance.update(
                    "inference", step / model_inputs.get("num_inference_steps", 50)
                )
    print({"callback_on_step_end": callback, "**model_inputs": model_inputs})

    # Check if the model is a StableDiffusionXL pipeline
    is_sdxl = (
        isinstance(model, StableDiffusionXLPipeline)
        or isinstance(model, StableDiffusionXLImg2ImgPipeline)
        or isinstance(model, StableDiffusionXLInpaintPipeline)
    )

    with torch.inference_mode():
        # Get the custom pipeline method from the call inputs if it exists
        custom_pipeline_method = call_inputs.get("custom_pipeline_method", None)

        # If there are weighted prompts in the call inputs, prepare them
        if call_inputs.get("compel_prompts", False):
            prepare_prompts(pipeline, model_inputs, is_sdxl)

        # Run the pipeline asynchronously
        try:
            async_pipeline = asyncio.to_thread(
                getattr(pipeline, custom_pipeline_method)
                if custom_pipeline_method
                else pipeline,
                callback_on_step_end=callback,
                **model_inputs,
            )
            # TODO ESS: research autocast and see if makes sense to uncomment
            # if call_inputs.get("PIPELINE") != "StableDiffusionPipeline":
            #    # autocast img2img and inpaint which are broken in diffusers 0.4.0, 0.4.1
            #    # still broken in 0.5.1
            #    with autocast(device_id):
            #        images = (await async_pipeline).images
            # else:

            # Await the result of the pipeline
            pipeResult = await async_pipeline
            images = pipeResult.images

        except Exception as err:
            return {
                "$error": {
                    "code": "PIPELINE_ERROR",
                    "name": type(err).__name__,
                    "message": str(err),
                    "stack": traceback.format_exc(),
                }
            }

    # Convert the returned images to base64 in the specified format (default is PNG)
    images_base64 = []
    image_format = call_inputs.get("image_format", "PNG")
    image_opts = (
        {"lossless": True} if image_format == "PNG" or image_format == "WEBP" else {}
    )
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format=image_format, **image_opts)
        images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    # Send a 'done' status update for the inference
    await pipeline_run.send_event_update("inference", "done", {"startRequestId": startRequestId}, status_update_options)

    # Add the images to the result dictionary
    if len(images_base64) > 1:
        result = result | {"images_base64": images_base64}
    else:
        result = result | {"image_base64": images_base64[0]}

    # Add an NFSW flag to the result if the pipeline detected NSFW content
    nsfw_content_detected = pipeResult.get("nsfw_content_detected", None)
    if nsfw_content_detected:
        result = result | {"nsfw_content_detected": nsfw_content_detected}

    result = await finalize_run(result, status_update_options)

    return result


async def finalize_run(result, status_update_options):

    pipeline_run = status_update_options.get("pipeline_run")
    if pipeline_run:
        await pipeline_run.send_event_update("completion", "start", options=status_update_options)

    # Add the timings and memory usage to the result dictionary
    mem_usage = calculate_memory_usage()
    return result | {"$timings": pipeline_run.get_process_durations(), "$mem_usage": mem_usage}
