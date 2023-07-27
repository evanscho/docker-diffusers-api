import asyncio
from sched import scheduler
import torch

from torch import autocast
from diffusers import __version__
import base64
from io import BytesIO
import PIL
import json
from loadModel import loadModel
from send import send, getTimings, clearSession
from status import status
import os
import numpy as np
import skimage
import skimage.measure
from getScheduler import getScheduler, SCHEDULERS
from getPipeline import getPipelineForModel, listAvailablePipelines, clearPipelines
import re
import requests
from download import download_model, normalize_model_id
import traceback
from precision import MODEL_REVISION, MODEL_PRECISION
from device import device, device_id, device_name
from diffusers.models.cross_attention import CrossAttnProcessor
from utils import Storage
from hashlib import sha256
from threading import Timer
import extras

from lib.textual_inversions import handle_textual_inversions
from lib.vars import (
    RUNTIME_DOWNLOADS,
    USE_DREAMBOOTH,
    MODEL_ID,
    PIPELINE,
    HF_AUTH_TOKEN,
    HOME,
    MODELS_DIR,
)

if USE_DREAMBOOTH:
    from train_dreambooth import TrainDreamBooth
print(os.environ.get("USE_PATCHMATCH"))
if os.environ.get("USE_PATCHMATCH") == "1":
    from PyPatchMatch import patch_match

torch.set_grad_enabled(False)
always_normalize_model_id = None


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model  # needed for bananna optimizations
    global always_normalize_model_id

    asyncio.run(
        send(
            "init",
            "start",
            {
                "device": device_name,
                "hostname": os.getenv("HOSTNAME"),
                "model_id": MODEL_ID,
                "diffusers": __version__,
            },
        )
    )

    if MODEL_ID == "ALL" or RUNTIME_DOWNLOADS:
        global last_model_id
        last_model_id = None

    if not RUNTIME_DOWNLOADS:
        normalized_model_id = normalize_model_id(MODEL_ID, MODEL_REVISION)
        model_dir = os.path.join(MODELS_DIR, normalized_model_id)
        if os.path.isdir(model_dir):
            always_normalize_model_id = model_dir
        else:
            normalized_model_id = MODEL_ID

        model = loadModel(
            model_id=always_normalize_model_id or MODEL_ID,
            load=True,
            precision=MODEL_PRECISION,
            revision=MODEL_REVISION,
        )
    else:
        model = None

    asyncio.run(send("init", "done"))


def decodeBase64Image(imageStr: str, name: str) -> PIL.Image:
    image = PIL.Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image


def getFromUrl(url: str, name: str) -> PIL.Image:
    response = requests.get(url)
    image = PIL.Image.open(BytesIO(response.content))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image


def truncateInputs(inputs: dict):
    clone = inputs.copy()
    if "modelInputs" in clone:
        modelInputs = clone["modelInputs"] = clone["modelInputs"].copy()
        for item in ["init_image", "mask_image", "image", "input_image"]:
            if item in modelInputs:
                modelInputs[item] = modelInputs[item][0:6] + "..."
        if "instance_images" in modelInputs:
            modelInputs["instance_images"] = list(
                map(lambda str: str[0:6] + "...", modelInputs["instance_images"])
            )
    return clone


# last_xformers_memory_efficient_attention = {}
last_attn_procs = None
last_lora_weights = None


# Inference is ran for every server call
# Reference your preloaded global model variable here.
async def inference(all_inputs: dict, response) -> dict:
    global model
    global pipelines
    global last_model_id
    global schedulers
    # global last_xformers_memory_efficient_attention
    global always_normalize_model_id
    global last_attn_procs
    global last_lora_weights

    clearSession()

    print(json.dumps(truncateInputs(all_inputs), indent=2))
    model_inputs = all_inputs.get("modelInputs", None)
    call_inputs = all_inputs.get("callInputs", None)
    result = {"$meta": {}}

    send_opts = {}
    if call_inputs.get("SEND_URL", None):
        send_opts.update({"SEND_URL": call_inputs.get("SEND_URL")})
    if call_inputs.get("SIGN_KEY", None):
        send_opts.update({"SIGN_KEY": call_inputs.get("SIGN_KEY")})
    if response:
        send_opts.update({"response": response})

        async def sendStatusAsync():
            await response.send(json.dumps(status.get()) + "\n")

        def sendStatus():
            try:
                asyncio.run(sendStatusAsync())
                Timer(1.0, sendStatus).start()
            except:
                pass

        Timer(1.0, sendStatus).start()

    if model_inputs == None or call_inputs == None:
        return {
            "$error": {
                "code": "INVALID_INPUTS",
                "message": "Expecting on object like { modelInputs: {}, callInputs: {} } but got "
                + json.dumps(all_inputs),
            }
        }

    startRequestId = call_inputs.get("startRequestId", None)

    use_extra = call_inputs.get("use_extra", None)
    if use_extra:
        extra = getattr(extras, use_extra, None)
        if not extra:
            return {
                "$error": {
                    "code": "NO_SUCH_EXTRA",
                    "message": 'Requested "'
                    + use_extra
                    + '", available: "'
                    + '", "'.join(extras.keys())
                    + '"',
                }
            }
        return await extra(
            model_inputs,
            call_inputs,
            send_opts=send_opts,
            startRequestId=startRequestId,
        )

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
        result["$meta"].update({"MODEL_ID": MODEL_ID})
    normalized_model_id = model_id

    if RUNTIME_DOWNLOADS:
        hf_model_id = call_inputs.get("HF_MODEL_ID", None)
        model_revision = call_inputs.get("MODEL_REVISION", None)
        model_precision = call_inputs.get("MODEL_PRECISION", None)
        checkpoint_url = call_inputs.get("CHECKPOINT_URL", None)
        checkpoint_config_url = call_inputs.get("CHECKPOINT_CONFIG_URL", None)
        normalized_model_id = normalize_model_id(model_id, model_revision)
        model_dir = os.path.join(MODELS_DIR, normalized_model_id)
        if last_model_id != normalized_model_id:
            # if not downloaded_models.get(normalized_model_id, None):
            if not os.path.isdir(model_dir):
                model_url = call_inputs.get("MODEL_URL", None)
                if not model_url:
                    # return {
                    #     "$error": {
                    #         "code": "NO_MODEL_URL",
                    #         "message": "Currently RUNTIME_DOWNOADS requires a MODEL_URL callInput",
                    #     }
                    # }
                    normalized_model_id = hf_model_id or model_id
                await download_model(
                    model_id=model_id,
                    model_url=model_url,
                    model_revision=model_revision,
                    checkpoint_url=checkpoint_url,
                    checkpoint_config_url=checkpoint_config_url,
                    hf_model_id=hf_model_id,
                    model_precision=model_precision,
                    send_opts=send_opts,
                )
                # downloaded_models.update({normalized_model_id: True})
            clearPipelines()
            if model:
                model.to("cpu")  # Necessary to avoid a memory leak
            await send(
                "loadModel", "start", {"startRequestId": startRequestId}, send_opts
            )
            model = await asyncio.to_thread(
                loadModel,
                model_id=normalized_model_id,
                load=True,
                precision=model_precision,
                revision=model_revision,
                send_opts=send_opts,
            )
            await send(
                "loadModel", "done", {"startRequestId": startRequestId}, send_opts
            )
            last_model_id = normalized_model_id
            last_attn_procs = None
            last_lora_weights = None
    else:
        if always_normalize_model_id:
            normalized_model_id = always_normalize_model_id
        print(
            {
                "always_normalize_model_id": always_normalize_model_id,
                "normalized_model_id": normalized_model_id,
            }
        )

    if MODEL_ID == "ALL":
        if last_model_id != normalized_model_id:
            clearPipelines()
            model = loadModel(normalized_model_id, send_opts=send_opts)
            last_model_id = normalized_model_id
    else:
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
        if not pipeline_name:
            pipeline_name = "StableDiffusionPipeline"
            result["$meta"].update({"PIPELINE": pipeline_name})

        pipeline = getPipelineForModel(
            pipeline_name,
            model,
            normalized_model_id,
            model_revision=model_revision if RUNTIME_DOWNLOADS else MODEL_REVISION,
            model_precision=model_precision if RUNTIME_DOWNLOADS else MODEL_PRECISION,
        )
        if not pipeline:
            return {
                "$error": {
                    "code": "NO_SUCH_PIPELINE",
                    "message": f'"{pipeline_name}" is not an official nor community Diffusers pipelines',
                    "requested": pipeline_name,
                    "available": listAvailablePipelines(),
                }
            }
    else:
        pipeline = model

    scheduler_name = call_inputs.get("SCHEDULER", None)
    if not scheduler_name:
        scheduler_name = "DPMSolverMultistepScheduler"
        result["$meta"].update({"SCHEDULER": scheduler_name})

    pipeline.scheduler = getScheduler(normalized_model_id, scheduler_name)
    if pipeline.scheduler == None:
        return {
            "$error": {
                "code": "INVALID_SCHEDULER",
                "message": "",
                "requeted": call_inputs.get("SCHEDULER", None),
                "available": ", ".join(SCHEDULERS),
            }
        }

    safety_checker = call_inputs.get("safety_checker", True)
    pipeline.safety_checker = model.safety_checker if safety_checker else None
    is_url = call_inputs.get("is_url", False)
    image_decoder = getFromUrl if is_url else decodeBase64Image

    textual_inversions = call_inputs.get("textual_inversions", [])
    handle_textual_inversions(textual_inversions, model)

    # Better to use new lora_weights in next section
    attn_procs = call_inputs.get("attn_procs", None)
    if attn_procs is not last_attn_procs:
        print(
            "[DEPRECATED] Using `attn_procs` for LoRAs is deprecated. "
            + "Please use `lora_weights` instead."
        )
        last_attn_procs = attn_procs
        if attn_procs:
            storage = Storage(attn_procs, no_raise=True)
            if storage:
                hash = sha256(attn_procs.encode("utf-8")).hexdigest()
                attn_procs_from_safetensors = call_inputs.get(
                    "attn_procs_from_safetensors", None
                )
                fname = storage.url.split("/").pop()
                if attn_procs_from_safetensors and not re.match(
                    r".safetensors", attn_procs
                ):
                    fname += ".safetensors"
                if True:
                    # TODO, way to specify explicit name
                    path = os.path.join(
                        MODELS_DIR, "attn_proc--url_" + hash[:7] + "--" + fname
                    )
                attn_procs = path
                if not os.path.exists(path):
                    storage.download_and_extract(path)
            print("Load attn_procs " + attn_procs)
            # Workaround https://github.com/huggingface/diffusers/pull/2448#issuecomment-1453938119
            if storage and not re.search(r".safetensors", attn_procs):
                attn_procs = torch.load(attn_procs, map_location="cpu")
            pipeline.unet.load_attn_procs(attn_procs)
        else:
            print("Clearing attn procs")
            pipeline.unet.set_attn_processor(CrossAttnProcessor())

    # Currently we only support a single string, but we should allow
    # and array too in anticipation of multi-LoRA support in diffusers
    # tracked at https://github.com/huggingface/diffusers/issues/2613.
    lora_weights = call_inputs.get("lora_weights", None)
    if lora_weights is not last_lora_weights:
        last_lora_weights = lora_weights
        if lora_weights:
            pipeline.unet.set_attn_processor(CrossAttnProcessor())
            storage = Storage(lora_weights, no_raise=True)
            if storage:
                storage_query_fname = storage.query.get("fname")
                if storage_query_fname:
                    fname = storage_query_fname[0]
                else:
                    hash = sha256(lora_weights.encode("utf-8")).hexdigest()
                    fname = "url_" + hash[:7] + "--" + storage.url.split("/").pop()
                cache_fname = "lora_weights--" + fname
                path = os.path.join(MODELS_DIR, cache_fname)
                if not os.path.exists(path):
                    storage.download_and_extract(path, status=status)
                print("Load lora_weights `" + lora_weights + "` from `" + path + "`")
                pipeline.load_lora_weights(
                    MODELS_DIR, weight_name=cache_fname, local_files_only=True
                )
            else:
                print("Loading from huggingface not supported yet: " + lora_weights)
                # maybe something like sayakpaul/civitai-light-shadow-lora#lora=l_a_s.s9s?
                # lora_model_id = "sayakpaul/civitai-light-shadow-lora"
                # lora_filename = "light_and_shadow.safetensors"
                # pipeline.load_lora_weights(lora_model_id, weight_name=lora_filename)
        else:
            print("Clearing attn procs")
            pipeline.unet.set_attn_processor(CrossAttnProcessor())

    # TODO, generalize
    cross_attention_kwargs = model_inputs.get("cross_attention_kwargs", None)
    if isinstance(cross_attention_kwargs, str):
        model_inputs["cross_attention_kwargs"] = json.loads(cross_attention_kwargs)

    # Parse out your arguments
    # prompt = model_inputs.get("prompt", None)
    # if prompt == None:
    #     return {"message": "No prompt provided"}
    #
    #   height = model_inputs.get("height", 512)
    #  width = model_inputs.get("width", 512)
    # num_inference_steps = model_inputs.get("num_inference_steps", 50)
    # guidance_scale = model_inputs.get("guidance_scale", 7.5)
    # seed = model_inputs.get("seed", None)
    #   strength = model_inputs.get("strength", 0.75)

    if "init_image" in model_inputs:
        model_inputs["init_image"] = image_decoder(
            model_inputs.get("init_image"), "init_image"
        )

    if "image" in model_inputs:
        model_inputs["image"] = image_decoder(model_inputs.get("image"), "image")

    if "mask_image" in model_inputs:
        model_inputs["mask_image"] = image_decoder(
            model_inputs.get("mask_image"), "mask_image"
        )

    if "instance_images" in model_inputs:
        model_inputs["instance_images"] = list(
            map(
                lambda str: image_decoder(str, "instance_image"),
                model_inputs["instance_images"],
            )
        )

    await send("inference", "start", {"startRequestId": startRequestId}, send_opts)

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

    # Turning on takes 3ms and turning off 1ms... don't worry, I've got your back :)
    # x_m_e_a = call_inputs.get("xformers_memory_efficient_attention", True)
    # last_x_m_e_a = last_xformers_memory_efficient_attention.get(pipeline, None)
    # if x_m_e_a != last_x_m_e_a:
    #     if x_m_e_a == True:
    #         print("pipeline.enable_xformers_memory_efficient_attention()")
    #         pipeline.enable_xformers_memory_efficient_attention()  # default on
    #     elif x_m_e_a == False:
    #         print("pipeline.disable_xformers_memory_efficient_attention()")
    #         pipeline.disable_xformers_memory_efficient_attention()
    #     else:
    #         return {
    #             "$error": {
    #                 "code": "INVALID_XFORMERS_MEMORY_EFFICIENT_ATTENTION_VALUE",
    #                 "message": f"x_m_e_a expects True or False, not: {x_m_e_a}",
    #                 "requested": x_m_e_a,
    #                 "available": [True, False],
    #             }
    #         }
    #     last_xformers_memory_efficient_attention.update({pipeline: x_m_e_a})

    # Run the model
    # with autocast(device_id):
    # image = pipeline(**model_inputs).images[0]

    if call_inputs.get("train", None) == "dreambooth":
        if not USE_DREAMBOOTH:
            return {
                "$error": {
                    "code": "TRAIN_DREAMBOOTH_NOT_AVAILABLE",
                    "message": 'Called with callInput { train: "dreambooth" } but built with USE_DREAMBOOTH=0',
                }
            }

        if RUNTIME_DOWNLOADS:
            if os.path.isdir(model_dir):
                normalized_model_id = model_dir

        torch.set_grad_enabled(True)
        result = result | await asyncio.to_thread(
            TrainDreamBooth,
            normalized_model_id,
            pipeline,
            model_inputs,
            call_inputs,
            send_opts=send_opts,
        )
        torch.set_grad_enabled(False)
        await send("inference", "done", {"startRequestId": startRequestId}, send_opts)
        result.update({"$timings": getTimings()})
        return result

    # Do this after dreambooth as dreambooth accepts a seed int directly.
    seed = model_inputs.get("seed", None)
    if seed == None:
        generator = torch.Generator(device=device)
        generator.seed()
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
        del model_inputs["seed"]

    model_inputs.update({"generator": generator})

    callback = None
    if model_inputs.get("callback_steps", None):

        def callback(step: int, timestep: int, latents: torch.FloatTensor):
            asyncio.run(
                send(
                    "inference",
                    "progress",
                    {"startRequestId": startRequestId, "step": step},
                    send_opts,
                )
            )

    else:

        def callback(step: int, timestep: int, latents: torch.FloatTensor):
            status.update(
                "inference", step / model_inputs.get("num_inference_steps", 50)
            )

    with torch.inference_mode():
        custom_pipeline_method = call_inputs.get("custom_pipeline_method", None)

        try:
            async_pipeline = asyncio.to_thread(
                getattr(pipeline, custom_pipeline_method)
                if custom_pipeline_method
                else pipeline,
                callback=callback,
                **model_inputs,
            )
            if call_inputs.get("PIPELINE") != "StableDiffusionPipeline":
                # autocast im2img and inpaint which are broken in 0.4.0, 0.4.1
                # still broken in 0.5.1
                with autocast(device_id):
                    images = (await async_pipeline).images
            else:
                images = (await async_pipeline).images

        except Exception as err:
            return {
                "$error": {
                    "code": "PIPELINE_ERROR",
                    "name": type(err).__name__,
                    "message": str(err),
                    "stack": traceback.format_exc(),
                }
            }

    images_base64 = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        images_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    await send("inference", "done", {"startRequestId": startRequestId}, send_opts)

    # Return the results as a dictionary
    if len(images_base64) > 1:
        result = result | {"images_base64": images_base64}
    else:
        result = result | {"image_base64": images_base64[0]}

    # TODO, move and generalize in device.py
    mem_usage = 0
    if torch.cuda.is_available():
        mem_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()

    result = result | {"$timings": getTimings(), "$mem_usage": mem_usage}

    return result
