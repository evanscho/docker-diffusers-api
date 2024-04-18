import os
import time
from diffusers import schedulers as _schedulers
from lib.vars import HF_AUTH_TOKEN, MODELS_DIR

SCHEDULERS = [
    "DPMSolverMultistepScheduler",
    "LMSDiscreteScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
]


def init_scheduler(model_id: str, scheduler_id: str, download=False):
    print(f"Initializing {scheduler_id} for {model_id}...")
    start = time.time()
    scheduler = getattr(_schedulers, scheduler_id)
    if scheduler == None:
        return None

    model_dir = os.path.join(MODELS_DIR, model_id)
    if not os.path.isdir(model_dir):
        model_dir = None

    inittedScheduler = scheduler.from_pretrained(
        model_dir or model_id,
        subfolder="scheduler",
        token=HF_AUTH_TOKEN,
        local_files_only=not download,
    )
    diff = round((time.time() - start) * 1000)
    print(f"Initialized {scheduler_id} for {model_id} in {diff}ms")

    return inittedScheduler


schedulers = {}


def get_scheduler(model_id: str, scheduler_id: str, download=False):
    schedulersByModel = schedulers.get(model_id, None)
    if schedulersByModel == None:
        schedulersByModel = {}
        schedulers.update({model_id: schedulersByModel})

    # Check for use of old names
    deprecated_map = {
        "LMS": "LMSDiscreteScheduler",
        "DDIM": "DDIMScheduler",
        "PNDM": "PNDMScheduler",
    }
    scheduler_renamed = deprecated_map.get(scheduler_id, None)
    if scheduler_renamed != None:
        print(
            f'[Deprecation Warning]: Scheduler "{scheduler_id}" is now '
            f'called "{scheduler_id}".  Please rename as this will '
            f"stop working in a future release."
        )
        scheduler_id = scheduler_renamed

    scheduler = schedulersByModel.get(scheduler_id, None)
    if scheduler == None:
        scheduler = init_scheduler(model_id, scheduler_id, download)
        schedulersByModel.update({scheduler_id: scheduler})

    return scheduler
