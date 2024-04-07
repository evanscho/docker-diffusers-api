import os
import time
from diffusers import schedulers as _schedulers

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
HOME = os.path.expanduser("~")
MODELS_DIR = os.path.join(HOME, ".cache", "diffusers-api")

SCHEDULERS = [
    "DPMSolverMultistepScheduler",
    "LMSDiscreteScheduler",
    "DDIMScheduler",
    "PNDMScheduler",
    "EulerAncestralDiscreteScheduler",
    "EulerDiscreteScheduler",
]

DEFAULT_SCHEDULER = os.getenv("DEFAULT_SCHEDULER", SCHEDULERS[0])


def initScheduler(MODEL_ID: str, scheduler_id: str, download=False):
    print(f"Initializing {scheduler_id} for {MODEL_ID}...")
    start = time.time()
    scheduler = getattr(_schedulers, scheduler_id)
    if scheduler == None:
        return None

    model_dir = os.path.join(MODELS_DIR, MODEL_ID)
    if not os.path.isdir(model_dir):
        model_dir = None

    inittedScheduler = scheduler.from_pretrained(
        model_dir or MODEL_ID,
        subfolder="scheduler",
        use_auth_token=HF_AUTH_TOKEN,
        local_files_only=not download,
    )
    diff = round((time.time() - start) * 1000)
    print(f"Initialized {scheduler_id} for {MODEL_ID} in {diff}ms")

    return inittedScheduler


schedulers = {}


def getScheduler(MODEL_ID: str, scheduler_id: str, download=False):
    schedulersByModel = schedulers.get(MODEL_ID, None)
    if schedulersByModel == None:
        schedulersByModel = {}
        schedulers.update({MODEL_ID: schedulersByModel})

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
        scheduler = initScheduler(MODEL_ID, scheduler_id, download)
        schedulersByModel.update({scheduler_id: scheduler})

    return scheduler
