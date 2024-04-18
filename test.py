# This file is used to verify your http server acts as expected

import requests
import base64
import os
import json
import sys
import time
import argparse
from io import BytesIO
from PIL import Image
from pathlib import Path

# Base directory can be changed to the path of the current Python script: os.path.realpath(sys.argv[0])
path = "."
TESTS = Path(path) / "tests"
FIXTURES = TESTS / "fixtures"
OUTPUT = TESTS / "output"
TEST_URL = os.getenv("TEST_URL", "http://localhost:8000/")
OUTPUT.mkdir(parents=True, exist_ok=True)


# ================================
# Utility and helper functions for tests
# ================================

def encode_in_base64(filename: str):
    path = FIXTURES / filename if not isinstance(filename, Path) else filename
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("ascii")


def output_path(filename: str):
    return str(OUTPUT / filename)


def readable_size_from_byte_size(num: float, suffix: str = "B"):
    """Returns human-readable size from bytes size: https://stackoverflow.com/a/1094933/1839099"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def decode_base64_image_and_save(image_byte_string: str, name: str):
    image_bytes = base64.b64decode(image_byte_string.encode("utf-8"))
    image = Image.open(BytesIO(image_bytes))
    file_path = output_path(f"{name}.png")
    image.save(file_path)
    print(f"Saved {file_path}")
    size_formatted = readable_size_from_byte_size(file_path.stat().st_size)
    return f"[{image.width}x{image.height} {image.format} image, {size_formatted} bytes]"


def process_inputs_for_logging(input_value, key):
    # Check if the input is a dictionary and recursively process each value
    if isinstance(input_value, dict):  # for modelInputs and callInputs
        return {k: process_inputs_for_logging(v, k) for k, v in input_value.items()}
    # Check if the input is a list called 'instance_images'
    elif key == "instance_images" and isinstance(input_value, list):
        return f"[Array({len(input_value)})]"
    # Check for image data encoded in base64 format
    elif key in ["init_image", "image"] and isinstance(input_value, str):
        return "[image]"
    return input_value


def log_inputs(inputs):
    inputs_to_log = json.dumps({k: process_inputs_for_logging(v, k) for k, v in inputs.items()}, indent=4)
    print(inputs_to_log)
    print()


def print_exception_error(error):
    print()
    if code := error.get("code"):
        title = f"Exception {code} on container:"
        print(title)
        print("-" * len(title))
    if stack := error.get("stack"):
        print(stack)


def format_time(milliseconds):
    if milliseconds > 1000:
        return f"{milliseconds / 1000:.1f}s"
    return f"{milliseconds}ms"


def print_timing_information(result, elapsed_time):
    timings = result.get("$timings")
    if timings:
        # Create string of timings, but filter out init timings since they're not core to the current run
        timings_str = ", ".join(
            f"{key}: {format_time(value)}" for key, value in timings.items() if "init" not in key
        )
        print(f"Request took {elapsed_time:.1f}s ({timings_str})")
    else:
        print(f"Request took {elapsed_time:.1f}s")

# ================================
# Functions for making HTTP requests
# ================================


def request_error_handling(response):
    error_message = "Unknown error"  # Default error message
    try:
        # Attempt to decode the JSON response to fetch the 'error' key
        error_message = response.json().get('error', error_message)
    except ValueError:
        # Handle cases where the response is not in JSON format
        error_message = response.text or error_message

    # Limiting the error message to the first 5 lines if it's longer
    limited_error_message = '\n'.join(error_message.splitlines()[:10])
    if len(error_message.splitlines()) > 10:
        limited_error_message += "\n[...truncated...]"

    print(f"Unexpected HTTP response code: {response.status_code} - {response.reason}")
    print(f"Error: '{limited_error_message}'")


def post_request(url, payload, headers=None):
    print(url)
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        request_error_handling(response)
        sys.exit(1)
    return response


def get_request(url, headers=None):
    print(url)
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        request_error_handling(response)
        sys.exit(1)
    return response


# ================================
# Functions related to test executions
# ================================


def process_result(result, name):
    if result.get("images_base64") is None and result.get("image_base64") is None:
        error = result.get("$error")
        if error:
            print_exception_error(error)
            return
    elif images_base64 := result.get("images_base64"):
        for idx, image_byte_string in enumerate(images_base64):
            images_base64[idx] = decode_base64_image_and_save(image_byte_string, f"{name}_{idx}")
    elif image_base64 := result.get("image_base64"):
        result["image_base64"] = decode_base64_image_and_save(image_base64, name)

    print()
    print(json.dumps(result, indent=4))
    print()
    return result


def run_test_runpod(inputs, args):
    RUNPOD_API_URL = "https://api.runpod.ai/v2/"
    RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
    RUNPOD_MODEL_KEY = os.getenv("RUNPOD_MODEL_KEY")

    if not (RUNPOD_API_KEY and RUNPOD_MODEL_KEY):
        print("Error: RUNPOD_API_KEY or RUNPOD_MODEL_KEY not set, aborting...")
        sys.exit(1)

    url_base = RUNPOD_API_URL + RUNPOD_MODEL_KEY
    payload = {"input": inputs}
    headers = {"Authorization": "Bearer " + RUNPOD_API_KEY}

    response = post_request(f"{url_base}/run", payload, headers)
    print(response)
    result = response.json()
    print(result)

    while result["status"] != "COMPLETED":
        time.sleep(1)
        request = get_request(f"{url_base}/status/{result['id']}", headers)
        result = request.json()

    return result["output"]


def run_test_local(inputs, args):
    test_url = args.get("test_url", TEST_URL)
    stream_events = inputs.get("callInputs", {}).get("streamEvents", 0) != 0
    print({"stream_events": stream_events})

    if stream_events:
        response = requests.post(test_url, json=inputs, stream=True, timeout=(6.1, 60 * 20))
        last_result = None
        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = json.loads(line.decode('utf-8'))
                    last_result = decoded_line  # Update the last result with the current line

                    # Print streamed lines as they come in
                    if not decoded_line.get("$timings"):
                        print(decoded_line)
                    # Return once the final result is received
                    else:
                        return decoded_line
                except json.JSONDecodeError as error:
                    print(f"Error decoding JSON: {error}")
                    print(f"Received line: {line}")
                    sys.exit(1)
                except Exception as e:
                    import traceback
                    print(f"Error processing line: {line}")
                    print(f"Exception {e.__class__.__name__}: {str(e)}\n{traceback.format_exc()}")
                    sys.exit(1)

        if last_result:
            return last_result  # Return the last result if no $timings were found
        else:
            print("WARNING: No valid data received from the server.")
    else:
        response = post_request(test_url, inputs)
        try:
            return response.json()
        except json.JSONDecodeError as error:
            print(f"Error decoding JSON: {error}")
            print(f"Received text: {response.text}")
            sys.exit(1)


def run_test(name, args, extra_call_inputs, extra_model_inputs):
    orig_inputs = all_tests.get(name)
    inputs = {
        "modelInputs": {**orig_inputs.get("modelInputs", {}), **extra_model_inputs},
        "callInputs": {**orig_inputs.get("callInputs", {}), **extra_call_inputs},
    }

    print(f"Running test: {name}")
    log_inputs(inputs)

    start = time.time()

    if args.get("runpod"):
        result = run_test_runpod(inputs, args)
    else:
        result = run_test_local(inputs, args)

    finish = time.time() - start
    print_timing_information(result, finish)

    return process_result(result, name)


# ====================================
# Test cases and adding them to the test suite
# ====================================

def test(name, inputs):
    global all_tests
    all_tests.update({name: inputs})


all_tests = {}
test("txt2img", {
    "modelInputs": {
        "prompt": "painting of 3-year-old boy,sweet smile,realistic,from neck up,detailed hair,(((in watercolor style))),created from brush strokes,abstract painting,",
        "num_inference_steps": 40,
    },
    "callInputs": {
        # "MODEL_ID": "<override_default>",  # (default)
        # "PIPELINE": "StableDiffusionPipeline",  # (default)
        # "SCHEDULER": "DPMSolverMultistepScheduler",  # (default)
    },
})

# multiple images
test("txt2img-multiple", {
    "modelInputs": {
        "prompt": "realistic field of grass",
        "num_images_per_prompt": 2,
    }
})


test("img2img", {
    "modelInputs": {
        "prompt": "A fantasy landscape, trending on artstation",
        "image": encode_in_base64("sketch-mountains-input.jpg"),
    },
    "callInputs": {
        "PIPELINE": "StableDiffusionImg2ImgPipeline",
    },
})

test("inpaint-v1-4", {
    "modelInputs": {
        "prompt": "a cat sitting on a bench",
        "image": encode_in_base64("overture-creations-5sI6fQgYIuo.png"),
        "mask_image": encode_in_base64("overture-creations-5sI6fQgYIuo_mask.png"),
    },
    "callInputs": {
        "MODEL_ID": "CompVis/stable-diffusion-v1-4",
        "PIPELINE": "StableDiffusionInpaintPipelineLegacy",
        "SCHEDULER": "DDIMScheduler",  # Note, as of diffusers 0.3.0, no LMS yet
    },
})

test("inpaint-sd", {
    "modelInputs": {
        "prompt": "a cat sitting on a bench",
        "image": encode_in_base64("overture-creations-5sI6fQgYIuo.png"),
        "mask_image": encode_in_base64("overture-creations-5sI6fQgYIuo_mask.png"),
    },
    "callInputs": {
        "MODEL_ID": "runwayml/stable-diffusion-inpainting",
        "PIPELINE": "StableDiffusionInpaintPipeline",
        "SCHEDULER": "DDIMScheduler",  # Note, as of diffusers 0.3.0, no LMS yet
    },
})

test("checkpoint", {
    "modelInputs": {
        "prompt": "1girl",
    },
    "callInputs": {
        "MODEL_ID": "hakurei/waifu-diffusion-v1-3",
        "MODEL_URL": "s3://",
        "CHECKPOINT_URL": "http://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt",
    },
})

if os.getenv("USE_PATCHMATCH"):
    test("outpaint", {
        "modelInputs": {
            "prompt": "girl with a pearl earing standing in a big room",
            "image": encode_in_base64("girl_with_pearl_earing_outpainting_in.png"),
        },
        "callInputs": {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "PIPELINE": "StableDiffusionInpaintPipelineLegacy",
            "SCHEDULER": "DDIMScheduler",  # Note, as of diffusers 0.3.0, no LMS yet
            "FILL_MODE": "patchmatch",
        },
    })

# Actually we just want this to be a non-default test?
if True or os.getenv("USE_DREAMBOOTH"):
    # If you're calling from the command line, don't forget to
    # specify a destination if you want your fine-tuned model to
    # be uploaded somewhere at the end.
    test("dreambooth", {
        "modelInputs": {
            "instance_prompt": "a photo of daiton person",
            "instance_images": list(
                map(
                    encode_in_base64,
                    list(Path("tests/fixtures/dreambooth").iterdir()),
                )
            ),
            # Option 1: upload to HuggingFace (see notes below)
            # Make sure your HF API token has read/write access.
            "hub_model_id": "evanscho/davi-tests",
            "push_to_hub": True,
        },
        "callInputs": {
            "train": "dreambooth",
            "streamEvents": True,
            # Option 2: store on S3.  Note the **s3:///* (x3).  See notes below.
            # "dest_url": "s3:///bucket/filename.tar.zst".
        },
    })


# =============================
# Parse arguments and run tests
# =============================

def validate_tests(tests_to_run):
    invalid_tests = [test for test in tests_to_run if test not in all_tests]
    if invalid_tests:
        print(f"No such tests: {', '.join(invalid_tests)}")
        exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runpod", action="store_true")
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--call-arg", action="append", type=str, default=[])
    parser.add_argument("--model-arg", action="append", type=str, default=[])
    # Returns a tuple of (args, tests_to_run), tests_to_run being the remaining unparsed arguments
    return parser.parse_known_args()


def parse_key_value_args(args, arg_type):
    parsed_args = {}
    raw_args = getattr(args, f"{arg_type}_arg", [])
    for arg in raw_args:
        key, value = arg.split("=", 1)
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)
        parsed_args[key] = value
    return parsed_args


def main(tests_to_run, args, extra_call_inputs, extra_model_inputs):
    validate_tests(tests_to_run)
    for test in tests_to_run:
        run_test(test, args, extra_call_inputs, extra_model_inputs)


if __name__ == "__main__":
    args, tests_to_run = parse_arguments()
    call_inputs = parse_key_value_args(args, "call")
    model_inputs = parse_key_value_args(args, "model")

    if args.scheduler:
        call_inputs["SCHEDULER"] = args.scheduler

    if not tests_to_run:
        print("Usage: python3 test.py [--runpod] [--scheduler=SomeScheduler] [all / test1] [test2] [etc]")
        sys.exit()
    elif tests_to_run[0].upper() == "ALL":
        tests_to_run = list(all_tests.keys())

    main(
        tests_to_run,
        vars(args),
        extra_call_inputs=call_inputs,
        extra_model_inputs=model_inputs,
    )
