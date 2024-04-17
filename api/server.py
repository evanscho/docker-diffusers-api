# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from sanic import Sanic, response
from sanic.exceptions import SanicException
from sanic.response import json as json_response
from sanic.log import logger
import subprocess
import app as diffusers_model
import traceback
import os
import json
import sys
import time
from utils.logging import Tee
import logging
from event_completion_status import PercentageCompleteStatus, PercentageCompleteStatusSender

# Open the log file and create a Tee object that writes to both the log file and stdout/stderr
log_file = open('training.log', 'a')

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# Create the http server app
app = Sanic("my_app")
app.config.CORS_ORIGINS = os.getenv("CORS_ORIGINS") or "*"
app.config.RESPONSE_TIMEOUT = 60 * 60  # 1 hour (training and uploading can be long)
app.config.KEEP_ALIVE_TIMEOUT = 60 * 60  # for keeping the connection alive when streaming
app.config.NOISY_EXCEPTIONS = True


@app.before_server_start
async def initialize(app, loop):
    # We do the model load-to-GPU step on server startup
    # so the model object is available globally for reuse
    await diffusers_model.init()


@app.middleware("request")
async def log_request(request):
    try:
        request.ctx.start_time = time.time()
        logger.info(f"Request started: {request.method} {request.url}")
    except Exception as e:
        logger.error(f"Error in request middleware: {str(e)}")


@app.middleware("response")
async def log_response(request, response):
    try:
        total_time = time.time() - request.ctx.start_time
        logger.info(f"Request completed: {request.method} {request.url} in {total_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in response middleware: {str(e)}")


@app.route("/healthcheck", methods=["GET"])
def healthcheck(request):
    # Healthchecks verify that the environment is correct on Banana Serverless

    # Dependency-free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    logger.info(f"Healthcheck performed with GPU check: {'available' if gpu else 'not available'}")
    return response.json({"state": "healthy", "gpu": gpu})


@app.route("/", methods=["POST"])
async def inference(request):
    try:
        all_inputs = request.json
    except Exception as e:
        logging.error("Invalid JSON received.")
        return json_response({"error": "Invalid JSON received from user"}, status=400)

    call_inputs = all_inputs.get("callInputs", None)
    stream_events = call_inputs and call_inputs.get("streamEvents", 0) != 0

    status_sender = None
    streaming_response = None
    try:
        if stream_events:
            streaming_response = await request.respond(content_type="application/x-ndjson")

            # Start the timer to send status updates every second
            status_instance = PercentageCompleteStatus()
            status_sender = PercentageCompleteStatusSender(status_instance, streaming_response)
            await status_sender.send_status(1)

        output = await diffusers_model.inference(all_inputs, streaming_response if stream_events else None, status_instance if stream_events else None)
        if output is None:
            return json_response({"error": "No output from model"}, status=500)
        if streaming_response:
            await streaming_response.send(json.dumps(output) + "\n")
            await streaming_response.eof()
        else:
            return json_response(output)
    except SanicException as e:
        logging.exception(f"Failed to send streaming response: {str(e)}")
        return json_response({"error": "Server error"}, status=500)
    except Exception as e:
        return await handle_exception(streaming_response if stream_events else None, e)
    finally:
        if status_sender:
            await status_sender.stop()  # Stop sending status updates
        if streaming_response:
            await streaming_response.eof()


async def handle_exception(stream, exception):
    """Handle exceptions by sending an error message over a stream or returning it."""
    streaming_string = "streaming " if stream else ""
    logging.exception(f"Error during {streaming_string}inference processing.")

    error_response = create_error_response(exception)
    if stream:
        error_response = json.dumps(error_response)
        try:
            await stream.send(error_response + "\n")
        except SanicException as e:
            logging.error(f"Failed to send error response after initial error: {str(e)}")
    else:
        return json_response(error_response)


def create_error_response(exception):
    """Create a standardized error response JSON structure."""
    return {
        "$error": {
            "code": "APP_INFERENCE_ERROR",
            "name": type(exception).__name__,
            "message": str(exception),
            "stack": traceback.format_exc(),
        }
    }


if __name__ == "__main__":
    # ESS: formerly had workers=1, which was leading to a multiprocessing bug
    app.run(host="0.0.0.0", port="8000", single_process=True, debug=True, access_log=True)
