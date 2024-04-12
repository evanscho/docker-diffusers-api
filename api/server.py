# Do not edit if deploying to Banana Serverless
# This file is boilerplate for the http server, and follows a strict interface.

# Instead, edit the init() and inference() functions in app.py

from sanic import Sanic, response
import subprocess
import app as diffusers_model
import traceback
import os
import json
import sys
from utils.logging import Tee
import logging

# Open the log file and create a Tee object that writes to both the log file and stdout/stderr
log_file = open('training.log', 'a')
sys.stdout = Tee(log_file, sys.stdout)
sys.stderr = Tee(log_file, sys.stderr)

# Create the http server app
app = Sanic("my_app")
app.config.CORS_ORIGINS = os.getenv("CORS_ORIGINS") or "*"
app.config.RESPONSE_TIMEOUT = 60 * 60  # 1 hour (training can be long)


@app.before_server_start
async def initialize(app, loop):
    # We do the model load-to-GPU step on server startup
    # so the model object is available globally for reuse
    await diffusers_model.init()


@app.route("/healthcheck", methods=["GET"])
def healthcheck(request):
    # Healthchecks verify that the environment is correct on Banana Serverless

    # Dependency-free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})


@app.route("/", methods=["POST"])
async def inference(request):
    # Inference POST handler at '/' is called for every http call from Banana

    try:
        all_inputs = response.json.loads(request.json)
    except:
        all_inputs = request.json

    call_inputs = all_inputs.get("callInputs", None)
    stream_events = call_inputs and call_inputs.get("streamEvents", 0) != 0

    streaming_response = None
    if stream_events:
        streaming_response = await request.respond(content_type="application/x-ndjson")

    try:
        output = await diffusers_model.inference(all_inputs, streaming_response)
    except Exception as exception:
        logging.exception(exception)

        output = {
            "$error": {
                "code": "APP_INFERENCE_ERROR",
                "name": type(exception).__name__,
                "message": str(exception),
                "stack": traceback.format_exc(),
            }
        }

    if stream_events:
        await streaming_response.send(json.dumps(output) + "\n")
    else:
        return response.json(output)


if __name__ == "__main__":
    # ESS: formerly had workers=1, which was leading to a multiprocessing bug
    app.run(host="0.0.0.0", port="8000", single_process=True, debug=True, access_log=True)
