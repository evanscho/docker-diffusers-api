import json
import os
import datetime
import time
import hashlib
from requests_futures.sessions import FuturesSession
from status import status as status_instance

# Print a blank line for readability
print()

# Mask sensitive environmental variables and print all env variables, including the masked ones
environ = os.environ.copy()
for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_AUTH_TOKEN"]:
    if environ.get(key, None):
        environ[key] = "XXX"
print(environ)
print()


# Function to get the current time in milliseconds
def get_now():
    return round(time.time() * 1000)

# Get the URL to send data to, and the optional signing key that's used to verify the data
SEND_URL = os.getenv("SEND_URL")
if SEND_URL == "":
    SEND_URL = None
SIGN_KEY = os.getenv("SIGN_KEY", "")
if SIGN_KEY == "":
    SIGN_KEY = None

# Create a future session for asynchronous requests
future_session = FuturesSession()

# Get the container ID, either from the environment variable or by reading the /proc/self/mountinfo file
container_id = os.getenv("CONTAINER_ID")
if not container_id:
    with open("/proc/self/mountinfo") as file:
        line = file.readline().strip()
        while line:
            if "/containers/" in line:
                container_id = line.split("/containers/")[-1]  # Take only text to the right
                container_id = container_id.split("/")[0]  # Take only text to the left
                break
            line = file.readline().strip()

# Flag to check if the first session has been initialized yet.
# The first is the only time the session shouldn't be cleared when a new session is initialized, because the init() function will have already added
# status updates to the session, which we don't want to clear.
is_first_session_initialized = False

# Function to initialize a new session (by clearing the current session)
def initialize_session(on_module_import=False):
    global session
    global is_first_session_initialized

    if on_module_import or is_first_session_initialized:
        # Initiate the session with the creation time (_ctime)
        session = {"_ctime": get_now()}
    else:
        is_first_session_initialized = True
        # Don't clear the session because the init() function will have already added status updates to the session, which we don't want to clear.

# Function to calculate and retrieve the duration of each process in the session
def get_process_durations():
    process_durations = {}

    # Iterate over each process in the session
    for process_name in session.keys():
        # Skip the '_ctime' entry as it's not a process, just the creation time of the session
        if process_name == "_ctime": continue

        # Get the start and end times of the process
        start_time = session[process_name].get("start", None)
        end_time = session[process_name].get("done", None)

        # If both start and end times are available, calculate the duration
        if start_time and end_time:
            duration = end_time - start_time
            process_durations.update({process_name: duration})
        else:
            # If either start or end time is missing, set the duration as -1
            process_durations.update({process_name: -1})

    return process_durations

# Asynchronous function to send status updates about a process
async def send_status_update(process_name: str, status: str, payload: dict = {}, options: dict = {}):
    # Get the current time
    current_time = get_now()

    # Retrieve the URL and signing key from the options or environment variables
    send_url = options.get("SEND_URL", SEND_URL)
    signing_key = options.get("SIGN_KEY", SIGN_KEY)

    # Add the start or done timestamp to the given process, within the current session
    if status == "start":
        session.update({process_name: {"start": current_time, "last_time": current_time}})
    elif status == "done":
        session[process_name].update({"done": current_time, "diff": current_time - session[process_name]["start"]})
    else:
        session[process_name]["last_time"] = current_time

    # Prepare the data payload
    data_payload = {
        "type": process_name,
        "status": status,
        "container_id": container_id,
        "time": current_time,
        "t": current_time - session["_ctime"],
        "tsl": current_time - session[process_name]["last_time"],
        "payload": payload,
    }

    # Update the status instance based on whether the process has just started or ended
    if status == "start":
        status_instance.update(process_name, 0.0)
    elif status == "done":
        status_instance.update(process_name, 1.0)

    # If a signing key is available, sign the data payload
    if send_url and signing_key:
        input_string = json.dumps(data_payload, separators=(",", ":")) + signing_key
        signature = hashlib.md5(input_string.encode("utf-8")).hexdigest()
        data_payload["sig"] = signature

    # Print the current time and data payload
    print(datetime.datetime.now(), data_payload)

    # If a send URL is available, send the data payload
    if send_url:
        future_session.post(send_url, json=data_payload)

    # If a response object is provided, send the data payload as a response
    response_option = options.get("response")
    if response_option:
        print("streaming above")
        await response_option.send(json.dumps(data_payload) + "\n")

# Initialize the session on module import
initialize_session(True)
