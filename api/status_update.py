import json
import os
import datetime
import time
import hashlib
import logging
from requests_futures.sessions import FuturesSession


class EventTimingsTracker:
    _instance = None  # Class attribute to hold the single instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EventTimingsTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # Ensures initialization happens only once

            # Initiate the run with the creation time (_ctime)
            self.run = {"_ctime": self.get_now()}

            # Opens a connection to the server, so create FuturesSession here to allow for persistence across requests
            # FuturesSession is a subclass of Session that returns a Future object for each request, so the request is non-blocking
            self.future_session = FuturesSession()

            # Get the URL to send data to, and the optional signing key that's used to verify the data
            # If variable is set but is an empty string, os.getenv() returns "" instead of None
            self.send_url = os.getenv("SEND_URL") or None
            self.signing_key = os.getenv("SIGN_KEY") or None

            self.container_id = self.retrieve_container_id()
            self.is_first_run_after_server_start = True
            self.initialized = True

    def start_new_run(self):
        """Start a new run by clearing the current run and initializing a new one"""
        """Unless it's the first run after the server starts, in which case the run was already initialized in init() and we don't want to clear it"""
        if self.is_first_run_after_server_start:
            self.is_first_run_after_server_start = False
        else:
            # Initiate the run with the creation time (_ctime)
            self.run = {"_ctime": self.get_now()}

        return self

    def get_now(self):
        """Get the current time in milliseconds"""
        return round(time.time() * 1000)

    def retrieve_container_id(self):
        """Retrieve the container ID from environment or filesystem"""
        container_id = os.getenv("CONTAINER_ID")
        if not container_id:
            try:
                with open("/proc/self/mountinfo") as file:
                    line = file.readline().strip()
                    while line:
                        if "/containers/" in line:
                            container_id = line.split("/containers/")[-1].split("/")[0]
                            break
                        line = file.readline().strip()
            except FileNotFoundError:
                logging.error("mountinfo not found, running outside a container?")
        return container_id

    def get_process_durations(self):
        """Calculate and retrieve the duration of each process in the run"""
        process_durations = {}

        for process_name in self.run.keys():
            # Skip the '_ctime' entry as it's not a process, just the creation time of the run
            if process_name == "_ctime":
                continue

            # If both start and end times are available, calculate the duration
            start_time = self.run[process_name].get("start")
            end_time = self.run[process_name].get("done")
            if start_time and end_time:
                duration = end_time - start_time
                process_durations[process_name] = duration
            else:
                # If either start or end time is missing, set the duration as -1
                process_durations[process_name] = -1

        return process_durations

    def update_process_with_current_timestamp(self, process_name, status, current_time):
        """Add the start or done timestamp to the given process, within the current run"""
        if status == "start":
            self.run[process_name] = {"start": current_time, "last_time": current_time}
        elif status == "done":
            if process_name in self.run and "start" in self.run[process_name]:
                self.run[process_name].update({
                    "done": current_time,
                    "diff": current_time - self.run[process_name]["start"]
                })
            else:
                logging.warning(
                    f'Process "{process_name}" does not have a start time so cannot log the end time')
        else:
            if process_name in self.run and "start" in self.run[process_name]:
                self.run[process_name]["last_time"] = current_time
            else:
                logging.warning(f'Process "{process_name}" did not have a starting time logged')
                self.run[process_name] = {"last_time": current_time}

    async def send_event_update(self, process_name, status, payload={}, options={}):
        """Asynchronously send status updates about an event (typically "start" or "done" of a process)"""

        current_time = self.get_now()
        self.update_process_with_current_timestamp(process_name, status, current_time)

        data_payload = {
            "type": process_name,
            "status": status,
            "container_id": self.container_id,
            "time": current_time,
            "t": current_time - self.run["_ctime"],
            "tsl": current_time - self.run[process_name]["last_time"],
            "payload": payload,
        }

        # Update the status instance based on whether the process has just started or ended
        status_instance = options.get("status_instance", None)
        if status_instance:
            if status == "start":
                status_instance.update(process_name, 0.0)
            elif status == "done":
                status_instance.update(process_name, 1.0)

        # If a signing key is available, sign the data payload
        if self.send_url:
            if self.signing_key:
                input_string = json.dumps(data_payload, separators=(",", ":")) + self.signing_key
                signature = hashlib.md5(input_string.encode("utf-8")).hexdigest()
                data_payload["sig"] = signature
            self.future_session.post(self.send_url, json=data_payload)

        # If a response object is provided, send the data payload as a response
        response_option = options.get("response")
        if response_option:
            await response_option.send(json.dumps(data_payload) + "\n")

        # Print the data payload to the server even if not sending it anywhere, such as during init()
        print(datetime.datetime.now(), data_payload)
