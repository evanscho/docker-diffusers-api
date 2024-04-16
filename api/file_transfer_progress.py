import io
import re
import sys
import threading
import time
from utils.logging import Tee
import asyncio


def monitor_stderr(stderr_output, process_name, file_transfer_thread, status_instance):
    try:
        last_position = 0  # Keep track of the last read position

        while True:
            stderr_output.seek(last_position)  # Move to the last read position
            all_content = stderr_output.read()  # Read new content since last read
            if all_content:
                lines = all_content.splitlines()
                if lines:
                    last_line = lines[-1]
                    progress = parse_progress(last_line)
                    if progress is not None:
                        status_instance.update(process_name, progress)
                last_position = stderr_output.tell()  # Update the position after reading

            if not file_transfer_thread.is_alive():
                print('Exiting monitoring thread as file transfer thread has finished')
                break  # Exit if the file transfer thread has finished

            time.sleep(1)  # Sleep to wait for more data

    except Exception as e:
        print('Unhandling exception in monitoring stderr:', e)


def parse_progress(output):
    # Use regex to extract the progress
    pattern = re.compile(r'(\d+(?:\.\d+)?)([kMG]?)/(\d+(?:\.\d+)?)([kMG]?)')
    match = pattern.search(output)
    if match:
        current, current_unit, total, total_unit = match.groups()
        # Convert to bytes for uniformity
        current_bytes = convert_to_bytes(float(current), current_unit)
        total_bytes = convert_to_bytes(float(total), total_unit)
        progress = current_bytes / total_bytes if total_bytes > 0 else 0
        return progress
    return None


def convert_to_bytes(number, unit):
    unit_factors = {'': 1, 'k': 1024, 'M': 1024**2, 'G': 1024**3}
    return number * unit_factors[unit]


def perform_while_tracking_progress(function_to_run, function_kwargs, process_name, status_instance):
    result_container = [None]  # A simple list to store the result of the thread
    print(f'Starting file transfer thread for {process_name}')

    def target_for_file_transfer_thread():
        # Run the function and store the result
        result_container[0] = function_to_run(**function_kwargs)

    transfer_progress_output = io.StringIO()
    if isinstance(sys.stderr, Tee):
        sys.stderr.add_log_file(transfer_progress_output)
    else:
        sys.stderr = Tee(sys.stderr, transfer_progress_output)

    file_transfer_thread = threading.Thread(target=target_for_file_transfer_thread)
    file_transfer_thread.start()

    # Wait for a few seconds before starting the monitor thread so the file transfer thread can start more quickly
    time.sleep(3)

    monitor_thread = threading.Thread(target=monitor_stderr, args=(
        transfer_progress_output, process_name, file_transfer_thread, status_instance))
    monitor_thread.start()

    file_transfer_thread.join()
    monitor_thread.join()

    if isinstance(sys.stderr, Tee):
        sys.stderr.remove_log_file(transfer_progress_output)

    return result_container[0]


async def async_perform_while_tracking_progress(function_to_run, function_kwargs, process_name, status_instance):
    loop = asyncio.get_running_loop()

    # None uses the default executor (ThreadPoolExecutor)
    result = await loop.run_in_executor(None, perform_while_tracking_progress, function_to_run, function_kwargs, process_name, status_instance)

    return result
