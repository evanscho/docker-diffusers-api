import sys


class Tee:
    """Class that writes to both a stream and a list of log files, names after 'tee' command in Unix"""

    def __init__(self, stream, log_files=None):
        self.stream = stream

        # Check if log_files is a list or a single log file-like object
        if isinstance(log_files, list):
            self.log_files = log_files
        elif log_files is not None:
            self.log_files = [log_files]  # Wrap a single log file in a list
        else:
            self.log_files = []

        # self.original_stream = sys.__stderr__  # Store original stdout or stderr

    def write(self, message):
        # Write to the primary stream
        self.stream.write(message)

        # Also write to all log files
        for log_file in self.log_files:
            log_file.write(message)

        self.flush()

        # Directly write debug message to the original stdout or stderr to avoid recursion
        # self.original_stream.write(f"Debug: Writing to log_file and stderr: {message}\n")

    def flush(self):
        # Flush the primary stream
        self.stream.flush()

        # Flush all log files
        for log_file in self.log_files:
            log_file.flush()

    def add_log_file(self, log_file):
        """Add a new log file to the list of log files being managed by Tee."""
        self.log_files.append(log_file)

    def remove_log_file(self, log_file):
        """Remove a log file from the list of log files being managed by Tee."""
        try:
            self.log_files.remove(log_file)
        except ValueError:
            print(f"Attempted to remove a log file that was not in the list: {log_file}")

    def __getattr__(self, name):
        # Delegate attribute lookups to the underlying stream object
        return getattr(self.stream, name)
