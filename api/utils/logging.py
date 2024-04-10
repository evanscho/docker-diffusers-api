class Tee:
    def __init__(self, log_file, stream):
        self.log_file = log_file
        self.stream = stream

    def write(self, message):
        self.log_file.write(message)
        self.stream.write(message)
        self.flush()

    def flush(self):
        self.log_file.flush()
        self.stream.flush()

    # Delegate any attribute lookups that aren't found in Tee (like isatty) to the underlying stream object (sys.stdout or sys.stderr)
    def __getattr__(self, name):
        return getattr(self.stream, name)
