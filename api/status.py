import json
import asyncio


class PercentageCompleteStatus:
    """Stores and updates the completion status of the current process"""

    def __init__(self):
        self.type = "init"
        self.progress = 0.0

    def update(self, type, progress):
        self.type = type
        self.progress = progress

    def get(self):
        progress_percentage = round(self.progress * 100, 1)
        return {"type": self.type, "progress": f"{progress_percentage}%"}


class PercentageCompleteStatusSender:
    """Sends completion status updates periodically (for use when streaming updates to a client)"""

    def __init__(self, status, response, loop=None):
        self.status = status
        self.response = response
        self.loop = loop or asyncio.get_event_loop()
        self.task = None
        self.running = asyncio.Event()  # Use an asyncio.Event for running

    async def _send_status_async(self, frequency):
        self.running.set()  # Signal that the loop is about to start
        try:
            while self.running.is_set():
                await self.response.send(json.dumps(self.status.get()) + "\n")
                await asyncio.sleep(frequency)
        except asyncio.CancelledError:
            print('Cancelled status update task')
        except Exception as e:
            print(f"Exception {type(e).__name__} in _send_status_async: {e}")

    async def send_status(self, frequency=1.0):
        """Start sending status updates every {frequency} seconds."""
        if self.task is not None:
            self.stop()  # Stop the existing task if it's running
        self.task = asyncio.create_task(self._send_status_async(frequency))
        await self.running.wait()  # Wait for the loop to start

    def stop(self):
        """Stop sending status updates."""
        if self.task and not self.task.done():
            self.running.clear()  # Clear the event to stop the loop
            # Don't actually cancel the task with self.task.cancel() since get into issues of
            # using multiple event loops when calling it from a different thread
            self.task = None
