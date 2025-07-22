import threading
import time
from collections import deque

import numpy as np
import zwoasi as asi


class CameraStream:
    """
    A class that continuously captures frames from an ASI camera using a background thread
    """

    def __init__(self, camera: asi.Camera):
        """
        Initializes the CameraStream object

        Parameters:
            camera (asi.Camera): The ASI camera object to capture frames from
        """
        super().__init__()
        self.camera = camera
        self.latest_frame = None  # Store the latest frame
        self._lock = threading.Lock()
        self.capture_time_buffer = deque(maxlen=6)  # Stores frame capture times to later compute FPS
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._stopped = False

    def start(self):
        """
        Starts the background thread and enables video mode
        """
        print("Starting camera stream thread\n")
        self.camera.start_video_capture()
        self._stopped = False
        self._thread.start()
        return self

    def _update(self):
        """
        The method that runs in the background thread continuously
        capturing frames and saving them to latest_frame
        """
        while not self._stopped:
            frame = self.camera.capture_video_frame()
            with self._lock:
                self.latest_frame = frame
                self.capture_time_buffer.append(time.time())  # Record current time for later FPS calculation

        # When stopping, end the camera video capture
        self.camera.stop_video_capture()
        print("Camera stream thread stopped")

    def get_fps(self) -> float:
        """
        Computes and returns the approximate FPS from the timestamps buffer.

        Returns:
            float: Estimated frames per second.
        """
        with self._lock:
            if len(self.capture_time_buffer) < 2:
                print("Buffer is not full enough to compute FPS")
                return 0.0

            time_elapsed: float = self.capture_time_buffer[1] - self.capture_time_buffer[0]

            if time_elapsed != 0:
                return (len(self.capture_time_buffer) - 1) / time_elapsed
            else:
                print("Cannot divide by 0. (time elapsed is 0)")
                return 0.0

    def read(self) -> np.ndarray:
        """
        Returns the most recent frame captured by the camera

        Returns:
            np.ndarray: The latest video frame (or None if not yet available)
        """
        with self._lock:
            return self.latest_frame

    def stop(self):
        """
        Stops the camera stream and waits for the background thread to finish
        """
        print("Stopping camera stream thread")
        self._stopped = True
        self._thread.join()
