
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)

class CameraCapture:
    def __init__(self, index: int = config.CAMERA_INDEX,
                 width: int = config.FRAME_WIDTH,
                 height: int = config.FRAME_HEIGHT,
                 fps: int = config.CAMERA_FPS):
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_frame_time: float = 0.0
        self._frame_id = 0          # minden új képkockánál nő; ebből látja a main.py, hogy jött-e új kép

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            log.error("Nem sikerült megnyitni a kamerát: %s", self._index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name="camera-capture")
        self._thread.start()
        log.info("Kamera elindult: %s (%dx%d, %d fps)", self._index,
                 self._width, self._height, self._fps)
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        log.debug("Kamera leállítva. Összesen %d kép érkezett.", self._frame_count)

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_new_frame(self, last_id: int):
        """(kép, azonosító), ha a last_id óta jött új képkocka; különben (None, last_id)."""
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame.copy(), self._frame_id

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_healthy(self) -> bool:
        return (self._last_frame_time > 0.0 and
                (time.time() - self._last_frame_time) < config.CAMERA_HEALTH_TIMEOUT_S)

    _FAILURE_THRESHOLD = 10
    _RECONNECT_BACKOFF = 3.0
    def _capture_loop(self) -> None:
        interval = 1.0 / self._fps
        consecutive_failures = 0
        while self._running:
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= self._FAILURE_THRESHOLD:
                    log.error("Kamera %s: %d egymás utáni hiba — újracsatlakozás …",
                              self._index, consecutive_failures)
                    self._cap.release()
                    time.sleep(self._RECONNECT_BACKOFF)
                    self._cap = cv2.VideoCapture(self._index)
                    if self._cap.isOpened():
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        log.info("Kamera újracsatlakozott: %s", self._index)
                        consecutive_failures = 0
                    else:
                        log.warning("Kamera %s: az újracsatlakozás nem sikerült, újra %.1f s múlva.",
                                    self._index, self._RECONNECT_BACKOFF)
                else:
                    time.sleep(interval)
                continue
            consecutive_failures = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._last_frame_time = time.time()
            self._frame_count += 1
            elapsed = time.time() - t0
            sleep_s = max(0.0, interval - elapsed)
            time.sleep(sleep_s)
