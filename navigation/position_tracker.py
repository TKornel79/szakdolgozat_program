
import time
import threading
from typing import Optional

import numpy as np

import config
from vision.pose_estimator import PoseEstimate
from utils.logger import get_logger

log = get_logger(__name__)

_DIM = 7
def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def _angle_diff(a: float, b: float) -> float:
    return _wrap(a - b)

class PositionTracker:
    def __init__(self):
        self._lock = threading.Lock()

        self._initialized   = False
        self._last_update   = time.time()
        self._last_predict  = time.time()
        self._x = np.zeros(_DIM)
        self._P = np.eye(_DIM) * 100.0

        qp = config.EKF_PROCESS_NOISE_POS
        qv = config.EKF_PROCESS_NOISE_VEL
        self._Q = np.diag([qp, qp, qp, qv, qv, qv, 0.01]) ** 2

        rp = config.EKF_MEAS_NOISE_POS
        ry = config.EKF_MEAS_NOISE_YAW
        rv = config.EKF_MEAS_NOISE_VEL
        self._R_pose = np.diag([rp, rp, rp, ry]) ** 2
        self._R_pos  = np.diag([rp, rp, rp])     ** 2
        self._R_vel  = np.diag([rv, rv, rv])      ** 2

    @property
    def is_initialized(self) -> bool:
        with self._lock:
            return self._initialized

    def initialize(self, pos_ned: np.ndarray, yaw: float = 0.0) -> None:
        with self._lock:
            self._initialize_locked(pos_ned, yaw)

    def predict(self) -> None:
        with self._lock:
            if not self._initialized:
                return
            now = time.time()
            dt  = now - self._last_predict
            self._last_predict = now
            F = self._build_F(dt)
            self._x = F @ self._x
            self._x[6] = _wrap(self._x[6])
            self._P = F @ self._P @ F.T + self._Q

    def update_vision(self, estimate: PoseEstimate) -> None:
        with self._lock:
            if not self._initialized:
                self._initialize_locked(estimate.position_ned, estimate.yaw_rad)
                return

            scale = 1.0 / max(estimate.confidence, 0.1)
            R = self._R_pose * scale

            z = np.array([*estimate.position_ned, estimate.yaw_rad])
            H = np.zeros((4, _DIM))
            H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = 1.0

            innov = z - H @ self._x
            innov[3] = _angle_diff(z[3], (H @ self._x)[3])

            self._x, self._P = self._ekf_update(self._x, self._P, H, R, innov)
            self._x[6] = _wrap(self._x[6])
            self._last_update = time.time()

    def update_velocity(self, vel_ned: np.ndarray,
                        confidence: float = 0.8) -> None:
        with self._lock:
            if not self._initialized:
                return
            R = self._R_vel / max(confidence, 0.1)
            H = np.zeros((3, _DIM))
            H[0, 3] = H[1, 4] = H[2, 5] = 1.0
            innov = vel_ned - H @ self._x
            self._x, self._P = self._ekf_update(self._x, self._P, H, R, innov)

    def update_position_only(self, pos_ned: np.ndarray,
                             confidence: float = 1.0) -> None:
        with self._lock:
            if not self._initialized:
                return
            R = self._R_pos / max(confidence, 0.1)
            H = np.zeros((3, _DIM))
            H[0, 0] = H[1, 1] = H[2, 2] = 1.0
            innov = pos_ned - H @ self._x
            self._x, self._P = self._ekf_update(self._x, self._P, H, R, innov)
            self._last_update = time.time()

    def get_state(self) -> Optional[dict]:
        with self._lock:
            if not self._initialized:
                return None
            return {
                "pos_ned":         self._x[:3].copy(),
                "vel_ned":         self._x[3:6].copy(),
                "yaw_rad":         float(self._x[6]),
                "uncertainty_pos": np.sqrt(np.diag(self._P)[:3]),
                "uncertainty_yaw": float(np.sqrt(self._P[6, 6])),
                "age_s":           time.time() - self._last_update,
            }

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            return (self._initialized and
                    (time.time() - self._last_update) < config.WATCHDOG_TIMEOUT_S)

    def _initialize_locked(self, pos_ned: np.ndarray, yaw: float) -> None:
        self._x = np.zeros(_DIM)
        self._x[:3] = pos_ned
        self._x[6]  = yaw
        self._P = np.diag([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.1]) ** 2
        self._initialized  = True
        now = time.time()
        self._last_update  = now
        self._last_predict = now
        log.info("Pozícióbecslő (EKF) elindult: É=%.2f K=%.2f Le=%.2f m, irány=%.1f°",
                 *pos_ned, np.degrees(yaw))

    @staticmethod
    def _ekf_update(x, P, H, R, innov):
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_new = x + K @ innov
        IKH   = np.eye(len(x)) - K @ H
        P_new = IKH @ P @ IKH.T + K @ R @ K.T
        P_new = (P_new + P_new.T) * 0.5
        return x_new, P_new

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        F = np.eye(_DIM)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F
