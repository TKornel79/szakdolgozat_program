import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class PIDGains:
    kp:      float = 1.0
    ki:      float = 0.0
    kd:      float = 0.0
    i_max:   float = 2.0
    out_max: float = 5.0
    d_lpf_alpha: float = 0.2

class PIDAxis:

    def __init__(self, gains: PIDGains):
        self._g = gains
        self._integral    = 0.0
        self._prev_meas   = None
        self._prev_d_filt = 0.0
        self._prev_time   = None

    def reset(self) -> None:
        self._integral    = 0.0
        self._prev_meas   = None
        self._prev_d_filt = 0.0
        self._prev_time   = None

    def compute(self, setpoint: float, measurement: float,
                dt: Optional[float] = None) -> float:
        now = time.time()
        if dt is None:
            dt = (now - self._prev_time) if self._prev_time else 0.02
        self._prev_time = now
        dt = max(dt, 1e-4)

        error = setpoint - measurement

        p_out = self._g.kp * error

        self._integral += error * dt
        self._integral = _clamp(self._integral, -self._g.i_max, self._g.i_max)
        i_out = self._g.ki * self._integral

        if self._prev_meas is not None:
            d_raw = -(measurement - self._prev_meas) / dt
            alpha = self._g.d_lpf_alpha
            self._prev_d_filt = alpha * d_raw + (1 - alpha) * self._prev_d_filt
        self._prev_meas = measurement
        d_out = self._g.kd * self._prev_d_filt

        output = _clamp(p_out + i_out + d_out, -self._g.out_max, self._g.out_max)
        return output

class PositionPIDController:

    def __init__(self):
        self._x   = PIDAxis(PIDGains(**config.PID_X))
        self._y   = PIDAxis(PIDGains(**config.PID_Y))
        self._z   = PIDAxis(PIDGains(**config.PID_Z))
        self._yaw = PIDAxis(PIDGains(**config.PID_YAW))

    def reset(self) -> None:
        for ax in (self._x, self._y, self._z, self._yaw):
            ax.reset()

    def compute(self,
                target_pos_ned:  np.ndarray,
                current_pos_ned: np.ndarray,
                target_yaw_rad:  float,
                current_yaw_rad: float,
                dt: Optional[float] = None,
                max_speed: float = config.MAX_SPEED_MPS) -> tuple:
        vx = self._x.compute(target_pos_ned[0], current_pos_ned[0], dt)
        vy = self._y.compute(target_pos_ned[1], current_pos_ned[1], dt)
        vz = self._z.compute(target_pos_ned[2], current_pos_ned[2], dt)

        yaw_err = _angle_diff(target_yaw_rad, current_yaw_rad)
        yaw_rate = self._yaw.compute(0.0, -yaw_err, dt)

        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > max_speed:
            scale = max_speed / speed
            vx *= scale; vy *= scale; vz *= scale

        return np.array([vx, vy, vz], dtype=float), float(yaw_rate)

    def compute_station_keep(self,
                             hold_pos_ned:    np.ndarray,
                             current_pos_ned: np.ndarray,
                             hold_yaw_rad:    float,
                             current_yaw_rad: float,
                             dt: Optional[float] = None) -> tuple:
        return self.compute(hold_pos_ned, current_pos_ned,
                            hold_yaw_rad, current_yaw_rad,
                            dt=dt, max_speed=config.POSITION_HOLD_SPEED)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi
