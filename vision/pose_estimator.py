import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import config
from vision.marker_detector import DetectedMarker
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class PoseEstimate:
    position_ned: np.ndarray
    yaw_rad: float
    confidence: float
    n_markers: int


_R_CAM_TO_BODY = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
], dtype=np.float64)

_MIN_RAY_DOWN = 0.2


def _rotation_body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr           ],
    ], dtype=np.float64)


@dataclass
class _Obs:
    marker: DetectedMarker
    cam_to_dot: np.ndarray
    cam_off: np.ndarray
    weight: float


class PoseEstimator:
    def __init__(self,
                 camera_matrix: np.ndarray = config.CAMERA_MATRIX,
                 dist_coeffs: np.ndarray   = config.DIST_COEFFS,
                 map_free: Optional[bool]  = None,
                 dot_map: Optional[list]   = None):
        self._K    = camera_matrix
        self._dist = dist_coeffs
        self._fx   = camera_matrix[0, 0]

        self._dot_d    = float(config.DOT_DIAMETER_M)
        self._w_size   = float(getattr(config, "DOT_SIZE_ALT_WEIGHT", 0.5))
        self._t_cam    = np.asarray(config.T_BODY_TO_CAM, dtype=np.float64)

        self.map_free = bool(getattr(config, "MAP_FREE", False) if map_free is None else map_free)
        if self.map_free:
            self._gate_m = float(getattr(config, "DOT_ASSOC_GATE_MAPFREE_M", 2.0))
        else:
            self._gate_m = float(getattr(config, "DOT_ASSOC_GATE_M", 4.0))
        self._new_dot_max_m  = float(getattr(config, "NEW_DOT_MAX_DIST_M", 3.5))
        self._new_dot_sep_m  = float(getattr(config, "NEW_DOT_MIN_SEPARATION_M", 3.0))
        self._assoc_margin_m = float(getattr(config, "DOT_ASSOC_MARGIN_M", 2.0))

        self._landmarks: List[dict] = []
        if not self.map_free:
            src = getattr(config, "DOT_MAP", []) if dot_map is None else dot_map
            for d in src:
                self._landmarks.append(dict(color=d["color"], n=0, anchored=True,
                                            pos=np.array([d["n"], d["e"], d.get("z", 0.0)], dtype=np.float64)))
            if not self._landmarks:
                log.warning("config.DOT_MAP üres: a markerek nem adnak pozíció-fixet.")
        else:
            log.info("Térkép nélküli mód: a markerek helyét menet közben, a markerlánc alapján tanulom meg.")


    @property
    def landmarks(self) -> List[dict]:
        return [dict(l, pos=l["pos"].copy()) for l in self._landmarks]

    def estimate(self, markers: List[DetectedMarker],
                 altitude_m: Optional[float] = None,
                 current_yaw_rad: float = 0.0,
                 roll_rad: float = 0.0,
                 pitch_rad: float = 0.0,
                 predicted_pos_ned: Optional[np.ndarray] = None,
                 altitude_weight: float = 1.0
                 ) -> Optional[PoseEstimate]:
        R_b2n = _rotation_body_to_ned(roll_rad, pitch_rad, current_yaw_rad)

        obs: List[_Obs] = []
        for m in markers:
            if m.strategy != "color" or m.touches_border or m.center_px is None:
                continue
            o = self._observe(m, altitude_m, altitude_weight, R_b2n)
            if o is not None:
                obs.append(o)
        if not obs:
            return None

        matched, unmatched = [], []
        for o in obs:
            res = self._associate(o, predicted_pos_ned)
            if res is None:
                unmatched.append(o)
            elif res == "ambiguous":
                continue
            else:
                lm, implied = res
                matched.append((lm, o, implied))

        pose = None
        if matched:
            total_w = sum(o.weight for _, o, _ in matched)
            pos_ned = sum(imp * o.weight for _, o, imp in matched) / total_w
            for lm, _, _ in matched:
                lm["n"] += 1
            pose = PoseEstimate(position_ned=pos_ned, yaw_rad=current_yaw_rad,
                                confidence=min(total_w / len(matched), 1.0),
                                n_markers=len(matched))

        if self.map_free and unmatched:
            base = pose.position_ned if pose is not None else predicted_pos_ned
            if base is not None:
                for o in unmatched:
                    self._maybe_add(o, base, anchored=pose is not None)

        return pose


    def _ray_ned(self, center_px: np.ndarray, R_b2n: np.ndarray) -> np.ndarray:
        pt = np.asarray(center_px, dtype=np.float64).reshape(1, 1, 2)
        nx, ny = cv2.undistortPoints(pt, self._K, self._dist)[0, 0]
        return R_b2n @ (_R_CAM_TO_BODY @ np.array([nx, ny, 1.0]))

    def _observe(self, m: DetectedMarker, altitude_m: Optional[float],
                 altitude_weight: float, R_b2n: np.ndarray) -> Optional[_Obs]:
        ray = self._ray_ned(m.center_px, R_b2n)
        if ray[2] < _MIN_RAY_DOWN:
            return None
        cam_off = R_b2n @ self._t_cam

        heights, weights = [], []
        if altitude_m is not None and not math.isnan(altitude_m):
            heights.append(altitude_m - cam_off[2])
            weights.append(max(altitude_weight, 0.0))
        if m.diameter_px > 0 and self._w_size > 0:
            depth = self._fx * self._dot_d / m.diameter_px
            heights.append(depth * ray[2])
            weights.append(self._w_size)
        if not heights or sum(weights) <= 0:
            return None
        h_cam = float(np.dot(heights, weights) / np.sum(weights))

        cam_to_dot = (h_cam / ray[2]) * ray
        slant = float(np.linalg.norm(cam_to_dot))
        weight = m.confidence / (1.0 + 0.02 * slant)
        return _Obs(marker=m, cam_to_dot=cam_to_dot, cam_off=cam_off, weight=weight)

    def _associate(self, o: _Obs, predicted: Optional[np.ndarray]):
        cands = [lm for lm in self._landmarks if lm["color"] == o.marker.color]
        if not cands:
            return None
        implied = [lm["pos"] - o.cam_to_dot - o.cam_off for lm in cands]
        if predicted is None:
            return (cands[0], implied[0]) if len(cands) == 1 else "ambiguous"
        dists = [float(np.linalg.norm(q[:2] - predicted[:2])) for q in implied]
        order = np.argsort(dists)
        i = int(order[0])
        if dists[i] > self._gate_m:
            second = dists[int(order[1])] if len(order) > 1 else float("inf")
            if (self.map_free and dists[i] <= self._new_dot_sep_m
                    and second - dists[i] >= self._assoc_margin_m):
                log.debug("Marker %s: %.1f m-re a jósolt helytől, de egyértelmű — párosítom.",
                          o.marker.color, dists[i])
                o.weight *= 0.5
                return cands[i], implied[i]
            log.debug("Marker %s: nincs párja %.1f m-en belül (legközelebbi %.1f m)",
                      o.marker.color, self._gate_m, dists[i])
            return None
        if (len(order) > 1 and dists[int(order[1])] <= self._gate_m
                and dists[int(order[1])] - dists[i] < 1.0):
            return "ambiguous"
        return cands[i], implied[i]

    def _maybe_add(self, o: _Obs, base_pos: np.ndarray, anchored: bool) -> None:
        horiz = float(np.hypot(o.cam_to_dot[0] + o.cam_off[0], o.cam_to_dot[1] + o.cam_off[1]))
        if horiz > self._new_dot_max_m:
            return
        world = base_pos + o.cam_off + o.cam_to_dot
        world[2] = 0.0
        for lm in self._landmarks:
            if lm["color"] == o.marker.color and np.linalg.norm(lm["pos"][:2] - world[:2]) < self._new_dot_sep_m:
                return
        self._landmarks.append(dict(color=o.marker.color, pos=world, anchored=anchored, n=1))
        log.debug("Új marker a láncban: %s É=%.2f K=%.2f (%s)", o.marker.color, world[0], world[1],
                  "kézátadással" if anchored else "becsült helyzetből")
