from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

import config
from utils.logger import get_logger

log = get_logger(__name__)


COLOR_NAMES_HU = {
    "blue":   "KÉK",
    "black":  "FEKETE",
    "yellow": "SÁRGA",
    "red":    "PIROS",
    "green":  "ZÖLD",
    "purple": "LILA",
}


@dataclass
class DetectedMarker:
    marker_id: str
    strategy: str
    corners_px: np.ndarray
    world_pos: np.ndarray
    confidence: float = 1.0
    color: str = ""
    center_px: Optional[np.ndarray] = None
    diameter_px: float = 0.0
    touches_border: bool = False


class ColorMarkerDetector:

    _SHAPE_SCORE_CIRC_THRESH = 0.6

    def __init__(self):
        self._max_per_color = int(getattr(config, "DOT_MAX_CANDIDATES_PER_COLOR",
                                          getattr(config, "DOT_MAX_PER_COLOR", 3)))
        self._scale = float(getattr(config, "DETECT_SCALE", 0.5))
        self._use_clahe = bool(getattr(config, "DETECT_CLAHE", True))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        k = max(3, int(round(5 * self._scale)) | 1)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        used = set(getattr(config, "MARKER_COMMANDS", {}).keys())
        used |= {d["color"] for d in getattr(config, "DOT_MAP", [])}
        self._colors = {name: cfg for name, cfg in config.COLOR_MARKERS.items() if name in used}
        if not self._colors:
            self._colors = dict(config.COLOR_MARKERS)
        log.info("Markerfelismerés: %s színek, %.0f%%-os képméret, CLAHE %s",
                 ", ".join(COLOR_NAMES_HU.get(c, c) for c in self._colors),
                 100 * self._scale, "be" if self._use_clahe else "ki")

    def detect(self, frame: np.ndarray) -> List[DetectedMarker]:
        s = self._scale
        if s != 1.0:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img_h, img_w = frame.shape[:2]

        if self._use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        results: List[DetectedMarker] = []

        for color_name, cfg in self._colors.items():
            mask = None
            for lo, hi in cfg["hsv_ranges"]:
                m = cv2.inRange(hsv, lo, hi)
                mask = m if mask is None else (mask | m)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            min_area = cfg["min_area_px"] * s * s
            found = 0
            for cnt in contours:
                if found >= self._max_per_color:
                    break
                area = cv2.contourArea(cnt)
                if area < min_area:
                    break

                if len(cnt) >= 5:
                    (_, _), (ax1, ax2), _ = cv2.fitEllipse(cnt)
                    major, minor = max(ax1, ax2), min(ax1, ax2)
                else:
                    major = minor = 2.0 * np.sqrt(area / np.pi)
                if major <= 0 or minor <= 0:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                a_ax, b_ax = major / 2.0, minor / 2.0
                h_ram = ((a_ax - b_ax) / (a_ax + b_ax)) ** 2
                p_ell = np.pi * (a_ax + b_ax) * (1 + 3 * h_ram / (10 + np.sqrt(4 - 3 * h_ram)))
                circ_ideal = 4 * np.pi * (np.pi * a_ax * b_ax) / (p_ell ** 2)
                shape = circularity / max(circ_ideal, 1e-6)
                confidence = float(min(shape / self._SHAPE_SCORE_CIRC_THRESH, 1.0))

                M = cv2.moments(cnt)
                cx = M["m10"] / (M["m00"] + 1e-6)
                cy = M["m01"] / (M["m00"] + 1e-6)

                x, y, w, h = cv2.boundingRect(cnt)
                touches = x <= 1 or y <= 1 or x + w >= img_w - 1 or y + h >= img_h - 1

                cx_f = (cx + 0.5) / s - 0.5
                cy_f = (cy + 0.5) / s - 0.5
                major_f = major / s
                r = major_f / 2.0
                corners = np.array([
                    [cx_f - r, cy_f - r],
                    [cx_f + r, cy_f - r],
                    [cx_f + r, cy_f + r],
                    [cx_f - r, cy_f + r],
                ], dtype=np.float64)

                results.append(DetectedMarker(
                    marker_id=f"color_{color_name}",
                    strategy="color",
                    corners_px=corners,
                    world_pos=cfg["world_pos"],
                    confidence=confidence,
                    color=color_name,
                    center_px=np.array([cx_f, cy_f], dtype=np.float64),
                    diameter_px=float(major_f),
                    touches_border=bool(touches),
                ))
                found += 1

        return results


class MarkerDetector:

    def __init__(self):
        self._color = ColorMarkerDetector()

    def detect(self, frame: np.ndarray, debug: bool = False
               ) -> List[DetectedMarker]:
        markers: List[DetectedMarker] = self._color.detect(frame)

        if debug and markers:
            self._draw(frame, markers)

        return markers

    @staticmethod
    def _draw(frame: np.ndarray, markers: List[DetectedMarker]) -> None:
        for m in markers:
            col = (128, 128, 128) if m.touches_border else (0, 165, 255)
            cx, cy = (int(v) for v in m.center_px)
            cv2.circle(frame, (cx, cy), max(int(m.diameter_px / 2), 2), col, 2)
            cv2.putText(frame, f"{m.marker_id}({m.confidence:.2f})",
                        (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1, cv2.LINE_AA)
