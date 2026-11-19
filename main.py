import math
import signal
from collections import deque
import sys
import time
from enum import Enum, auto
from typing import Optional

import numpy as np

import config
from mavlink_interface.connection import MAVLinkConnection
from mavlink_interface.controller import DroneController
from navigation.position_tracker import PositionTracker
from motion_planning.pid_controller import PositionPIDController
from vision.camera import CameraCapture
from vision.marker_detector import MarkerDetector
from vision.pose_estimator import PoseEstimator, PoseEstimate
from utils.logger import get_logger

log = get_logger(__name__)

class State(Enum):
    INIT             = auto()
    PREFLIGHT        = auto()
    ARMING           = auto()
    TAKEOFF          = auto()
    MARKER_FOLLOWING = auto()
    SEARCHING        = auto()
    HOVER            = auto()
    LAND             = auto()
    DONE             = auto()
    EMERGENCY        = auto()
    PAUSED           = auto()
    RETURN_HOME      = auto()

_GROUND_STATES = {State.INIT, State.PREFLIGHT, State.ARMING,
                  State.TAKEOFF, State.LAND, State.DONE}

_COMMAND_MARKER_IDS = {f"color_{c}" for c in config.MARKER_COMMANDS}

_SEV_CRITICAL = 2
_SEV_WARNING  = 4
_SEV_NOTICE   = 5
_SEV_INFO     = 6

_STATE_HU = {
    State.INIT:             "INDULÁS",
    State.PREFLIGHT:        "FELSZÁLLÁS ELŐTTI ELLENŐRZÉS",
    State.ARMING:           "ÉLESÍTÉS",
    State.TAKEOFF:          "FELSZÁLLÁS",
    State.MARKER_FOLLOWING: "MARKERKÖVETÉS",
    State.SEARCHING:        "KERESÉS (360°)",
    State.HOVER:            "LEBEGÉS",
    State.LAND:             "LESZÁLLÁS",
    State.DONE:             "KÉSZ",
    State.EMERGENCY:        "VÉSZHELYZET",
    State.PAUSED:           "SZÜNET (kézi irányítás)",
    State.RETURN_HOME:      "VISSZATÉRÉS",
}

_COLOR_HU = {"blue": "KÉK", "black": "FEKETE", "yellow": "SÁRGA", "red": "PIROS",
             "green": "ZÖLD", "purple": "LILA"}
_CMD_HU = {"straight": "egyenesen tovább", "turn_right": "jobbra fordulás",
           "turn_left": "balra fordulás", "land": "leszállás"}


def _hu_color(marker_id: str) -> str:
    c = marker_id.replace("color_", "")
    return _COLOR_HU.get(c, c)


_ASCII = str.maketrans("áéíóöőúüűÁÉÍÓÖŐÚÜŰ°", "aeiooouuuAEIOOOUUU ")


def _ascii(text: str) -> str:
    return text.translate(_ASCII).encode("ascii", "replace").decode("ascii")


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + math.pi) % (2 * math.pi) - math.pi

def _simplify_path(points, tol: float):
    pts = [np.asarray(p, dtype=float) for p in points]
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-6:
        dists = [float(np.linalg.norm(p - a)) for p in pts[1:-1]]
    else:
        dists = [abs(float(ab[0] * (p - a)[1] - ab[1] * (p - a)[0])) / n for p in pts[1:-1]]
    k = int(np.argmax(dists)) + 1
    if dists[k - 1] <= tol:
        return [a, b]
    left = _simplify_path(pts[:k + 1], tol)
    right = _simplify_path(pts[k:], tol)
    return left[:-1] + right


def _dot_plausible(m, altitude_m: Optional[float],
                   roll: float = 0.0, pitch: float = 0.0) -> bool:
    if m.confidence < getattr(config, "DOT_MIN_SHAPE_CONF", 0.0):
        return False
    if not altitude_m or altitude_m <= 0.3 or m.center_px is None or m.diameter_px <= 0:
        return True
    K = config.CAMERA_MATRIX
    nx = (float(m.center_px[0]) - K[0, 2]) / K[0, 0]
    ny = (float(m.center_px[1]) - K[1, 2]) / K[1, 1]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ray_down = -sp * (-ny) + cp * sr * nx + cp * cr * 1.0
    if ray_down < 0.2:
        return True
    h_cam = altitude_m - float(config.T_BODY_TO_CAM[2])
    expected = K[0, 0] * config.DOT_DIAMETER_M * ray_down / h_cam
    tol = getattr(config, "DOT_SIZE_TOLERANCE", 0.35)
    return (1.0 - tol) <= m.diameter_px / expected <= (1.0 + tol)


class NavigationSystem:
    def __init__(self):
        self._state   = State.INIT
        self._running = False

        self._conn      = MAVLinkConnection()
        self._ctrl: Optional[DroneController] = None
        self._camera    = CameraCapture()
        self._detector  = MarkerDetector()
        self._estimator = PoseEstimator()
        self._tracker   = PositionTracker()
        self._pid       = PositionPIDController()

        self._last_fix_time: float = 0.0
        self._hold_pos: Optional[np.ndarray] = None
        self._hold_yaw: float = 0.0
        self._hover_start: float = 0.0

        self._takeoff_sent:  bool  = False
        self._takeoff_start: float = 0.0

        self._land_sent:  bool  = False
        self._land_start: float = 0.0

        self._preflight_params_done: bool  = False
        self._preflight_start:       float = 0.0

        self._guided_mode_set:       bool  = False

        self._current_heading:       float = 0.0
        self._marker_cooldown_until: float = 0.0
        self._tracked_cmd_marker:    Optional[tuple] = None
        self._spent_dots: deque = deque(maxlen=int(getattr(config, "MARKER_SPENT_MEMORY", 3)))
        self._search_start_time:     float = 0.0

        self._battery_warning_sent:  bool  = False
        self._camera_warning_sent:   bool  = False
        self._land_reason:           str   = ""

        self._last_frame_id:         int   = -1
        self._line_dots: list = []
        self._line_dots_t:           float = 0.0
        self._lat_i:                 float = 0.0
        self._xtrack_ema:            float = 0.0
        self._drift_since                  = None
        self._last_drift_warn:       float = 0.0
        self._last_wind_warn:        float = 0.0
        self._last_hover_drift_warn: float = 0.0
        self._slowing:               bool  = False
        self._vision_seq:            int   = 0
        self._confirm_color                = None
        self._confirm_count:         int   = 0
        self._confirm_seq:           int   = -1
        self._alt_source                   = None
        self._route: list = []
        self._returning:             bool  = False
        self._rth_path: list = []
        self._rth_seg:  int  = 0
        self._rth_turning: bool = False
        self._rth_start_batt               = None
        self._follow_start_t               = None
        self._follow_start_batt            = None
        self._announced_dots: deque = deque(maxlen=20)
        self._tick_count:            int   = 0
        self._slow_ticks:            int   = 0
        self._slowest_tick_s:        float = 0.0
        self._recent_slow: deque = deque(maxlen=int(10 * config.CONTROL_LOOP_HZ))
        self._last_slow_warning:     float = 0.0
        self._frames_processed:      int   = 0
        self._ticks_without_frame:   int   = 0

        self._origin_sends:          int   = 0
        self._last_origin_send:      float = 0.0

        self._stop_requested:        bool  = False
        self._land_cmd_ok:           bool  = False
        self._land_last_try:         float = 0.0
        self._emerg_started:         bool  = False
        self._emerg_start:           float = 0.0
        self._last_hb_sent:          float = 0.0
        self._dots_rejected:         int   = 0
        self._alt_violation_since          = None

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self) -> None:
        log.info("=== GPS nélküli, kamerás navigáció indul ===")

        cm = config.CAMERA_MATRIX
        if cm[0, 0] == 800.0 and cm[1, 1] == 800.0:
            log.warning("KAMERA: alapértelmezett kameraparaméterek (fx=fy=800). "
                        "Valós repülés előtt kalibráld a kamerát (cv2.calibrateCamera), "
                        "és frissítsd a config.CAMERA_MATRIX értékét.")

        if not self._conn.connect():
            log.error("Nem sikerült csatlakozni a repülésvezérlőhöz.")
            sys.exit(1)
        self._ctrl = DroneController(self._conn)
        self._request_message_rates()

        if not self._camera.start():
            log.error("A kamera nem érhető el.")
            self._conn.disconnect()
            sys.exit(1)

        self._running = True
        try:
            self._run_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        self._log_stats()
        self._running = False
        self._camera.stop()
        self._conn.disconnect()

    def _run_loop(self) -> None:
        dt = 1.0 / config.CONTROL_LOOP_HZ
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as exc:
                log.exception("Váratlan hiba a vezérlőciklusban: %s", exc)
                self._transition(State.EMERGENCY)
            elapsed = time.time() - t0
            self._note_tick_time(elapsed, dt)
            time.sleep(max(0.0, dt - elapsed))

    def _tick(self) -> None:
        self._send_heartbeat()
        if self._stop_requested:
            self._handle_stop_request()

        self._tracker.predict()

        frame = self._get_new_frame()
        if frame is not None:
            self._frames_processed += 1
            self._process_vision(frame)
        else:
            self._ticks_without_frame += 1

        self._fuse_ardupilot_velocity()

        self._send_vision_feedback()

        if not self._safety_ok():
            return

        {
            State.INIT:             self._state_init,
            State.PREFLIGHT:        self._state_preflight,
            State.ARMING:           self._state_arming,
            State.TAKEOFF:          self._state_takeoff,
            State.MARKER_FOLLOWING: self._state_marker_following,
            State.SEARCHING:        self._state_searching,
            State.HOVER:            self._state_hover,
            State.LAND:             self._state_land,
            State.DONE:             self._state_done,
            State.EMERGENCY:        self._state_emergency,
            State.PAUSED:           self._state_paused,
            State.RETURN_HOME:      self._state_return_home,
        }.get(self._state, lambda: None)()

    def _get_new_frame(self):
        getter = getattr(self._camera, "get_new_frame", None)
        if getter is None:
            return self._camera.latest_frame
        frame, self._last_frame_id = getter(self._last_frame_id)
        return frame

    def _process_vision(self, frame) -> None:
        roll_rad = pitch_rad = current_yaw = 0.0
        state = self._tracker.get_state()
        if state:
            current_yaw = state["yaw_rad"]
        att = self._ctrl.get_attitude() if self._ctrl else None
        if att:
            roll_rad  = att[0]
            pitch_rad = att[1]
            current_yaw = att[2]

        altitude_m, alt_weight = self._altitude()

        all_markers = self._detector.detect(frame, debug=False)
        kept, per_color = [], {}
        for m in all_markers:
            if not (m.touches_border or _dot_plausible(m, altitude_m, roll_rad, pitch_rad)):
                continue
            if per_color.get(m.color, 0) >= config.DOT_MAX_PER_COLOR:
                continue
            per_color[m.color] = per_color.get(m.color, 0) + 1
            kept.append(m)
        if len(kept) != len(all_markers):
            self._dots_rejected += len(all_markers) - len(kept)
            log.debug("%d folt nem marker alakú vagy méretű — kihagyom.",
                      len(all_markers) - len(kept))
        all_markers = kept
        self._vision_seq += 1
        self._line_dots_t = time.time()
        if not all_markers:
            self._line_dots = []
            self._tracked_cmd_marker = None
            return

        cmd_markers = [m for m in all_markers
                       if m.marker_id in _COMMAND_MARKER_IDS and not m.touches_border]
        if self._state in (State.MARKER_FOLLOWING, State.HOVER, State.SEARCHING):
            self._announce_new_dots(cmd_markers, state, current_yaw, altitude_m)

        cy_, sy_ = math.cos(current_yaw), math.sin(current_yaw)
        line = []
        for m in cmd_markers:
            fwd, right = self._dot_offset_body(m.center_px, altitude_m)
            line.append((m.color, fwd * cy_ - right * sy_, fwd * sy_ + right * cy_))
        self._line_dots = line

        pos_markers = [m for m in all_markers if m.strategy == "color"]

        if (self._state == State.MARKER_FOLLOWING
                and time.time() >= self._marker_cooldown_until):
            fresh = [m for m in cmd_markers
                     if not self._is_spent_dot(m.center_px, state, current_yaw, altitude_m)]
            if fresh:
                best = min(fresh, key=lambda m: math.hypot(
                    float(m.center_px[0]) - config.FRAME_WIDTH / 2.0,
                    float(m.center_px[1]) - config.FRAME_HEIGHT / 2.0))
                cx, cy = (float(v) for v in best.center_px)
                self._tracked_cmd_marker = (best.marker_id, cx, cy)
            else:
                self._tracked_cmd_marker = None
        else:
            self._tracked_cmd_marker = None

        if not pos_markers:
            return

        predicted = state["pos_ned"] if (state and self._tracker.is_initialized) else None
        pose = self._estimator.estimate(pos_markers,
                                        altitude_m=altitude_m,
                                        current_yaw_rad=current_yaw,
                                        roll_rad=roll_rad,
                                        pitch_rad=pitch_rad,
                                        predicted_pos_ned=predicted,
                                        altitude_weight=alt_weight)
        if pose is None or pose.confidence < 0.2:
            return

        if self._tracker.is_initialized:
            cur = self._tracker.get_state()
            if cur is not None:
                jump = float(np.linalg.norm(pose.position_ned[:2] - cur["pos_ned"][:2]))
                max_jump = max(config.EKF_MAX_POSITION_JUMP_M,
                               3.0 * float(np.max(cur["uncertainty_pos"][:2])))
                if jump > max_jump:
                    log.warning("Túl nagy pozícióugrás (%.2f m > %.2f m) — ezt a kamerás "
                                "mérést eldobom.", jump, max_jump)
                    return

        self._tracker.update_vision(pose)
        self._last_fix_time = time.time()
        log.debug("Kamerás pozíció: É=%.2f K=%.2f Le=%.2f m, irány=%.1f°, "
                  "megbízhatóság=%.2f, markerek=%d",
                  *pose.position_ned, math.degrees(pose.yaw_rad),
                  pose.confidence, pose.n_markers)

    def _fuse_ardupilot_velocity(self) -> None:
        if not self._ctrl or not self._tracker.is_initialized:
            return
        vel = self._ctrl.get_velocity_ned()
        if vel is not None:
            self._tracker.update_velocity(np.array(vel, dtype=float))

    def _send_vision_feedback(self) -> None:
        if not self._ctrl or not self._tracker.is_initialized:
            return
        state = self._tracker.get_state()
        if state is None:
            return

        att = self._ctrl.get_attitude()
        roll  = att[0] if att else 0.0
        pitch = att[1] if att else 0.0

        self._ctrl.send_vision_position_estimate(
            state["pos_ned"], roll, pitch, state["yaw_rad"],
            pos_std=state["uncertainty_pos"], yaw_std=state.get("uncertainty_yaw"),
        )

    def _ensure_ekf_origin(self) -> None:
        if not getattr(config, "SET_EKF_ORIGIN", False) or not self._ctrl:
            return
        now = time.time()
        if self._origin_sends >= 20 or now - self._last_origin_send < 1.0:
            return
        self._ctrl.set_ekf_origin(config.EKF_ORIGIN_LAT, config.EKF_ORIGIN_LON,
                                  config.EKF_ORIGIN_ALT_M)
        if self._origin_sends == 0:
            log.info("EKF kiindulópont elküldve: szél.=%.6f hossz.=%.6f mag.=%.1f m",
                     config.EKF_ORIGIN_LAT, config.EKF_ORIGIN_LON, config.EKF_ORIGIN_ALT_M)
        self._origin_sends += 1
        self._last_origin_send = now

    def _state_init(self) -> None:
        log.info("Állapot: INDULÁS")
        self._transition(State.PREFLIGHT)

    def _state_preflight(self) -> None:
        self._ensure_ekf_origin()
        if not self._preflight_params_done:
            self._preflight_start = time.time()

            if config.CONFIGURE_ARDUPILOT:
                log.info("Felszállás előtt: az ArduPilot EKF3 paramétereinek beállítása …")
                ok, reboot_needed = self._ctrl.configure_for_gps_free()
                if not ok:
                    log.error("A paraméterek beállítása nem sikerült.")
                    self._transition(State.EMERGENCY)
                    return
                if reboot_needed:
                    log.warning(
                        "Az EKF3/GPS_TYPE paraméterek megváltoztak — "
                        "a repülésvezérlőt újra kell indítani. "
                        "Automatikus újraindítás …"
                    )
                    self._ctrl.reboot_autopilot()
                    time.sleep(2.0)
                    self._conn.disconnect()
                    time.sleep(6.0)
                    if not self._conn.connect():
                        log.error("Újraindítás után nem sikerült újracsatlakozni.")
                        self._transition(State.EMERGENCY)
                        return
                    self._ctrl = DroneController(self._conn)
                    self._request_message_rates()
                    self._preflight_start = time.time()

            self._preflight_params_done = True

            if getattr(config, "USE_KNOWN_START", False) and not self._tracker.is_initialized:
                att = self._ctrl.get_attitude()
                start = np.array(config.START_POSITION_NED, dtype=float)
                self._tracker.update_vision(PoseEstimate(
                    position_ned=start,
                    yaw_rad=att[2] if att else 0.0,
                    confidence=1.0,
                    n_markers=0,
                ))
                self._last_fix_time = time.time()
                log.info("Felszállás előtt: ismert kiindulópont É=%.1f K=%.1f Le=%.1f m", *start)

            log.info("Felszállás előtt: várakozás az első pozícióra (max. %.0f s) …",
                     config.PREFLIGHT_FIX_TIMEOUT_S)
            return

        batt = self._ctrl.get_battery_pct()
        if batt is not None and batt < config.BATTERY_WARNING_PCT:
            log.error("Felszállás előtt: az akkumulátor túl alacsony (%d%%) a biztonságos repüléshez.", batt)
            self._transition(State.EMERGENCY)
            return

        if self._tracker.is_initialized:
            state = self._tracker.get_state()
            unc = state["uncertainty_pos"]
            if (unc < config.EKF_MAX_UNCERTAINTY_M).all():
                log.info("Felszállás előtt: megvan a pozíció. Bizonytalanság: %.2f / %.2f / %.2f m",
                         *unc)
                self._transition(State.ARMING)
                return

        if time.time() - self._preflight_start > config.PREFLIGHT_FIX_TIMEOUT_S:
            log.error("Felszállás előtt: lejárt az idő, nincs pozíció. "
                      "Ellenőrizd: config.USE_KNOWN_START / START_POSITION_NED "
                      "és config.DOT_MAP.")
            self._transition(State.EMERGENCY)

    def _state_arming(self) -> None:
        self._ensure_ekf_origin()
        if self._ctrl.is_armed():
            alt = self._ctrl.get_altitude_m()
            if not math.isnan(alt) and alt > 1.0:
                log.error("A drón már a levegőben van (%.1f m) — valószínűleg az előző "
                          "futás félbeszakadt. Leszállítom; utána indítsd újra a programot.", alt)
                self._land_reason = "a drón már a levegőben volt induláskor"
                self._transition(State.LAND)
                return
            log.info("A drón már élesítve van.")
            self._guided_mode_set = False
            self._transition(State.TAKEOFF)
            return
        if not self._guided_mode_set:
            log.info("Élesítés: GUIDED üzemmód kérése.")
            if not self._ctrl.set_guided_mode():
                log.warning("Az üzemmódváltás nem sikerült, újrapróbálom …")
                return
            self._guided_mode_set = True
        now = time.time()
        if now - getattr(self, "_arm_last_try", 0.0) < 1.0:
            return
        self._arm_last_try = now
        if self._ctrl.arm():
            if getattr(self, "_arm_first_fail", None) is not None:
                log.info("Az élesítés sikerült (%.0f s várakozás után).", now - self._arm_first_fail)
            self._arm_first_fail = None
            self._guided_mode_set = False
            self._transition(State.TAKEOFF)
        else:
            if getattr(self, "_arm_first_fail", None) is None:
                self._arm_first_fail = now
                self._arm_last_log = 0.0
            if now - self._arm_last_log >= 5.0:
                self._arm_last_log = now
                log.warning("Az ArduPilot még nem engedi az élesítést (%.0f s óta) — "
                            "a pontos okot a SITL konzol (2. terminál) írja ki "
                            "(\"PreArm: …\"). Várok és újrapróbálom …",
                            now - self._arm_first_fail)

    def _state_takeoff(self) -> None:
        if not self._takeoff_sent:
            log.info("Felszállás %.1f m magasra.", config.PLANNING_ALTITUDE_M)
            if self._ctrl.takeoff(config.PLANNING_ALTITUDE_M):
                self._takeoff_sent  = True
                self._takeoff_start = time.time()
            else:
                log.warning("A felszállási parancsot elutasította, újrapróbálom …")
            return

        alt = self._ctrl.get_altitude_m()
        if not math.isnan(alt) and alt >= config.PLANNING_ALTITUDE_M * 0.9:
            log.info("Felszállás kész, magasság: %.1f m. Indul a markerkövetés.", alt)
            self._takeoff_sent = False
            state = self._tracker.get_state()
            if state:
                self._current_heading = state["yaw_rad"]
            else:
                att = self._ctrl.get_attitude()
                self._current_heading = att[2] if att else 0.0
            self._last_fix_time = time.time()
            start = np.array(config.START_POSITION_NED[:2], dtype=float)
            self._route = [start]
            self._follow_start_t = time.time()
            self._follow_start_batt = self._ctrl.get_battery_pct()
            self._transition(State.MARKER_FOLLOWING)
            return

        if time.time() - self._takeoff_start > config.TAKEOFF_TIMEOUT_S:
            log.error("A felszállás %.0f s alatt sem fejeződött be.", config.TAKEOFF_TIMEOUT_S)
            self._takeoff_sent = False
            self._transition(State.EMERGENCY)

    def _state_marker_following(self) -> None:
        if time.time() - self._last_fix_time > config.WATCHDOG_TIMEOUT_S:
            log.warning("Nincs marker a képen (3 s óta nincs pozíció) — helyben lebegek.")
            self._enter_watchdog_hold()
            return

        now = time.time()
        dt  = 1.0 / config.CONTROL_LOOP_HZ
        h   = self._current_heading

        self._record_route()
        if self._should_return_home(now):
            self._start_return_home()
            return

        e = self._cross_track_error(now)
        v_lat = 0.0
        if e is not None:
            i_max = config.LATERAL_I_MAX
            self._lat_i = float(np.clip(self._lat_i + e * dt, -i_max, i_max))
            v_lat = float(np.clip(config.LATERAL_KP * e + config.LATERAL_KI * self._lat_i,
                                  -config.LATERAL_MAX_MPS, config.LATERAL_MAX_MPS))
            self._update_drift_monitor(e, now)
        else:
            self._lat_i *= 0.995

        speed = config.MARKER_FOLLOW_SPEED_MPS * (config.DRIFT_SLOW_FACTOR if self._slowing else 1.0)
        vx = speed * math.cos(h) - v_lat * math.sin(h)
        vy = speed * math.sin(h) + v_lat * math.cos(h)

        if self._tracked_cmd_marker is not None:
            marker_id, cx_px, cy_px = self._tracked_cmd_marker
            offset_px     = cx_px - config.FRAME_WIDTH / 2.0
            offset_fwd_px = cy_px - config.FRAME_HEIGHT / 2.0
            log.debug("Követett marker: %s, eltérés oldalra %.0f px, előre-hátra %.0f px, oldalsebesség %.2f m/s",
                      _hu_color(marker_id), offset_px, offset_fwd_px, v_lat)

            if math.hypot(offset_px, offset_fwd_px) < config.MARKER_CENTER_THRESHOLD_PX:
                color = marker_id.replace("color_", "")
                if self._confirm_seq != self._vision_seq:
                    self._confirm_seq = self._vision_seq
                    if color == self._confirm_color:
                        self._confirm_count += 1
                    else:
                        self._confirm_color, self._confirm_count = color, 1
                cmd = config.MARKER_COMMANDS.get(color)
                if cmd and self._confirm_count >= config.MARKER_CONFIRM_FRAMES:
                    trk = self._tracker.get_state()
                    att = self._ctrl.get_attitude() if self._ctrl else None
                    yaw = att[2] if att else (trk["yaw_rad"] if trk else 0.0)
                    alt, _ = self._altitude()
                    dot = self._dot_ground_ned((cx_px, cy_px), trk, yaw, alt)
                    if dot is not None:
                        self._spent_dots.append(dot)
                        self._announced_dots.append((color, dot))
                        self._record_route(point=dot)
                    self._confirm_color, self._confirm_count = None, 0
                    self._execute_marker_command(cmd)
                    return
            else:
                self._confirm_color, self._confirm_count = None, 0

        yaw_rate = 0.0
        state = self._tracker.get_state()
        if state:
            yaw_rate = float(np.clip(
                2.0 * _angle_diff(self._current_heading, state["yaw_rad"]),
                -1.0, 1.0,
            ))

        self._ctrl.send_velocity_ned(vx, vy, 0.0, yaw_rate=yaw_rate)

    def _cross_track_error(self, now: float) -> Optional[float]:
        if now - self._line_dots_t > 0.4 or not self._line_dots:
            return None
        h = self._current_heading
        ch, sh = math.cos(h), math.sin(h)
        best, best_score = None, None
        for _, dn, de in self._line_dots:
            along = dn * ch + de * sh
            cross = -dn * sh + de * ch
            if along < -1.0 or abs(cross) > config.CENTER_GATE_M:
                continue
            score = abs(cross) + 0.3 * abs(along)
            if best_score is None or score < best_score:
                best, best_score = cross, score
        return best

    def _update_drift_monitor(self, e: float, now: float) -> None:
        self._xtrack_ema += 0.2 * (e - self._xtrack_ema)
        a = abs(self._xtrack_ema)
        side = "balra" if self._xtrack_ema > 0 else "jobbra"

        if a > config.DRIFT_WARN_M:
            if self._drift_since is None:
                self._drift_since = now
            if now - self._drift_since > 1.0 and now - self._last_drift_warn > 5.0:
                self._last_drift_warn = now
                log.warning("Sodródás: %s tolódtam a markerek vonalától — visszakorrigálok.", side)
        else:
            self._drift_since = None

        if not self._slowing and a > config.DRIFT_SLOW_M:
            self._slowing = True
            log.warning("Erősen elsodródtam (%s) — lassítok, amíg visszaérek a vonalra.", side)
        elif self._slowing and a < config.DRIFT_RECOVER_M:
            self._slowing = False
            log.info("Visszaértem a vonalra — normál sebességgel megyek tovább.")

        wind_comp = config.LATERAL_KI * self._lat_i
        if abs(wind_comp) > config.WIND_WARN_MPS and now - self._last_wind_warn > 15.0:
            self._last_wind_warn = now
            log.warning("Erős oldalszél %s — folyamatosan ellen kell tartanom.",
                        "balról" if wind_comp < 0 else "jobbról")

    def _altitude(self):
        if not self._ctrl:
            return None, 1.0
        rng = self._ctrl.get_rangefinder_m() if getattr(config, "RANGEFINDER_IS_ALTIMETER", False) else None
        if rng is not None and config.RANGEFINDER_MIN_M <= rng <= config.RANGEFINDER_MAX_M:
            att = self._ctrl.get_attitude()
            if att:
                rng *= math.cos(att[0]) * math.cos(att[1])
            if self._alt_source != "laser":
                self._alt_source = "laser"
                log.info("Lézeres magasságmérés aktív.")
            return float(rng), config.RANGEFINDER_ALT_WEIGHT
        alt = self._ctrl.get_altitude_m()
        if math.isnan(alt):
            return None, 1.0
        if self._alt_source != "baro" and self._state not in _GROUND_STATES:
            self._alt_source = "baro"
            log.warning("Nincs lézeres magasságmérés — a barométert használom (pontatlanabb).")
        return float(alt), 1.0

    def _request_message_rates(self) -> None:
        req = getattr(self._ctrl, "request_message_interval", None)
        if req is None:
            return
        for msg_id, hz in ((30, 25.0),
                           (32, 10.0),
                           (132, 20.0)):
            req(msg_id, hz)

    def _dot_offset_body(self, center_px, altitude_m: Optional[float]):
        h = altitude_m if altitude_m else config.PLANNING_ALTITUDE_M
        fx = float(config.CAMERA_MATRIX[0, 0]); cx0 = float(config.CAMERA_MATRIX[0, 2])
        fy = float(config.CAMERA_MATRIX[1, 1]); cy0 = float(config.CAMERA_MATRIX[1, 2])
        nx = (float(center_px[0]) - cx0) / fx
        ny = (float(center_px[1]) - cy0) / fy
        ray = np.array([-ny, nx, 1.0])
        att = self._ctrl.get_attitude() if self._ctrl else None
        if att:
            cr, sr = math.cos(att[0]), math.sin(att[0])
            cp, sp = math.cos(att[1]), math.sin(att[1])
            r_level = np.array([[cp, sp * sr, sp * cr],
                                [0.0, cr, -sr],
                                [-sp, cp * sr, cp * cr]])
            ray = r_level @ ray
        if ray[2] < 0.2:
            ray = np.array([ray[0], ray[1], 0.2])
        t = h / ray[2]
        return float(t * ray[0]), float(t * ray[1])

    def _dot_ground_ned(self, center_px, state, yaw_rad: float,
                        altitude_m: Optional[float]) -> Optional[np.ndarray]:
        if state is None or center_px is None:
            return None
        fwd, right = self._dot_offset_body(center_px, altitude_m)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        n = state["pos_ned"][0] + fwd * c - right * s
        e = state["pos_ned"][1] + fwd * s + right * c
        return np.array([n, e])

    def _is_spent_dot(self, center_px, state, yaw_rad: float,
                      altitude_m: Optional[float]) -> bool:
        if not self._spent_dots:
            return False
        p = self._dot_ground_ned(center_px, state, yaw_rad, altitude_m)
        if p is None:
            return False
        r = getattr(config, "MARKER_REARM_DIST_M", 2.5)
        return any(float(np.linalg.norm(p - q)) < r for q in self._spent_dots)

    def _announce_new_dots(self, markers, state, yaw_rad: float,
                           altitude_m: Optional[float]) -> None:
        for m in markers:
            p = self._dot_ground_ned(m.center_px, state, yaw_rad, altitude_m)
            if p is None:
                continue
            if any(c == m.color and float(np.linalg.norm(p - q)) < 3.0
                   for c, q in self._announced_dots):
                continue
            self._announced_dots.append((m.color, p))
            fwd, right = self._dot_offset_body(m.center_px, altitude_m)
            cmd = config.MARKER_COMMANDS.get(m.color, "")
            where = []
            if abs(fwd) >= 0.5:
                where.append("előttem" if fwd > 0 else "mögöttem")
            if abs(right) >= 0.5:
                where.append("jobbra" if right > 0 else "balra")
            log.info("Látok egy %s markert (%s) — %s",
                     _hu_color(m.marker_id), _CMD_HU.get(cmd, "nincs parancs"),
                     " ".join(where) if where else "pont alattam")

    def _note_tick_time(self, elapsed: float, dt: float) -> None:
        self._tick_count += 1
        slow = elapsed > config.LOOP_SLOW_FACTOR * dt
        self._recent_slow.append(slow)
        self._slowest_tick_s = max(self._slowest_tick_s, elapsed)
        if not slow:
            return
        self._slow_ticks += 1
        log.debug("Lassú kör: %.1f ms (cél %.1f ms)", elapsed * 1000, dt * 1000)
        now = time.time()
        n = len(self._recent_slow)
        if (n >= self._recent_slow.maxlen and sum(self._recent_slow) > 0.25 * n
                and now - self._last_slow_warning > 30.0):
            self._last_slow_warning = now
            log.warning("A vezérlőciklus tartósan lassú: az utolsó 10 s-ban a körök %.0f%%-a "
                        "tartott %.0f ms-nál tovább (túlterhelt a gép?).",
                        100.0 * sum(self._recent_slow) / n,
                        config.LOOP_SLOW_FACTOR * dt * 1000)

    def _log_stats(self) -> None:
        if getattr(self, "_stats_logged", False):
            return
        self._stats_logged = True
        if self._tick_count:
            log.debug("Vezérlőciklus: %d kör, ebből %d lassú (%.1f%%), a leglassabb %.0f ms.",
                     self._tick_count, self._slow_ticks,
                     100.0 * self._slow_ticks / self._tick_count, self._slowest_tick_s * 1000)
        log.debug("Kamera: %d új képet dolgoztam fel, %d körben nem jött új kép.",
                 self._frames_processed, self._ticks_without_frame)
        if self._dots_rejected:
            log.info("Kiszűrt hamis foltok (alak vagy méret alapján): %d.", self._dots_rejected)

    def _execute_marker_command(self, cmd: str) -> None:
        self._tracked_cmd_marker = None

        if cmd == "land":
            log.info(">>> PIROS marker a drón alatt → LESZÁLLÁS")
            self._land_reason = "elértem a piros markert"
            self._transition(State.LAND)
            return

        if cmd == "turn_left":
            self._current_heading = (
                (self._current_heading - math.pi / 2) + math.pi
            ) % (2 * math.pi) - math.pi
            log.info(">>> SÁRGA marker a drón alatt → BALRA fordulok")
            self._reset_lateral()
        elif cmd == "turn_right":
            self._current_heading = (
                (self._current_heading + math.pi / 2) + math.pi
            ) % (2 * math.pi) - math.pi
            log.info(">>> FEKETE marker a drón alatt → JOBBRA fordulok")
            self._reset_lateral()
        elif cmd == "straight":
            log.info(">>> KÉK marker a drón alatt → EGYENESEN tovább előre")
            return

        self._marker_cooldown_until = time.time() + config.MARKER_COMMAND_COOLDOWN_S


    def _record_route(self, point=None) -> None:
        if point is None:
            st = self._tracker.get_state()
            if st is None or not self._route:
                return
            p = np.asarray(st["pos_ned"][:2], dtype=float)
            if np.linalg.norm(p - self._route[-1]) < config.RTH_BREADCRUMB_M:
                return
        else:
            p = np.asarray(point[:2], dtype=float)
            if self._route and np.linalg.norm(p - self._route[-1]) < 0.8:
                self._route[-1] = p
                return
        self._route.append(p)

    def _route_length_home(self, pos) -> float:
        if not self._route:
            return 0.0
        pts = [np.asarray(pos[:2], dtype=float)] + self._route[::-1]
        return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))

    def _should_return_home(self, now: float) -> bool:
        if not getattr(config, "RTH_ENABLED", True) or self._returning:
            return False
        t_test = getattr(config, "RTH_TEST_AFTER_S", None)
        if t_test is not None and self._follow_start_t is not None and now - self._follow_start_t >= t_test:
            log.warning("Teszt: %.0f s markerkövetés után visszatérés indul.", t_test)
            return True
        batt = self._ctrl.get_battery_pct()
        if batt is None:
            if now - getattr(self, "_last_batt_log", 0.0) > 30.0:
                self._last_batt_log = now
                log.warning("Nincs akkumulátoradat az ArduPilottól — a visszatérést nem tudom kiszámolni.")
            return False
        if now - getattr(self, "_last_batt_log", 0.0) > 30.0:
            self._last_batt_log = now
            need_ = self._battery_needed_home()
            if need_ is None:
                log.info("Akkumulátor: %d%%.", batt)
            else:
                log.info("Akkumulátor: %d%% (a hazaútra most kb. %.0f%% kellene).", batt, need_)
        if batt <= config.RTH_BATTERY_PCT:
            log.warning("Alacsony akkumulátor (%d%%) — visszafordulok a felszállási ponthoz.", batt)
            return True
        need = self._battery_needed_home()
        if need is not None and batt <= need + config.BATTERY_FAILSAFE_PCT + config.RTH_RESERVE_PCT:
            log.warning("Az akkumulátor (%d%%) már csak a hazaútra elég (kb. %.0f%% kell hozzá) — "
                        "visszafordulok a felszállási ponthoz.", batt, need)
            return True
        return False

    def _battery_needed_home(self):
        if self._follow_start_t is None or self._follow_start_batt is None:
            return None
        batt = self._ctrl.get_battery_pct()
        st = self._tracker.get_state()
        elapsed = time.time() - self._follow_start_t
        used = self._follow_start_batt - (batt if batt is not None else self._follow_start_batt)
        if st is None or elapsed < 10.0 or used < 1:
            return None
        rate = used / elapsed
        travelled = self._route_length_home(st["pos_ned"])
        speed = min(config.RTH_SPEED_MPS, max(0.3, travelled / elapsed))
        t_home = travelled / speed * getattr(config, "RTH_TIME_MARGIN", 1.0)
        t_land = (config.PLANNING_ALTITUDE_M / 0.8 + 5.0) * (config.RTH_SPEED_MPS / speed)
        return rate * (t_home + t_land)

    def _start_return_home(self) -> None:
        st = self._tracker.get_state()
        pos = np.asarray(st["pos_ned"][:2] if st is not None else self._route[-1], dtype=float)
        path = _simplify_path([pos] + [np.asarray(p, dtype=float) for p in self._route[::-1]],
                              config.RTH_PATH_SIMPLIFY_M)
        while len(path) > 2 and np.linalg.norm(path[1] - path[0]) < config.RTH_WAYPOINT_RADIUS_M:
            path.pop(1)
        self._rth_path = path
        self._rth_seg = 0
        self._rth_turning = True
        self._returning = True
        self._rth_start_batt = self._ctrl.get_battery_pct()
        self._tracked_cmd_marker = None
        self._reset_lateral()
        log.info("Visszatérés: a bejárt úton megyek vissza a felszállási ponthoz (%d egyenes szakasz).",
                 len(path) - 1)
        self._notify(_SEV_NOTICE, "Visszateres a felszallasi ponthoz")
        self._transition(State.RETURN_HOME)

    def _state_return_home(self) -> None:
        if time.time() - self._last_fix_time > config.WATCHDOG_TIMEOUT_S:
            log.warning("Nincs marker a képen (3 s óta nincs pozíció) — helyben lebegek.")
            self._enter_watchdog_hold()
            return
        st = self._tracker.get_state()
        path = getattr(self, "_rth_path", None)
        if st is None or not path or len(path) < 2:
            self._land_reason = "visszaértem a felszállási ponthoz"
            self._transition(State.LAND)
            return

        pos = np.asarray(st["pos_ned"][:2], dtype=float)

        while True:
            a, b = path[self._rth_seg], path[self._rth_seg + 1]
            seg = b - a
            seg_len = float(np.linalg.norm(seg))
            last = self._rth_seg + 2 >= len(path)
            if seg_len < 1e-3:
                along, u = 0.0, np.array([math.cos(self._current_heading), math.sin(self._current_heading)])
            else:
                u = seg / seg_len
                along = float(np.dot(pos - a, u))
            to_end = seg_len - along
            done = (float(np.linalg.norm(b - pos)) < config.RTH_HOME_RADIUS_M or to_end < 0.15) if last \
                else to_end < config.RTH_SEGMENT_END_M
            if not done:
                break
            if last:
                log.info("Visszaértem a felszállási ponthoz — leszállok.")
                self._land_reason = "visszaértem a felszállási ponthoz"
                self._ctrl.send_velocity_ned(0.0, 0.0, 0.0)
                self._transition(State.LAND)
                return
            self._rth_seg += 1
            nb = path[self._rth_seg + 1] - path[self._rth_seg]
            if np.linalg.norm(nb) > 1e-3 and abs(_angle_diff(math.atan2(nb[1], nb[0]),
                                                              math.atan2(u[1], u[0]))) > math.radians(45):
                self._reset_lateral()
                self._rth_turning = True
                log.info("Hazaút: kanyar — %s fordulok.",
                         "JOBBRA" if _angle_diff(math.atan2(nb[1], nb[0]), math.atan2(u[1], u[0])) > 0
                         else "BALRA")

        h = math.atan2(u[1], u[0])
        self._current_heading = h
        yaw_err = _angle_diff(h, st["yaw_rad"])
        yaw_rate = float(np.clip(2.0 * yaw_err, -1.0, 1.0))
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([-math.sin(h), math.cos(h)])

        if self._rth_turning:
            if abs(yaw_err) < math.radians(10):
                self._rth_turning = False
            else:
                corr = -float(np.dot(pos - a, right))
                v = right * float(np.clip(0.5 * corr, -0.5, 0.5))
                self._ctrl.send_velocity_ned(float(v[0]), float(v[1]), 0.0, yaw_rate=yaw_rate)
                return

        now = time.time()
        dt = 1.0 / config.CONTROL_LOOP_HZ

        e = self._cross_track_error(now)
        if e is None:
            e_geo = -float(np.dot(pos - a, right))
            e = float(np.clip(e_geo, -config.CENTER_GATE_M, config.CENTER_GATE_M))
            self._lat_i *= 0.995
            v_lat = float(np.clip(config.LATERAL_KP * e + config.LATERAL_KI * self._lat_i,
                                  -config.LATERAL_MAX_MPS, config.LATERAL_MAX_MPS))
        else:
            i_max = config.LATERAL_I_MAX
            self._lat_i = float(np.clip(self._lat_i + e * dt, -i_max, i_max))
            v_lat = float(np.clip(config.LATERAL_KP * e + config.LATERAL_KI * self._lat_i,
                                  -config.LATERAL_MAX_MPS, config.LATERAL_MAX_MPS))
            self._update_drift_monitor(e, now)

        speed = config.RTH_SPEED_MPS * (config.DRIFT_SLOW_FACTOR if self._slowing else 1.0)
        speed *= max(0.0, math.cos(yaw_err))
        speed = min(speed, 0.6 * max(to_end, 0.0) + 0.3)
        v = fwd * speed + right * v_lat
        self._ctrl.send_velocity_ned(float(v[0]), float(v[1]), 0.0, yaw_rate=yaw_rate)

    def _reset_lateral(self) -> None:
        self._lat_i = 0.0
        self._xtrack_ema = 0.0
        self._drift_since = None
        self._slowing = False

    def _state_searching(self) -> None:
        if time.time() - self._last_fix_time < config.WATCHDOG_TIMEOUT_S:
            if self._returning:
                log.info("Újra látok markert — folytatom a visszatérést.")
                self._transition(State.RETURN_HOME)
            else:
                log.info("Újra látok markert — folytatom a markerkövetést.")
                self._transition(State.MARKER_FOLLOWING)
            return

        self._ctrl.send_velocity_ned(0.0, 0.0, 0.0,
                                     yaw_rate=config.SEARCH_YAW_RATE_RADS)

        elapsed   = time.time() - self._search_start_time
        full_turn = (2 * math.pi) / config.SEARCH_YAW_RATE_RADS
        if elapsed >= full_turn:
            log.warning("Körbefordultam (360°), de nem találtam markert. Leszállok.")
            self._land_reason = "elveszett a marker, a 360°-os keresés sem talált"
            self._notify(_SEV_WARNING, "ELVESZETT A MARKER - LESZALLOK")
            self._transition(State.LAND)

    def _state_hover(self) -> None:
        if time.time() - self._last_fix_time < config.WATCHDOG_TIMEOUT_S:
            self._pid.reset()
            self._last_fix_time = time.time()
            if self._returning:
                log.info("Újra van pozíció — folytatom a visszatérést.")
                self._transition(State.RETURN_HOME)
            else:
                log.info("Újra van pozíció — folytatom a markerkövetést.")
                self._transition(State.MARKER_FOLLOWING)
            return

        state = self._tracker.get_state()
        if state is None or self._hold_pos is None:
            self._ctrl.send_velocity_ned(0.0, 0.0, 0.0)
        else:
            vel_ned, yaw_rate = self._pid.compute_station_keep(
                self._hold_pos, state["pos_ned"],
                self._hold_yaw, state["yaw_rad"],
            )
            off = float(np.linalg.norm(state["pos_ned"][:2] - self._hold_pos[:2]))
            if off > config.HOVER_DRIFT_WARN_M and time.time() - self._last_hover_drift_warn > 5.0:
                self._last_hover_drift_warn = time.time()
                log.warning("Lebegés közben elsodródtam — visszatérek a helyemre.")
            self._ctrl.send_velocity_ned(vel_ned[0], vel_ned[1], 0.0,
                                         yaw_rate=yaw_rate)

        if time.time() - self._hover_start > config.HOVER_TIMEOUT_S:
            log.warning("%.0f s lebegés után sincs marker — 360°-os keresés indul.",
                        config.HOVER_TIMEOUT_S)
            self._search_start_time = time.time()
            self._transition(State.SEARCHING)

    def _state_land(self) -> None:
        if not self._land_sent:
            reason = self._land_reason if self._land_reason else "a küldetés befejeződött"
            log.info("Leszállás — oka: %s", reason)
            self._notify(_SEV_NOTICE, f"Leszallas: {reason}")
            self._land_sent  = True
            self._land_start = time.time()
            self._land_cmd_ok = self._command_land()
            return

        if not self._land_cmd_ok and time.time() - self._land_last_try > 1.0:
            log.warning("A repülésvezérlő még nem fogadta el a leszállást — újrapróbálom.")
            self._land_cmd_ok = self._command_land()

        alt = self._ctrl.get_altitude_m()
        if not math.isnan(alt) and alt < config.LAND_COMPLETE_ALT_M:
            log.info("Leszállás kész, magasság: %.1f m.", alt)
            self._land_sent = False
            self._transition(State.DONE)
            return

        if not self._ctrl.is_armed():
            log.info("Leszállás kész — a motorok leálltak.")
            self._land_sent = False
            self._transition(State.DONE)
            return

        if time.time() - self._land_start > config.LAND_TIMEOUT_S:
            log.error("A leszállás %.0f s alatt sem fejeződött be — a program leáll, "
                      "a leszállást a repülésvezérlő fejezi be (LAND üzemmód).", config.LAND_TIMEOUT_S)
            set_mode = getattr(self._ctrl, "set_mode", None)
            if set_mode is not None:
                set_mode(config.LAND_MODE_ID)
            self._land_sent = False
            self._transition(State.DONE)

    def _state_done(self) -> None:
        log.info("A küldetés befejeződött. Leállás.")
        self._log_stats()
        self._running = False

    def _state_emergency(self) -> None:
        now = time.time()
        if not self._emerg_started:
            self._emerg_started = True
            self._emerg_start = now
            log.error("VÉSZHELYZET: azonnali leszállás helyben.")
            self._notify(_SEV_CRITICAL, "VESZHELYZET - LESZALLOK")
            if not self._ctrl or not self._ctrl.is_armed():
                log.info("A drón nincs élesítve (a földön van) — leállok.")
                self._running = False
                return
            self._land_cmd_ok = self._command_land()
            return

        if not self._ctrl.is_armed():
            log.info("Vészleszállás kész — a motorok leálltak.")
            self._running = False
            return

        if not self._land_cmd_ok and now - self._land_last_try > 1.0:
            log.warning("A repülésvezérlő még nem fogadta el a vészleszállást — újrapróbálom.")
            self._land_cmd_ok = self._command_land()

        if now - self._emerg_start > config.LAND_WAIT_TIMEOUT_S:
            log.error("A drón %.0f s után sem ért földet — a program leáll, "
                      "a leszállást a repülésvezérlő fejezi be.", config.LAND_WAIT_TIMEOUT_S)
            self._running = False

    def _command_land(self) -> bool:
        self._land_last_try = time.time()
        try:
            cmd = getattr(self._ctrl, "command_land", None)
            ok = cmd() if cmd is not None else self._ctrl.land()
        except Exception as exc:
            log.error("A leszállási parancs küldése nem sikerült: %s", exc)
            return False
        if not ok:
            log.error("A repülésvezérlő a leszállási parancsot sem nyugtázta, "
                      "sem LAND üzemmódra nem váltott.")
        return bool(ok)

    def _send_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_hb_sent < 1.0:
            return
        self._last_hb_sent = now
        send = getattr(self._conn, "send_heartbeat", None)
        if send is not None:
            send()

    def _handle_stop_request(self) -> None:
        if self._state in (State.LAND, State.EMERGENCY, State.DONE):
            return
        if self._state == State.PAUSED:
            log.warning("Leállítás kézi irányítás közben — nem avatkozom be, a drón a pilótánál marad.")
            self._running = False
            return
        armed = bool(self._ctrl and self._ctrl.is_armed())
        if self._state in _GROUND_STATES and not armed:
            self._running = False
            return
        log.warning("Leállítás kérve, a drón élesítve van — leszállok. "
                    "(Azonnali kilépés: Ctrl+C még egyszer.)")
        self._land_reason = "a kezelő leállította a programot"
        self._transition(State.LAND)

    def _state_paused(self) -> None:
        hb = self._conn.get_message("HEARTBEAT")
        if hb is not None and hb.custom_mode == config.GUIDED_MODE_ID:
            log.info("Újra GUIDED üzemmód — folytatom az önálló repülést.")
            self._notify(_SEV_NOTICE, "GUIDED vissza - onallo repules folytatodik")
            self._transition(State.RETURN_HOME if self._returning else State.MARKER_FOLLOWING)

    def _safety_ok(self) -> bool:
        if not self._conn.is_healthy():
            log.error("Megszakadt a MAVLink-kapcsolat!")
            if self._state != State.EMERGENCY:
                self._transition(State.EMERGENCY)
                return False
            return True

        if not self._ctrl:
            return True

        if self._state == State.EMERGENCY:
            return True

        batt = self._ctrl.get_battery_pct()
        _airborne_states = {State.TAKEOFF, State.MARKER_FOLLOWING, State.HOVER,
                            State.SEARCHING, State.RETURN_HOME}
        if batt is not None and batt < config.BATTERY_FAILSAFE_PCT:
            if self._state in _airborne_states:
                log.error("Kritikus akkumulátorszint: %d%% — leszállás.", batt)
                self._land_reason = f"kritikus akkumulátorszint ({batt}%)"
                self._notify(_SEV_CRITICAL, f"AKKU KRITIKUS {batt}% - LESZALLOK")
                self._transition(State.LAND)
                return False
        if batt is not None and batt < config.BATTERY_WARNING_PCT:
            if not self._battery_warning_sent:
                self._notify(_SEV_WARNING, f"Akku alacsony: {batt}%")
                self._battery_warning_sent = True

        if self._tracker.is_initialized:
            ekf = self._tracker.get_state()
            if ekf is not None:
                pos = ekf["pos_ned"]
                n_min, n_max = config.GEOFENCE_NORTH
                e_min, e_max = config.GEOFENCE_EAST
                if not (n_min <= pos[0] <= n_max and e_min <= pos[1] <= e_max):
                    log.error("Kirepültem a megengedett területről! É=%.1f K=%.1f m — "
                              "határok: É [%.0f, %.0f], K [%.0f, %.0f]",
                              pos[0], pos[1], n_min, n_max, e_min, e_max)
                    self._transition(State.EMERGENCY)
                    return False

        if self._state in _GROUND_STATES:
            return True

        alt, _ = self._altitude()
        too_low  = alt is not None and alt < config.MIN_ALTITUDE_M
        too_high = alt is not None and alt > config.MAX_ALTITUDE_M
        if too_low or too_high:
            now = time.time()
            if self._alt_violation_since is None:
                self._alt_violation_since = now
            if now - self._alt_violation_since >= getattr(config, "ALT_LIMIT_PERSIST_S", 0.0):
                if too_low:
                    log.error("Túl alacsonyan vagyok: %.1f m < %.1f m!", alt,
                              config.MIN_ALTITUDE_M)
                else:
                    log.error("Túl magasan vagyok: %.1f m > %.1f m — VÉSZHELYZET.",
                              alt, config.MAX_ALTITUDE_M)
                self._transition(State.EMERGENCY)
                return False
        else:
            self._alt_violation_since = None

        att = self._ctrl.get_attitude()
        if att is not None:
            max_tilt = math.radians(config.MAX_TILT_DEG)
            if abs(att[0]) > max_tilt or abs(att[1]) > max_tilt:
                log.error("Túl nagy dőlés! oldalra %.1f°, előre %.1f° — VÉSZHELYZET.",
                          math.degrees(att[0]), math.degrees(att[1]))
                self._transition(State.EMERGENCY)
                return False

        if not self._camera.is_healthy:
            if not self._camera_warning_sent:
                self._notify(_SEV_WARNING, "Kamerahiba - a navigacio korlatozott")
                self._camera_warning_sent = True
        else:
            self._camera_warning_sent = False

        rng = None if getattr(config, "RANGEFINDER_IS_ALTIMETER", False) else self._ctrl.get_rangefinder_m()
        if rng is not None and rng < config.OBSTACLE_CLEARANCE_M:
            log.error("Akadály %.2f m-re — közelebb, mint %.2f m! VÉSZHELYZET.",
                      rng, config.OBSTACLE_CLEARANCE_M)
            self._notify(_SEV_CRITICAL, f"AKADALY {rng:.1f} m - VESZLEALLAS")
            self._transition(State.EMERGENCY)
            return False

        _flight_states = {State.MARKER_FOLLOWING, State.SEARCHING, State.HOVER, State.RETURN_HOME}
        if self._state in _flight_states:
            hb = self._conn.get_message("HEARTBEAT")
            if hb is not None and hb.custom_mode != config.GUIDED_MODE_ID:
                log.warning("A repülésvezérlő kilépett a GUIDED üzemmódból (mód=%d) — "
                            "az önálló irányítás szünetel.", hb.custom_mode)
                self._notify(_SEV_WARNING, "Kezi iranyitas - onallo mod szunetel")
                self._transition(State.PAUSED)
                return False

        return True

    def _notify(self, severity: int, message: str) -> None:
        print(f"\a\n{'!' * 50}\n  {message}\n{'!' * 50}\n")
        log.warning("ÉRTESÍTÉS: %s", message)
        if self._ctrl:
            self._ctrl.send_statustext(severity, _ascii(message))

    def _transition(self, new_state: State) -> None:
        if new_state != self._state:
            log.info("Állapot: %s → %s", _STATE_HU.get(self._state, self._state.name),
                     _STATE_HU.get(new_state, new_state.name))
            self._state = new_state

    def _enter_watchdog_hold(self) -> None:
        state = self._tracker.get_state()
        if state:
            self._hold_pos = state["pos_ned"].copy()
            self._hold_yaw = state["yaw_rad"]
        self._pid.reset()
        self._hover_start = time.time()
        self._transition(State.HOVER)

    def _signal_handler(self, sig, _) -> None:
        if self._stop_requested:
            log.warning("Ismételt leállítás — azonnali kilépés.")
            raise SystemExit(1)
        log.warning("Leállítás (Ctrl+C) — biztonságos leállás.")
        self._stop_requested = True

if __name__ == "__main__":
    nav = NavigationSystem()
    nav.start()
