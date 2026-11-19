
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

# Parancs-pöttyök: a config.MARKER_COMMANDS színei (kék = egyenesen, fekete = jobbra, sárga = balra, piros = leszállás)
_COMMAND_MARKER_IDS = {f"color_{c}" for c in config.MARKER_COMMANDS}

_SEV_CRITICAL = 2
_SEV_WARNING  = 4
_SEV_NOTICE   = 5
_SEV_INFO     = 6

# ---------------------------------------------------------------- magyar kiírás
_STATE_HU = {
    State.INIT:             "INDULÁS",
    State.PREFLIGHT:        "FELSZÁLLÁS ELŐTTI ELLENŐRZÉS",
    State.ARMING:           "ÉLESÍTÉS",
    State.TAKEOFF:          "FELSZÁLLÁS",
    State.MARKER_FOLLOWING: "PÖTTYKÖVETÉS",
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
    """Douglas–Peucker vonalegyszerűsítés: a tol-nál közelebbi köztes pontokat elhagyja
    (a kis hullámzást kisimítja), a kanyarokat megtartja. Az első és az utolsó pont marad."""
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
    """Valódi pötty-e a felismert folt: elég kör alakú-e, és akkora-e, amekkorának
    egy DOT_DIAMETER_M átmérőjű pöttynek a mért magasságból látszania kell.

    A várt átmérő a lyukkamera-modellből: d = fx * D / Z, ahol Z a pötty távolsága
    a kamera optikai tengelye mentén. Z = h / r_le, ahol r_le a pötty felé mutató
    látósugár (kamera-z = 1 normálással) lefelé mutató összetevője a drón dőlésével
    együtt, h a kamera magassága. Ugyanezt az összefüggést használja a pózbecslő is."""
    if m.confidence < getattr(config, "DOT_MIN_SHAPE_CONF", 0.0):
        return False
    if not altitude_m or altitude_m <= 0.3 or m.center_px is None or m.diameter_px <= 0:
        return True                      # nincs megbízható magasság: méret szerint nem szűr
    K = config.CAMERA_MATRIX
    nx = (float(m.center_px[0]) - K[0, 2]) / K[0, 0]
    ny = (float(m.center_px[1]) - K[1, 2]) / K[1, 1]
    # látósugár a drón testkoordinátáiban (előre, jobbra, le), majd vízszintesre forgatva
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ray_down = -sp * (-ny) + cp * sr * nx + cp * cr * 1.0
    if ray_down < 0.2:
        return True                      # nagyon ferde rálátás: a méret nem mérvadó
    h_cam = altitude_m - float(config.T_BODY_TO_CAM[2])   # a kamera kicsit a drón középpontja fölött van
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
        # Az utolsó néhány végrehajtott parancs-pötty becsült helye (NED). Ezeket a pöttyöket
        # nem hajtja végre újra (különben a lassú fordulás után a 4 s-os cooldown lejártakor
        # a drón még a fekete/sárga pötty fölött van, és még egyszer fordul -> visszafelé repül).
        self._spent_dots: deque = deque(maxlen=int(getattr(config, "MARKER_SPENT_MEMORY", 3)))
        self._search_start_time:     float = 0.0

        self._battery_warning_sent:  bool  = False
        self._camera_warning_sent:   bool  = False
        self._land_reason:           str   = ""

        # Csak az új képkockákat dolgozza fel (a ciklus 20 Hz, a kamera ~30 fps,
        # de terhelés alatt a kamera lassabb lehet: ugyanazt a képet nem elemzi kétszer).
        self._last_frame_id:         int   = -1
        # Oldalirányú középre tartás és sodródásfigyelés
        self._line_dots: list = []           # [(szín, dn, de)] a legutóbbi képen látott pöttyök a drónhoz képest (NED, m)
        self._line_dots_t:           float = 0.0
        self._lat_i:                 float = 0.0   # a PI szabályzó integrálja (állandó oldalszél ellensúlyozása)
        self._xtrack_ema:            float = 0.0
        self._drift_since                  = None
        self._last_drift_warn:       float = 0.0
        self._last_wind_warn:        float = 0.0
        self._last_hover_drift_warn: float = 0.0
        self._slowing:               bool  = False
        # Parancs csak több egymást követő képen történő megerősítés után
        self._vision_seq:            int   = 0
        self._confirm_color                = None
        self._confirm_count:         int   = 0
        self._confirm_seq:           int   = -1
        # Magasságforrás (lézeres távolságmérő vagy barométer)
        self._alt_source                   = None
        # Visszatérés a felszállási ponthoz (alacsony akkumulátornál)
        self._route: list = []               # a bejárt út pontjai (NED n, e), a felszállási ponttól kezdve
        self._returning:             bool  = False   # visszatérés módban vagyunk (lebegés/keresés után is ide tér vissza)
        self._rth_path: list = []           # a hazaút egyenes szakaszainak végpontjai
        self._rth_seg:  int  = 0
        self._rth_turning: bool = False
        self._rth_start_batt               = None
        self._follow_start_t               = None
        self._follow_start_batt            = None
        # A már bemondott (naplózott) pöttyök helye, hogy egy pöttyöt csak egyszer írjon ki.
        self._announced_dots: deque = deque(maxlen=20)
        # Vezérlőciklus időzítése: lassú köröket nem írja ki egyenként, csak összesít.
        self._tick_count:            int   = 0
        self._slow_ticks:            int   = 0
        self._slowest_tick_s:        float = 0.0
        self._recent_slow: deque = deque(maxlen=int(10 * config.CONTROL_LOOP_HZ))   # utolsó ~10 s
        self._last_slow_warning:     float = 0.0
        self._frames_processed:      int   = 0
        self._ticks_without_frame:   int   = 0

        self._origin_sends:          int   = 0
        self._last_origin_send:      float = 0.0

        # Leállítás (Ctrl+C / SIGTERM): a jelkezelő csak jelzést állít be, a leszállást
        # a vezérlőciklus végzi, így közben is megy a kamerás pozíció a repülésvezérlőnek.
        self._stop_requested:        bool  = False
        # Leszállás / vészhelyzet: parancs nyugtázva-e, mikor próbáltuk utoljára
        self._land_cmd_ok:           bool  = False
        self._land_last_try:         float = 0.0
        self._emerg_started:         bool  = False
        self._emerg_start:           float = 0.0
        # Saját életjel a repülésvezérlőnek (1 Hz)
        self._last_hb_sent:          float = 0.0
        # Kiszűrt (nem pötty alakú / nem pötty méretű) foltok száma
        self._dots_rejected:         int   = 0
        # Mióta van a drón a megengedett magasságsávon kívül (None: nincs kívül)
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
            # A küldetés végén (KÉSZ / VÉSZHELYZET) is rendben le kell állítani a
            # kamera és a MAVLink háttérszálát. Ha futva maradnak, a Python kilépéskor
            # erőszakosan leállítja őket az OpenCV kódja közben, és ebből lesz a
            # "FATAL: exception not rethrown / Aborted (core dumped)" a legvégén.
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
        """Új képkocka, vagy None, ha az előző óta nem jött új."""
        getter = getattr(self._camera, "get_new_frame", None)
        if getter is None:                       # régi / teszt kamera: nincs azonosító
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
            # A pötty nem ad irányt: a yaw az ArduPilot attitűdjéből (kompasz) jön,
            # különben a tracker yaw-ja sosem korrigálódna.
            current_yaw = att[2]

        altitude_m, alt_weight = self._altitude()

        all_markers = self._detector.detect(frame, debug=False)
        # Hamis foltok kiszűrése alak és méret alapján (a kép szélébe lógó foltok
        # mérete nem megbízható, azokat itt nem vizsgálja: parancshoz és
        # pozícióhoz úgysem használja őket).
        kept, per_color = [], {}
        for m in all_markers:
            if not (m.touches_border or _dot_plausible(m, altitude_m, roll_rad, pitch_rad)):
                continue
            if per_color.get(m.color, 0) >= config.DOT_MAX_PER_COLOR:
                continue                 # színenként legfeljebb ennyit használ (a legnagyobbakat)
            per_color[m.color] = per_color.get(m.color, 0) + 1
            kept.append(m)
        if len(kept) != len(all_markers):
            self._dots_rejected += len(all_markers) - len(kept)
            log.debug("%d folt nem pötty alakú vagy méretű — kihagyom.",
                      len(all_markers) - len(kept))
        all_markers = kept
        self._vision_seq += 1
        self._line_dots_t = time.time()
        if not all_markers:
            self._line_dots = []
            self._tracked_cmd_marker = None
            return

        # A kép szélébe lógó pötty középpontja nem megbízható: parancshoz nem használjuk.
        cmd_markers = [m for m in all_markers
                       if m.marker_id in _COMMAND_MARKER_IDS and not m.touches_border]
        if self._state in (State.MARKER_FOLLOWING, State.HOVER, State.SEARCHING):
            self._announce_new_dots(cmd_markers, state, current_yaw, altitude_m)

        # a középre tartáshoz: minden teljesen látszó pötty helye a drónhoz képest (NED irányban)
        cy_, sy_ = math.cos(current_yaw), math.sin(current_yaw)
        line = []
        for m in cmd_markers:
            fwd, right = self._dot_offset_body(m.center_px, altitude_m)
            line.append((m.color, fwd * cy_ - right * sy_, fwd * sy_ + right * cy_))
        self._line_dots = line

        # Minden teljesen látszó pötty pozíció-fix (a pötty-térkép alapján), a parancs-pöttyök is.
        pos_markers = [m for m in all_markers if m.strategy == "color"]

        if (self._state == State.MARKER_FOLLOWING
                and time.time() >= self._marker_cooldown_until):
            # a már végrehajtott pötty kiesik; a többiből a kép közepéhez legközelebbi
            # (nem a legnagyobb confidence: egyenlőségnél a COLOR_MARKERS sorrendje döntene)
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
                # csak vízszintesen: a magasságot a magasságmérő adja, egy függőleges
                # eltérés (pl. felszállás közben) nem jelent hibás pötty-felismerést
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
                  "megbízhatóság=%.2f, pöttyök=%d",
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
        """GPS nélkül az ArduPilotnak meg kell adni a kiindulási koordinátát (EKF origin).
        Másodpercenként elküldi, amíg élesítésre nem kerül sor (legfeljebb 20-szor)."""
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
                    # az újraindított repülésvezérlő az alap üzenetgyakoriságra áll vissza
                    self._request_message_rates()
                    self._preflight_start = time.time()

            self._preflight_params_done = True

            # A lefelé néző kamera a földön állva nem lát pöttyöt, ezért a
            # pozícióbecslés az ismert felszállási pontból indul.
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
                # Pl. az előző futást felszállás közben állították le, és a drón a
                # szimulációban fent maradt. Innen nem lehet "felszállni", és a program
                # a földi kiindulóponttal számolna -> előbb le kell szállni.
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
        # Élesítést legfeljebb másodpercenként egyszer kér, és 5 s-onként ír ki egy sort.
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
            log.info("Felszállás kész, magasság: %.1f m. Indul a pöttykövetés.", alt)
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
            log.warning("Nincs pötty a képen (3 s óta nincs pozíció) — helyben lebegek.")
            self._enter_watchdog_hold()
            return

        now = time.time()
        dt  = 1.0 / config.CONTROL_LOOP_HZ
        h   = self._current_heading

        self._record_route()
        if self._should_return_home(now):
            self._start_return_home()
            return

        # --- oldalirányú középre tartás: a pöttyök vonala legyen a drón alatt ---------
        e = self._cross_track_error(now)
        v_lat = 0.0
        if e is not None:
            i_max = config.LATERAL_I_MAX
            self._lat_i = float(np.clip(self._lat_i + e * dt, -i_max, i_max))
            v_lat = float(np.clip(config.LATERAL_KP * e + config.LATERAL_KI * self._lat_i,
                                  -config.LATERAL_MAX_MPS, config.LATERAL_MAX_MPS))
            self._update_drift_monitor(e, now)
        else:
            self._lat_i *= 0.995          # nincs vonal a képen: az oldalszél-becslés lassan felejt

        speed = config.MARKER_FOLLOW_SPEED_MPS * (config.DRIFT_SLOW_FACTOR if self._slowing else 1.0)
        vx = speed * math.cos(h) - v_lat * math.sin(h)
        vy = speed * math.sin(h) + v_lat * math.cos(h)

        # --- parancs-pötty végrehajtása, ha a drón fölötte van -------------------------
        if self._tracked_cmd_marker is not None:
            marker_id, cx_px, cy_px = self._tracked_cmd_marker
            offset_px     = cx_px - config.FRAME_WIDTH / 2.0
            offset_fwd_px = cy_px - config.FRAME_HEIGHT / 2.0
            log.debug("Követett pötty: %s, eltérés oldalra %.0f px, előre-hátra %.0f px, oldalsebesség %.2f m/s",
                      _hu_color(marker_id), offset_px, offset_fwd_px, v_lat)

            if math.hypot(offset_px, offset_fwd_px) < config.MARKER_CENTER_THRESHOLD_PX:
                color = marker_id.replace("color_", "")
                # megerősítés: ugyanaz a szín több egymást követő képen is középen legyen
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
                        self._record_route(point=dot)          # a végrehajtott pötty a visszaút egy pontja
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
        """Oldalirányú eltérés a pöttyök vonalától méterben (+ = a vonal a drón jobb oldalán van).

        Csak az aktuális haladási irány vonalán lévő pöttyöket nézi: ne legyen a drón
        mögött, és oldalra legfeljebb CENTER_GATE_M-re legyen (a szomszédos szakasz vagy
        a kanyar utáni vezető pötty így nem húzza el)."""
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
        """Sodródásfigyelés: figyelmeztet, ha a drón tartósan eltávolodik a pöttyök
        vonalától, lassít, ha nagyon, és jelzi, ha erős oldalszél ellen kell tartania."""
        self._xtrack_ema += 0.2 * (e - self._xtrack_ema)
        a = abs(self._xtrack_ema)
        side = "balra" if self._xtrack_ema > 0 else "jobbra"      # e > 0: a vonal jobbra van -> a drón balra tolódott

        if a > config.DRIFT_WARN_M:
            if self._drift_since is None:
                self._drift_since = now
            if now - self._drift_since > 1.0 and now - self._last_drift_warn > 5.0:
                self._last_drift_warn = now
                log.warning("Sodródás: %s tolódtam a pöttyök vonalától — visszakorrigálok.", side)
        else:
            self._drift_since = None

        if not self._slowing and a > config.DRIFT_SLOW_M:
            self._slowing = True
            log.warning("Erősen elsodródtam (%s) — lassítok, amíg visszaérek a vonalra.", side)
        elif self._slowing and a < config.DRIFT_RECOVER_M:
            self._slowing = False
            log.info("Visszaértem a vonalra — normál sebességgel megyek tovább.")

        wind_comp = config.LATERAL_KI * self._lat_i                # tartós oldalirányú ellentartás, m/s
        if abs(wind_comp) > config.WIND_WARN_MPS and now - self._last_wind_warn > 15.0:
            self._last_wind_warn = now
            log.warning("Erős oldalszél %s — folyamatosan ellen kell tartanom.",
                        "balról" if wind_comp < 0 else "jobbról")

    def _altitude(self):
        """(magasság a talaj felett m-ben vagy None, súly a pozícióbecsléshez).

        Elsődleges forrás a lefelé néző lézeres távolságmérő (dőléssel korrigálva),
        ha nincs vagy érvénytelen, a barométer (ArduPilot)."""
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
        """Gyakoribb dőlés-, pozíció- és távolságmérő-üzenetek kérése az ArduPilottól.
        Ritka ATTITUDE (pl. 2 Hz) mellett fordulás közben elcsúszik a kamerás pozíció."""
        req = getattr(self._ctrl, "request_message_interval", None)
        if req is None:
            return
        for msg_id, hz in ((30, 25.0),    # ATTITUDE
                           (32, 10.0),    # LOCAL_POSITION_NED
                           (132, 20.0)):  # DISTANCE_SENSOR
            req(msg_id, hz)

    def _dot_offset_body(self, center_px, altitude_m: Optional[float]):
        """A pötty helye a drónhoz képest, vízszintesen, méterben: (előre, jobbra).

        Lefelé néző kamera, kép teteje = előre. A drón dőlését (roll, pitch) is
        figyelembe veszi: 10° előre dőlésnél 10 m-ről a kép közepe ~1,8 m-rel a drón
        elé esik, ezt dőlés nélkül hibásan számolnánk.
        """
        h = altitude_m if altitude_m else config.PLANNING_ALTITUDE_M
        fx = float(config.CAMERA_MATRIX[0, 0]); cx0 = float(config.CAMERA_MATRIX[0, 2])
        fy = float(config.CAMERA_MATRIX[1, 1]); cy0 = float(config.CAMERA_MATRIX[1, 2])
        nx = (float(center_px[0]) - cx0) / fx
        ny = (float(center_px[1]) - cy0) / fy
        ray = np.array([-ny, nx, 1.0])               # kamera -> test (előre, jobbra, le)
        att = self._ctrl.get_attitude() if self._ctrl else None
        if att:
            cr, sr = math.cos(att[0]), math.sin(att[0])
            cp, sp = math.cos(att[1]), math.sin(att[1])
            r_level = np.array([[cp, sp * sr, sp * cr],
                                [0.0, cr, -sr],
                                [-sp, cp * sr, cp * cr]])   # test -> vízszintes (yaw nélkül)
            ray = r_level @ ray
        if ray[2] < 0.2:
            ray = np.array([ray[0], ray[1], 0.2])
        t = h / ray[2]
        return float(t * ray[0]), float(t * ray[1])

    def _dot_ground_ned(self, center_px, state, yaw_rad: float,
                        altitude_m: Optional[float]) -> Optional[np.ndarray]:
        """A pötty közelítő helye a talajon (NED n, e) a lefelé néző kamera képéből."""
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
        """Minden új parancs-pöttyöt egyszer kiír magyarul: szín, jelentés, hol van."""
        for m in markers:
            p = self._dot_ground_ned(m.center_px, state, yaw_rad, altitude_m)
            if p is None:
                continue
            # ugyanolyan színű pöttyök legalább 4 m-re vannak egymástól a pályán,
            # így 3 m-en belül ugyanazt a pöttyöt látjuk (fordulás közben a becslés pontatlanabb)
            if any(c == m.color and float(np.linalg.norm(p - q)) < 3.0
                   for c, q in self._announced_dots):
                continue
            self._announced_dots.append((m.color, p))
            fwd, right = self._dot_offset_body(m.center_px, altitude_m)
            cmd = config.MARKER_COMMANDS.get(m.color, "")
            # csak az irány a drónhoz képest (előre / hátra / jobbra / balra), távolság nélkül
            where = []
            if abs(fwd) >= 0.5:
                where.append("előttem" if fwd > 0 else "mögöttem")
            if abs(right) >= 0.5:
                where.append("jobbra" if right > 0 else "balra")
            log.info("Látok egy %s pöttyöt (%s) — %s",
                     _hu_color(m.marker_id), _CMD_HU.get(cmd, "nincs parancs"),
                     " ".join(where) if where else "pont alattam")

    def _note_tick_time(self, elapsed: float, dt: float) -> None:
        """Lassú körök: egyenként csak a naplófájlba (DEBUG), a konzolra csak akkor
        figyelmeztet, ha tartósan lassú (az utolsó 10 s-ban a körök >25%-a), legfeljebb 30 s-onként."""
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
            log.info(">>> PIROS pötty a drón alatt → LESZÁLLÁS")
            self._land_reason = "elértem a piros pöttyöt"
            self._transition(State.LAND)
            return

        if cmd == "turn_left":
            self._current_heading = (
                (self._current_heading - math.pi / 2) + math.pi
            ) % (2 * math.pi) - math.pi
            log.info(">>> SÁRGA pötty a drón alatt → BALRA fordulok")
            self._reset_lateral()
        elif cmd == "turn_right":
            self._current_heading = (
                (self._current_heading + math.pi / 2) + math.pi
            ) % (2 * math.pi) - math.pi
            log.info(">>> FEKETE pötty a drón alatt → JOBBRA fordulok")
            self._reset_lateral()
        elif cmd == "straight":
            # Nincs cooldown: az egyenes pötty nem változtat irányt, és a már végrehajtott
            # pöttyöt a _spent_dots úgyis kiszűri. Így a kanyar-pötty előtt 5 m-rel is lehet kék.
            log.info(">>> KÉK pötty a drón alatt → EGYENESEN tovább előre")
            return

        self._marker_cooldown_until = time.time() + config.MARKER_COMMAND_COOLDOWN_S

    # ------------------------------------------------------------------ visszatérés

    def _record_route(self, point=None) -> None:
        """A bejárt út feljegyzése: minden végrehajtott pötty, és közben legalább
        RTH_BREADCRUMB_M-enként az aktuális helyzet (a pöttylánc koordinátáiban)."""
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
                self._route[-1] = p                   # a pötty pontosabb, mint a köztes pont
                return
        self._route.append(p)

    def _route_length_home(self, pos) -> float:
        """A visszaút hossza: a jelenlegi helytől a bejárt úton visszafelé a felszállási pontig."""
        if not self._route:
            return 0.0
        pts = [np.asarray(pos[:2], dtype=float)] + self._route[::-1]
        return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))

    def _should_return_home(self, now: float) -> bool:
        if not getattr(config, "RTH_ENABLED", True) or self._returning:
            return False
        t_test = getattr(config, "RTH_TEST_AFTER_S", None)
        if t_test is not None and self._follow_start_t is not None and now - self._follow_start_t >= t_test:
            log.warning("Teszt: %.0f s pöttykövetés után visszatérés indul.", t_test)
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
        """Becslés: hány % akkumulátor kell a visszaúthoz és a leszálláshoz.
        A fogyasztást az eddigi repülésből méri; ha még nincs elég adat, None."""
        if self._follow_start_t is None or self._follow_start_batt is None:
            return None
        batt = self._ctrl.get_battery_pct()
        st = self._tracker.get_state()
        elapsed = time.time() - self._follow_start_t
        used = self._follow_start_batt - (batt if batt is not None else self._follow_start_batt)
        if st is None or elapsed < 10.0 or used < 1:
            return None
        rate = used / elapsed                                     # %/s
        # a hazaút idejét a MÉRT átlagsebességgel számolja (nem a beállított 2 m/s-mal):
        # ha a drón (vagy a szimuláció) lassabb, a beállított értékkel alábecsülné az időt
        travelled = self._route_length_home(st["pos_ned"])       # oda = vissza
        speed = min(config.RTH_SPEED_MPS, max(0.3, travelled / elapsed))
        t_home = travelled / speed * getattr(config, "RTH_TIME_MARGIN", 1.0)
        # a leszállás idejét ugyanúgy a mért lassulással arányosítja (lassabb szimuláció /
        # lassabb drón esetén a leszállás is tovább tart)
        t_land = (config.PLANNING_ALTITUDE_M / 0.8 + 5.0) * (config.RTH_SPEED_MPS / speed)
        return rate * (t_home + t_land)

    def _start_return_home(self) -> None:
        st = self._tracker.get_state()
        pos = np.asarray(st["pos_ned"][:2] if st is not None else self._route[-1], dtype=float)
        # A visszaút: a jelenlegi helytől a bejárt úton visszafelé a felszállási pontig.
        # A köztes pontok a becsült helyzetből jönnek, ezért kicsit "hullámosak": a
        # Douglas–Peucker-egyszerűsítés ezt kisimítja, a kanyarokat viszont megtartja.
        # Így a hazaút néhány egyenes szakaszból áll, mint odafelé a pöttyök vonala.
        path = _simplify_path([pos] + [np.asarray(p, dtype=float) for p in self._route[::-1]],
                              config.RTH_PATH_SIMPLIFY_M)
        while len(path) > 2 and np.linalg.norm(path[1] - path[0]) < config.RTH_WAYPOINT_RADIUS_M:
            path.pop(1)                                    # a közvetlenül mellettünk lévő pontot kihagyja
        self._rth_path = path
        self._rth_seg = 0
        self._rth_turning = True                           # előbb helyben a szakasz irányába fordul
        self._returning = True
        self._rth_start_batt = self._ctrl.get_battery_pct()
        self._tracked_cmd_marker = None
        self._reset_lateral()
        log.info("Visszatérés: a bejárt úton megyek vissza a felszállási ponthoz (%d egyenes szakasz).",
                 len(path) - 1)
        self._notify(_SEV_NOTICE, "Visszateres a felszallasi ponthoz")
        self._transition(State.RETURN_HOME)

    def _state_return_home(self) -> None:
        """A bejárt utat egyenes szakaszonként, visszafelé repüli végig.

        Minden szakaszon a drón a szakasz irányába néz (nem az épp aktuális útpont felé,
        mert az a zajos helyzetbecslés miatt össze-vissza ugrálna), és ugyanúgy a
        pöttyök vonalán, a pöttyöket középen tartva halad, mint odafelé. Ha épp nem lát
        pöttyöt, a szakasz egyeneséhez igazodik. A szakasz végén (kanyarnál) lelassít,
        helyben az új irányba fordul, és csak utána indul tovább. A pöttyök színét a
        hazaúton nem parancsként, csak helyjelzőként használja."""
        if time.time() - self._last_fix_time > config.WATCHDOG_TIMEOUT_S:
            log.warning("Nincs pötty a képen (3 s óta nincs pozíció) — helyben lebegek.")
            self._enter_watchdog_hold()
            return
        st = self._tracker.get_state()
        path = getattr(self, "_rth_path", None)
        if st is None or not path or len(path) < 2:
            self._land_reason = "visszaértem a felszállási ponthoz"
            self._transition(State.LAND)
            return

        pos = np.asarray(st["pos_ned"][:2], dtype=float)

        # --- melyik szakaszon vagyunk: a szakasz végén (vagy azon túl) jön a következő --------
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
                self._reset_lateral()                     # kanyar a hazaúton: új vonal, helyben fordul
                self._rth_turning = True
                log.info("Hazaút: kanyar — %s fordulok.",
                         "JOBBRA" if _angle_diff(math.atan2(nb[1], nb[0]), math.atan2(u[1], u[0])) > 0
                         else "BALRA")

        h = math.atan2(u[1], u[0])                        # a szakasz iránya (állandó a szakaszon)
        self._current_heading = h
        yaw_err = _angle_diff(h, st["yaw_rad"])
        yaw_rate = float(np.clip(2.0 * yaw_err, -1.0, 1.0))
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([-math.sin(h), math.cos(h)])

        # --- fordulás helyben (indulásnál és a kanyaroknál) ---------------------------------
        if self._rth_turning:
            if abs(yaw_err) < math.radians(10):
                self._rth_turning = False
            else:
                # közben a szakasz egyeneséhez húz, hogy ne sodródjon el a kanyarban
                corr = -float(np.dot(pos - a, right))
                v = right * float(np.clip(0.5 * corr, -0.5, 0.5))
                self._ctrl.send_velocity_ned(float(v[0]), float(v[1]), 0.0, yaw_rate=yaw_rate)
                return

        now = time.time()
        dt = 1.0 / config.CONTROL_LOOP_HZ

        # --- oldalirányú középre tartás: a pöttyök vonala, ha nincs, a szakasz egyenese ------
        e = self._cross_track_error(now)
        if e is None:
            e_geo = -float(np.dot(pos - a, right))          # + = a vonal a drón jobb oldalán
            e = float(np.clip(e_geo, -config.CENTER_GATE_M, config.CENTER_GATE_M))
            self._lat_i *= 0.995                           # geometriai hiba: az oldalszél-becslést nem tanítja
            v_lat = float(np.clip(config.LATERAL_KP * e + config.LATERAL_KI * self._lat_i,
                                  -config.LATERAL_MAX_MPS, config.LATERAL_MAX_MPS))
        else:
            i_max = config.LATERAL_I_MAX
            self._lat_i = float(np.clip(self._lat_i + e * dt, -i_max, i_max))
            v_lat = float(np.clip(config.LATERAL_KP * e + config.LATERAL_KI * self._lat_i,
                                  -config.LATERAL_MAX_MPS, config.LATERAL_MAX_MPS))
            self._update_drift_monitor(e, now)

        speed = config.RTH_SPEED_MPS * (config.DRIFT_SLOW_FACTOR if self._slowing else 1.0)
        speed *= max(0.0, math.cos(yaw_err))               # amíg nem néz pontosan előre, lassabban
        speed = min(speed, 0.6 * max(to_end, 0.0) + 0.3)   # a szakasz végén (kanyar / otthon) lassít
        v = fwd * speed + right * v_lat
        self._ctrl.send_velocity_ned(float(v[0]), float(v[1]), 0.0, yaw_rate=yaw_rate)

    def _reset_lateral(self) -> None:
        """Fordulás után új a vonal iránya: a középre tartás és a sodródásfigyelés újraindul."""
        self._lat_i = 0.0
        self._xtrack_ema = 0.0
        self._drift_since = None
        self._slowing = False

    def _state_searching(self) -> None:
        if time.time() - self._last_fix_time < config.WATCHDOG_TIMEOUT_S:
            if self._returning:
                log.info("Újra látok pöttyöt — folytatom a visszatérést.")
                self._transition(State.RETURN_HOME)
            else:
                log.info("Újra látok pöttyöt — folytatom a pöttykövetést.")
                self._transition(State.MARKER_FOLLOWING)
            return

        self._ctrl.send_velocity_ned(0.0, 0.0, 0.0,
                                     yaw_rate=config.SEARCH_YAW_RATE_RADS)

        elapsed   = time.time() - self._search_start_time
        full_turn = (2 * math.pi) / config.SEARCH_YAW_RATE_RADS
        if elapsed >= full_turn:
            log.warning("Körbefordultam (360°), de nem találtam pöttyöt. Leszállok.")
            self._land_reason = "elveszett a pötty, a 360°-os keresés sem talált"
            self._notify(_SEV_WARNING, "ELVESZETT A POTTY - LESZALLOK")
            self._transition(State.LAND)

    def _state_hover(self) -> None:
        # Csak akkor folytatja, ha valóban érkezett új pötty-fix (nem elég, hogy a
        # tracker bizonytalansága még kicsi), különben HOVER és követés között ingázna.
        if time.time() - self._last_fix_time < config.WATCHDOG_TIMEOUT_S:
            self._pid.reset()
            self._last_fix_time = time.time()
            if self._returning:
                log.info("Újra van pozíció — folytatom a visszatérést.")
                self._transition(State.RETURN_HOME)
            else:
                log.info("Újra van pozíció — folytatom a pöttykövetést.")
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
            log.warning("%.0f s lebegés után sincs pötty — 360°-os keresés indul.",
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

        # Ha a repülésvezérlő nem fogadta el a leszállást, másodpercenként újrapróbálja,
        # amíg el nem fogadja (közben a kamerás pozíció továbbra is megy).
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
        """Vészleszállás helyben. A program nem lép ki azonnal: a vezérlőciklus a
        földet érésig fut tovább, így a repülésvezérlő leszállás közben is megkapja a
        kamerás pozíciót (enélkül a szél elsodorhatná a drónt)."""
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
        """Leszállási parancs nyugta-ellenőrzéssel; ha nincs elfogadó nyugta, LAND üzemmód."""
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
        """A program életjele másodpercenként. A vezérlőciklusból megy, így ha a
        ciklus megáll (lefagyás), az életjel is elmarad, és a repülésvezérlő leszáll."""
        now = time.time()
        if now - self._last_hb_sent < 1.0:
            return
        self._last_hb_sent = now
        send = getattr(self._conn, "send_heartbeat", None)
        if send is not None:
            send()

    def _handle_stop_request(self) -> None:
        """Ctrl+C / SIGTERM után: ha a drón a levegőben van, leszállítja (a ciklus
        tovább fut a földet érésig), ha a földön van, a program leáll."""
        if self._state in (State.LAND, State.EMERGENCY, State.DONE):
            return                                   # már leszáll / leállt
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

        # Vészleszállás közben a többi ellenőrzés (magasság, terület, dőlés) már nem
        # válthat újra vészhelyzetbe, mert az megakasztaná a leszállás felügyeletét.
        if self._state == State.EMERGENCY:
            return True

        batt = self._ctrl.get_battery_pct()
        # Csak repülés közben: leszállás után (KÉSZ) vagy már leszállás közben nincs teendő.
        # (Korábban a KÉSZ állapotból is visszaváltott LESZÁLLÁSRA, és a kettő között
        # végtelenül ingázott, ha leszállás után is alacsony volt az akkumulátor.)
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
            # egyszer jelez (korábban minden körben, másodpercenként 20-szor kiírta)
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
            # csak a tartós túllépés vészhelyzet: egy-egy hibás mérés nem
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

        # A távolságmérő lefelé néz (magasságmérő), ezért itt nem akadályérzékelő:
        # a túl alacsony repülést a MIN_ALTITUDE_M ellenőrzés kezeli.
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
            # a repülésvezérlő felé csak ékezet nélküli szöveg megy (MAVLink STATUSTEXT)
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
        """Ctrl+C / SIGTERM. A jelkezelőben nem várunk (ott a vezérlőciklus állna, és a
        repülésvezérlő nem kapna kamerás pozíciót): csak jelzést állítunk be, a
        leszállást és a földet érés megvárását a vezérlőciklus végzi, a végén a
        start() finally ága állítja le a kamerát és a kapcsolatot."""
        if self._stop_requested:
            # második Ctrl+C: azonnali kilépés. Ha a drón már leszáll, a repülésvezérlő
            # befejezi; ha nem, a program életjele elmarad, és a repülésvezérlő leszáll.
            log.warning("Ismételt leállítás — azonnali kilépés.")
            raise SystemExit(1)
        log.warning("Leállítás (Ctrl+C) — biztonságos leállás.")
        self._stop_requested = True

if __name__ == "__main__":
    nav = NavigationSystem()
    nav.start()
