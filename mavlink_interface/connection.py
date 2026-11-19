
import os
import threading
import time
from typing import Optional

# MAVLink 2 kell: a VISION_POSITION_ESTIMATE kovariancia mezője csak MAVLink 2-ben létezik.
# A pymavlink alapból MAVLink 1-et használ, ha ez a változó nincs beállítva, ezért az
# importálás előtt kell megadni.
os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil

import config
from utils.logger import get_logger

log = get_logger(__name__)

def _is_autopilot_heartbeat(hb) -> bool:
    """Repülésvezérlőtől jött-e az életjel (és nem földi állomástól, kamerától,
    gimbaltól vagy egy fedélzeti programtól)."""
    return (hb.type not in (mavutil.mavlink.MAV_TYPE_GCS,
                            mavutil.mavlink.MAV_TYPE_GIMBAL,
                            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER)
            and hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID)


class MAVLinkConnection:

    def __init__(self, connection_string: str = config.MAVLINK_CONNECTION_STRING,
                 baud: int = config.MAVLINK_BAUD):
        self._conn_str = connection_string
        self._baud = baud
        self._master: Optional[mavutil.mavfile] = None
        self._lock = threading.Lock()
        self._running = False
        self._last_heartbeat = 0.0
        self._rx_thread: Optional[threading.Thread] = None

        self._messages: dict = {}
        self._msg_times: dict = {}      # üzenettípus -> mikor érkezett utoljára (time.time())
        self._msg_lock = threading.Lock()

        # A repülésvezérlő rendszerazonosítója. Csak az ettől érkező üzeneteket, és
        # életjelként csak a repülésvezérlő életjelét fogadjuk el: egy földi állomás
        # (Mission Planner, QGroundControl) életjele így nem írhatja felül az övét.
        self._fc_sys: Optional[int] = None

    def connect(self) -> bool:
        log.info("Csatlakozás: %s …", self._conn_str)
        try:
            self._master = mavutil.mavlink_connection(
                self._conn_str,
                baud=self._baud,
                source_system=config.MAVLINK_SOURCE_SYSTEM,
            )
        except Exception as exc:
            log.error("A csatlakozás nem sikerült: %s", exc)
            return False

        log.info("Várakozás a repülésvezérlő jelére (heartbeat) …")
        hb = self._wait_vehicle_heartbeat(config.HEARTBEAT_TIMEOUT_S)
        if hb is None:
            log.error("%.1f s alatt nem jött jel (heartbeat) a repülésvezérlőtől", config.HEARTBEAT_TIMEOUT_S)
            try:
                self._master.close()          # a megnyitott portot nem hagyjuk nyitva
            except Exception:
                pass
            self._master = None
            return False

        self._fc_sys = hb.get_srcSystem()
        self._master.target_system = self._fc_sys
        # A parancsok célkomponense marad a pymavlink alapértéke (0 = a rendszer minden
        # komponense), ahogy a SITL-tesztekben is működött.
        with self._msg_lock:
            self._messages["HEARTBEAT"] = hb
            self._msg_times["HEARTBEAT"] = time.time()
        self._last_heartbeat = time.time()
        log.info("Kapcsolat rendben: rendszer %d, komponens %d", self._fc_sys, hb.get_srcComponent())

        self._running = True
        self._rx_thread = threading.Thread(target=self._recv_loop, daemon=True,
                                           name="mavlink-rx")
        self._rx_thread.start()
        return True

    def disconnect(self) -> None:
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
            self._rx_thread = None
        if self._master:
            self._master.close()
            self._master = None
        with self._msg_lock:
            self._messages.clear()
            self._msg_times.clear()
        self._last_heartbeat = 0.0
        log.debug("Kapcsolat bontva a repülésvezérlővel.")

    def is_healthy(self) -> bool:
        return (time.time() - self._last_heartbeat) < config.HEARTBEAT_TIMEOUT_S

    def send_heartbeat(self) -> None:
        """A program saját életjele. Ha ez elmarad (a program lefagyott vagy kilépett),
        a repülésvezérlő a beállított vészműveletet hajtja végre (FS_GCS_ENABLE, lásd config)."""
        with self._lock:
            if self._master:
                try:
                    self._master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, mavutil.mavlink.MAV_STATE_ACTIVE)
                except Exception as exc:
                    log.debug("Életjel küldése nem sikerült: %s", exc)

    def send_raw(self, msg) -> None:
        with self._lock:
            if self._master:
                self._master.mav.send(msg)

    def get_message(self, msg_type: str):
        with self._msg_lock:
            return self._messages.get(msg_type)

    def get_message_age(self, msg_type: str) -> float:
        """Hány másodperce jött az utolsó ilyen üzenet (végtelen, ha még egy sem)."""
        with self._msg_lock:
            t = self._msg_times.get(msg_type)
        return float("inf") if t is None else time.time() - t

    def flush_message(self, msg_type: str) -> None:
        with self._msg_lock:
            self._messages.pop(msg_type, None)

    @property
    def master(self):
        return self._master

    @property
    def target_system(self) -> int:
        if self._fc_sys is not None:
            return self._fc_sys
        return self._master.target_system if self._master else 1

    @property
    def target_component(self) -> int:
        return self._master.target_component if self._master else 1

    def _wait_vehicle_heartbeat(self, timeout: float):
        """Az első olyan HEARTBEAT, amely repülésvezérlőtől jön (nem földi állomástól,
        nem kamerától stb.). None, ha a megadott időn belül nem érkezik ilyen."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            hb = self._master.recv_match(type="HEARTBEAT", blocking=True,
                                         timeout=max(0.1, deadline - time.time()))
            if hb is None:
                continue
            if _is_autopilot_heartbeat(hb):
                return hb
            log.debug("Nem repülésvezérlőtől jött életjel (rendszer %d, típus %d) — kihagyom.",
                      hb.get_srcSystem(), hb.type)
        return None

    def _recv_loop(self) -> None:
        while self._running:
            try:
                msg = self._master.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                mtype = msg.get_type()
                if mtype == "BAD_DATA":
                    continue
                # csak a repülésvezérlőtől érkező üzeneteket fogadja el
                if self._fc_sys is not None and msg.get_srcSystem() != self._fc_sys:
                    continue
                if mtype == "HEARTBEAT":
                    if not _is_autopilot_heartbeat(msg):
                        continue              # pl. a kamera, a gimbal vagy egy földi állomás életjele
                    self._last_heartbeat = time.time()
                with self._msg_lock:
                    self._messages[mtype] = msg
                    self._msg_times[mtype] = time.time()
            except Exception as exc:
                log.warning("Fogadási hiba: %s", exc)
