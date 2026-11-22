import math
import time
from typing import Optional, Tuple

from pymavlink import mavutil

import config
from mavlink_interface.connection import MAVLinkConnection
from utils.logger import get_logger

log = get_logger(__name__)

MAV_FRAME_LOCAL_NED = mavutil.mavlink.MAV_FRAME_LOCAL_NED

_MASK_VEL_YAWRATE = 0x07C7
_MASK_VEL_ONLY    = 0x0FC7
_MASK_POS_YAW     = 0x0BF8
class DroneController:
    def __init__(self, conn: MAVLinkConnection):
        self._conn = conn

    def get_parameter(self, name: str, timeout: float = 3.0) -> Optional[float]:
        self._conn.flush_message("PARAM_VALUE")
        self._conn.master.mav.param_request_read_send(
            self._conn.target_system,
            self._conn.target_component,
            name.encode("utf-8"),
            -1,
        )
        start = time.time()
        while time.time() - start < timeout:
            msg = self._conn.get_message("PARAM_VALUE")
            if msg and msg.param_id.rstrip("\x00") == name:
                return float(msg.param_value)
            time.sleep(0.02)
        log.warning("Paraméter olvasása időtúllépés: %s", name)
        return None

    def set_parameter(self, name: str, value: float,
                      timeout: float = 3.0) -> bool:
        self._conn.flush_message("PARAM_VALUE")
        self._conn.master.mav.param_set_send(
            self._conn.target_system,
            self._conn.target_component,
            name.encode("utf-8"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        start = time.time()
        while time.time() - start < timeout:
            msg = self._conn.get_message("PARAM_VALUE")
            if msg and msg.param_id.rstrip("\x00") == name:
                log.debug("Paraméter %s = %.4f", name, msg.param_value)
                return True
            time.sleep(0.02)
        log.error("Paraméter beállítása időtúllépés: %s", name)
        return False

    def configure_for_gps_free(self) -> Tuple[bool, bool]:
        reboot_needed = False
        critical = {"GPS_TYPE", "EK3_ENABLE", "EK3_SRC1_POSXY",
                    "EK3_SRC1_VELXY", "EK3_SRC1_POSZ", "EK3_SRC1_VELZ",
                    "EK3_SRC1_YAW"}
        for name, target in config.ARDUPILOT_PARAMS.items():
            is_critical = name in critical
            current = self.get_parameter(name)

            if current is not None and abs(current - float(target)) < 0.001:
                continue

            if not self.set_parameter(name, float(target)):
                if is_critical:
                    log.error("Nem sikerült beállítani a kritikus paramétert: %s — megszakítom.", name)
                    return False, False
                log.warning("Nem sikerült beállítani az opcionális paramétert: %s — folytatom.", name)
                continue

            if is_critical:
                reboot_needed = True

        return True, reboot_needed

    def reboot_autopilot(self) -> None:
        self._conn.master.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )
        log.info("Újraindítási parancs elküldve. Várakozás az újracsatlakozásra …")

    def send_vision_position_estimate(self, pos_ned: "np.ndarray",
                                      roll: float, pitch: float,
                                      yaw: float,
                                      pos_std: Optional["np.ndarray"] = None,
                                      yaw_std: Optional[float] = None) -> None:
        usec = int(time.time() * 1e6)
        cov = [float("nan")] * 21
        if pos_std is not None and yaw_std is not None:
            cov = [0.0] * 21
            cov[0], cov[6], cov[11] = (float(pos_std[0]) ** 2, float(pos_std[1]) ** 2,
                                       float(pos_std[2]) ** 2)
            cov[15] = cov[18] = 0.01 ** 2
            cov[20] = float(yaw_std) ** 2
        args = (usec, float(pos_ned[0]), float(pos_ned[1]), float(pos_ned[2]),
                float(roll), float(pitch), float(yaw))
        try:
            try:
                self._conn.master.mav.vision_position_estimate_send(*args, cov, 0)
            except TypeError:
                self._conn.master.mav.vision_position_estimate_send(*args)
        except Exception as exc:
            log.warning("VISION_POSITION_ESTIMATE küldési hiba: %s", exc)

    def set_mode(self, mode_id: int) -> bool:
        self._conn.flush_message("HEARTBEAT")
        self._conn.master.mav.set_mode_send(
            self._conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        start = time.time()
        while time.time() - start < 3.0:
            msg = self._conn.get_message("HEARTBEAT")
            if msg and msg.custom_mode == mode_id:
                log.info("Üzemmód átváltva: %d", mode_id)
                return True
            time.sleep(0.05)
        log.error("Az üzemmódváltás (%d) időtúllépéssel meghiúsult", mode_id)
        return False

    def set_guided_mode(self) -> bool:
        return self.set_mode(config.GUIDED_MODE_ID)

    def arm(self) -> bool:
        self._conn.flush_message("COMMAND_ACK")
        self._conn.master.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0, 0, 0, 0, 0,
        )
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

    def disarm(self) -> bool:
        self._conn.flush_message("COMMAND_ACK")
        self._conn.master.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

    def takeoff(self, altitude_m: float) -> bool:
        altitude_m = max(config.MIN_ALTITUDE_M,
                         min(altitude_m, config.MAX_ALTITUDE_M))
        self._conn.flush_message("COMMAND_ACK")
        self._conn.master.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0,
            altitude_m,
        )
        log.info("Felszállási parancs elküldve: %.1f m", altitude_m)
        return self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)

    def land(self) -> bool:
        self._conn.flush_message("COMMAND_ACK")
        self._conn.master.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        ok = self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_LAND)
        if ok:
            log.info("A leszállási parancsot nyugtázta.")
        return ok

    def command_land(self) -> bool:
        if self.land():
            return True
        log.warning("Nem jött elfogadó nyugta a leszállásra — LAND üzemmódra váltok.")
        return self.set_mode(config.LAND_MODE_ID)

    def get_mode(self) -> Optional[int]:
        msg = self._conn.get_message("HEARTBEAT")
        return None if msg is None else int(msg.custom_mode)

    def return_to_launch(self) -> None:
        self._conn.flush_message("COMMAND_ACK")
        self._conn.master.mav.command_long_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0,
        )
        self._wait_for_ack(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)
        log.info("A visszatérési (RTL) parancsot nyugtázta.")

    def send_velocity_ned(self, vx: float, vy: float, vz: float,
                          yaw_rate: float = 0.0) -> None:
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > config.MAX_SPEED_MPS:
            scale = config.MAX_SPEED_MPS / speed
            vx *= scale; vy *= scale; vz *= scale

        self._conn.master.mav.set_position_target_local_ned_send(
            0,
            self._conn.target_system,
            self._conn.target_component,
            MAV_FRAME_LOCAL_NED,
            _MASK_VEL_YAWRATE,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0,
            yaw_rate,
        )

    def send_position_ned(self, x: float, y: float, z: float,
                          yaw: float = 0.0) -> None:
        self._conn.master.mav.set_position_target_local_ned_send(
            0,
            self._conn.target_system,
            self._conn.target_component,
            MAV_FRAME_LOCAL_NED,
            _MASK_POS_YAW,
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            yaw,
            0,
        )

    def get_local_position(self) -> Optional[tuple]:
        msg = self._conn.get_message("LOCAL_POSITION_NED")
        if msg is None:
            return None
        return msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz

    def get_velocity_ned_stamped(self) -> Optional[tuple]:
        msg = self._conn.get_message("LOCAL_POSITION_NED")
        if msg is None:
            return None
        return msg.time_boot_ms, msg.vx, msg.vy, msg.vz

    def get_velocity_ned(self) -> Optional[tuple]:
        msg = self._conn.get_message("LOCAL_POSITION_NED")
        if msg is None:
            return None
        return msg.vx, msg.vy, msg.vz

    def get_attitude(self) -> Optional[tuple]:
        msg = self._conn.get_message("ATTITUDE")
        if msg is None:
            return None
        return msg.roll, msg.pitch, msg.yaw

    def set_ekf_origin(self, lat_deg: float, lon_deg: float, alt_m: float) -> None:
        m = self._conn.master
        lat, lon = int(lat_deg * 1e7), int(lon_deg * 1e7)
        try:
            m.mav.set_gps_global_origin_send(m.target_system, lat, lon, int(alt_m * 1000))
            m.mav.command_long_send(
                m.target_system, m.target_component,
                179,
                0,
                0,
                0, 0, 0,
                lat_deg, lon_deg, alt_m,
            )
        except Exception as exc:
            log.warning("Az EKF kiindulópont küldése nem sikerült: %s", exc)

    def get_altitude_m(self) -> float:
        msg = self._conn.get_message("LOCAL_POSITION_NED")
        if msg:
            return -float(msg.z)
        msg = self._conn.get_message("GLOBAL_POSITION_INT")
        if msg:
            return msg.relative_alt / 1000.0
        return float("nan")

    def get_battery_pct(self) -> Optional[int]:
        msg = self._conn.get_message("BATTERY_STATUS")
        if msg and msg.battery_remaining >= 0:
            return int(msg.battery_remaining)
        msg = self._conn.get_message("SYS_STATUS")
        if msg and msg.battery_remaining >= 0:
            return int(msg.battery_remaining)
        return None

    def get_rangefinder_m(self) -> Optional[float]:
        msg = self._conn.get_message("DISTANCE_SENSOR")
        if not msg or msg.current_distance <= 0:
            return None
        age = getattr(self._conn, "get_message_age", lambda _t: 0.0)("DISTANCE_SENSOR")
        if age > 0.5:
            return None
        orient = getattr(msg, "orientation", 25)
        if orient not in (25,):
            return None
        return msg.current_distance / 100.0

    def request_message_interval(self, msg_id: int, hz: float) -> None:
        try:
            self._conn.master.mav.command_long_send(
                self._conn.target_system, self._conn.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                float(msg_id), 1e6 / hz, 0, 0, 0, 0, 0)
        except Exception as exc:
            log.debug("Üzenetgyakoriság kérése nem sikerült (%d): %s", msg_id, exc)

    def send_statustext(self, severity: int, text: str) -> None:
        try:
            self._conn.master.mav.statustext_send(
                severity,
                text[:50].encode("utf-8"),
            )
        except Exception as exc:
            log.warning("STATUSTEXT küldési hiba: %s", exc)

    def is_armed(self) -> bool:
        msg = self._conn.get_message("HEARTBEAT")
        if msg is None:
            return False
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def _wait_for_ack(self, command: int, timeout: float = 3.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            msg = self._conn.get_message("COMMAND_ACK")
            if msg and msg.command == command:
                ok = msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
                if not ok:
                    log.debug("A %d. parancsot elutasította: eredmény=%d",
                                command, msg.result)
                return ok
            time.sleep(0.02)
        log.error("Nem jött nyugta a %d. parancsra", command)
        return False

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
