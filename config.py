
import numpy as np

MAVLINK_CONNECTION_STRING = "udp:127.0.0.1:14550"
MAVLINK_BAUD = 115200
# A program saját MAVLink-azonosítója. Nem 255, mert azt a földi állomások
# (Mission Planner, QGroundControl, MAVProxy) használják. Az ArduPilot SYSID_MYGCS
# paramétere is erre van állítva (lásd lent), így csak a program életjele számít.
MAVLINK_SOURCE_SYSTEM = 254
HEARTBEAT_TIMEOUT_S = 5.0
GUIDED_MODE_ID = 4
LAND_MODE_ID   = 9   # ArduCopter LAND üzemmód (vészleszálláshoz)
LAND_WAIT_TIMEOUT_S = 60.0   # leállításkor legfeljebb ennyit vár a földet érésre

CONFIGURE_ARDUPILOT = True
ARDUPILOT_PARAMS = {
    "GPS_TYPE":        0,
    "EK3_ENABLE":      1,
    "EK3_SRC1_POSXY":  6,
    "EK3_SRC1_VELXY":  0,
    "EK3_SRC1_POSZ":   1,
    "EK3_SRC1_VELZ":   0,
    "EK3_SRC1_YAW":    1,   # kompasz (a színes pötty nem ad irányt)
    "VISO_TYPE":       1,
    "VISO_POS_M_NSE":  0.2,
    "VISO_YAW_M_NSE":  0.05,
    # Vészműveletek: ha a program életjele 5 s-ig elmarad (lefagyott, kilépett),
    # a repülésvezérlő helyben leszáll (5 = LAND). GNSS nélkül az RTL nem használható.
    "SYSID_MYGCS":     254,   # = MAVLINK_SOURCE_SYSTEM
    "FS_GCS_ENABLE":   5,
    "FS_EKF_ACTION":   1,     # ha az EKF3 helyzetbecslése elromlik: leszállás
}

CAMERA_MATRIX = np.array([
    [800.0,   0.0, 640.0],
    [  0.0, 800.0, 360.0],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

CAMERA_FPS = 30
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

T_BODY_TO_CAM = np.array([0.0, 0.0, -0.05], dtype=np.float64)

ARUCO_DICT_ID = "DICT_4X4_100"
ARUCO_MARKER_SIZE_M = 0.30

ARUCO_WORLD_POSITIONS = {
     0: np.array([  0.0,   0.0, 0.0]),
     1: np.array([ 20.0,   0.0, 0.0]),
     2: np.array([ 20.0,  20.0, 0.0]),
     3: np.array([  0.0,  20.0, 0.0]),
    10: np.array([ 50.0,  10.0, 0.0]),
    11: np.array([100.0,  10.0, 0.0]),
}

DOT_DIAMETER_M = 0.60  # színes pötty átmérője (10 m-hez méretezve)

COLOR_MARKERS = {
    "red": {
        "hsv_ranges": [
            (np.array([  0, 120, 70]), np.array([ 10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255])),
        ],
        "min_area_px": 800,
        "world_pos": np.array([30.0, 5.0, 0.0]),
    },
    "blue": {
        "hsv_ranges": [
            (np.array([100, 150, 50]), np.array([140, 255, 255])),
        ],
        "min_area_px": 800,
        "world_pos": np.array([60.0, 15.0, 0.0]),
    },
    "yellow": {
        "hsv_ranges": [
            (np.array([20, 100, 100]), np.array([30, 255, 255])),
        ],
        "min_area_px": 800,
        "world_pos": np.array([80.0, 0.0, 0.0]),
    },
    "green": {
        "hsv_ranges": [
            (np.array([40, 70, 70]), np.array([80, 255, 255])),
        ],
        "min_area_px": 800,
        "world_pos": np.array([0.0, 0.0, 0.0]),
    },
    "purple": {
        "hsv_ranges": [
            (np.array([130, 60, 60]), np.array([160, 255, 255])),
        ],
        "min_area_px": 800,
        "world_pos": np.array([0.0, 0.0, 0.0]),
    },
    "black": {
        "hsv_ranges": [
            (np.array([0, 0, 0]), np.array([180, 255, 50])),
        ],
        "min_area_px": 800,
        "world_pos": np.array([0.0, 0.0, 0.0]),
    },
}

EKF_PROCESS_NOISE_POS = 0.05
EKF_PROCESS_NOISE_VEL = 0.10
EKF_MEAS_NOISE_POS    = 0.20
EKF_MEAS_NOISE_VEL    = 0.30
EKF_MEAS_NOISE_YAW    = 0.05
EKF_MAX_UNCERTAINTY_M = 3.0

PLANNING_ALTITUDE_M = 10.0

PID_X   = dict(kp=0.8, ki=0.05, kd=0.3, i_max=2.0, out_max=5.0)
PID_Y   = dict(kp=0.8, ki=0.05, kd=0.3, i_max=2.0, out_max=5.0)
PID_Z   = dict(kp=1.0, ki=0.10, kd=0.4, i_max=1.5, out_max=1.5)  # max 11 m miatt csökkentve
PID_YAW = dict(kp=1.2, ki=0.0,  kd=0.2, i_max=0.5, out_max=1.0)

CONTROL_LOOP_HZ     = 20
POSITION_HOLD_SPEED = 0.5

MAX_TILT_DEG             = 30.0
MAX_SPEED_MPS            = 8.0
MIN_ALTITUDE_M           = 3.0
# A 10 m-es repülési magasság fölött 2,5 m tartalék: a barométer néhány százalékos hibája
# és a felszállás végi túllendülés még ne váltson ki vészhelyzetet.
MAX_ALTITUDE_M           = 12.5
# A magassági határ túllépése csak akkor vészhelyzet, ha legalább ennyi ideig tart
# (egy-egy hibás magasságmérés így nem indít vészleszállást).
ALT_LIMIT_PERSIST_S      = 1.0
WATCHDOG_TIMEOUT_S       = 3.0
HOVER_TIMEOUT_S          = 5.0
TAKEOFF_TIMEOUT_S        = 45.0
LAND_TIMEOUT_S           = 60.0
LAND_COMPLETE_ALT_M      = 0.5

EKF_MAX_POSITION_JUMP_M  = 3.0
OBSTACLE_CLEARANCE_M     = 2.0
CAMERA_HEALTH_TIMEOUT_S  = 2.0
LOOP_SLOW_FACTOR         = 2.0
GEOFENCE_NORTH = (-10.0, 250.0)
GEOFENCE_EAST  = (-10.0, 250.0)

BATTERY_FAILSAFE_PCT = 15
BATTERY_WARNING_PCT  = 25

PREFLIGHT_FIX_TIMEOUT_S = 30.0
MARKER_COMMANDS = {
    "blue":   "straight",   # a zöld helyett (fűben a zöld téves jelzést adna)
    "black":  "turn_right",
    "yellow": "turn_left",
    "red":    "land",
}
# Ennyi méteren belül nem hajtja végre újra az utolsó néhány (MARKER_SPENT_MEMORY)
# már végrehajtott parancs-pöttyöt.
MARKER_REARM_DIST_M          = 2.5
# Egy parancs-pötty csak akkor fut le, ha ennyi egymást követő képen is középen van
# (egy-egy téves képkocka így nem fordítja el a drónt).
MARKER_CONFIRM_FRAMES        = 3
MARKER_SPENT_MEMORY          = 3
# Hamis pöttyök kiszűrése (pl. a drón árnyéka, sötét folt a fűben):
#   * a folt legalább ennyire legyen kör alakú (a felismerő 0..1 közötti értéke),
#   * a látszó átmérője legfeljebb ennyivel (arányosan) térjen el attól, amekkorának
#     egy DOT_DIAMETER_M méretű pöttynek a mért magasságból látszania kell.
DOT_MIN_SHAPE_CONF           = 0.8
DOT_SIZE_TOLERANCE           = 0.35

MARKER_FOLLOW_SPEED_MPS      = 2.0
MARKER_COMMAND_COOLDOWN_S    = 4.0
MARKER_CENTER_THRESHOLD_PX   = 120
MARKER_LATERAL_GAIN          = 0.003
SEARCH_YAW_RATE_RADS         = 0.3


# --- Színes pöttyös navigáció (ArUco helyett) ---------------------------------
# A pöttyök ismert helye a pályán (NED, méter, a felszállási ponthoz képest).
# Szimulációban a sim/config_sim.py tölti ki a sim/field_layout.py alapján;
# valós pályán ide kell felvenni őket, pl.:
#   DOT_MAP = [dict(color="blue", n=10.0, e=0.0), dict(color="black", n=40.0, e=0.0)]
DOT_MAP = []
DOT_ASSOC_GATE_M    = 4.0   # ennél messzebb a jósolt helyzettől nem fogad el pötty-azonosítást
DOT_MAX_PER_COLOR   = 3     # színenként legfeljebb ennyi pöttyöt használ egy képen
DOT_MAX_CANDIDATES_PER_COLOR = 8   # ennyi jelöltet vizsgál színenként a hamis foltok kiszűrése előtt
DOT_SIZE_ALT_WEIGHT = 0.5   # a pötty méretéből számolt magasság súlya (a barométeré 1.0)

# --- Térkép nélküli navigáció (pöttylánc) ---------------------------------------
# True: a drón NEM tudja előre a pöttyök helyét, menet közben tanulja meg őket.
# False: a régi működés, a DOT_MAP-ban megadott helyekkel.
MAP_FREE                    = True
DOT_ASSOC_GATE_MAPFREE_M    = 2.0   # ennyin belül számít egy látott pötty egy már ismertnek
NEW_DOT_MAX_DIST_M          = 3.5   # új pöttyöt csak a drón alól / közeléből vesz fel (pontosabb)
NEW_DOT_MIN_SEPARATION_M    = 3.0   # ennél közelebb egy azonos színű ismerthez: nem veszi fel újként
# Holtsáv kezelése: ha egy látott pötty a 2 m-es kapun kívül, de 3 m-en belül esik egy
# ismert, azonos színű pöttyhöz, és a következő legközelebbi azonos színű pötty legalább
# ennyivel messzebb van, akkor ugyanannak a pöttynek tekinti (a helyzetbecslés elsodródott).
DOT_ASSOC_MARGIN_M          = 2.0

# --- Lézeres magasságmérés --------------------------------------------------------
# A lefelé néző távolságmérő (szimulációban az ArduPilot SITL szimulált lézere,
# valós drónon pl. egy lézeres/ToF távolságmérő) adja a magasságot; ha nincs, a barométer.
RANGEFINDER_IS_ALTIMETER    = True
RANGEFINDER_MIN_M           = 0.3
RANGEFINDER_MAX_M           = 20.0  # valós szenzornál a gyári hatótáv
RANGEFINDER_ALT_WEIGHT      = 4.0   # mennyivel bízzon jobban benne, mint a pötty méretéből számolt magasságban

# --- Középre tartás és sodródásfigyelés -------------------------------------------
LATERAL_KP                  = 0.8   # oldalsebesség (m/s) 1 m oldalirányú eltérésre
LATERAL_KI                  = 0.15  # tartós eltérés (oldalszél) ellensúlyozása
LATERAL_I_MAX               = 6.0
LATERAL_MAX_MPS             = 1.5
CENTER_GATE_M               = 3.0   # csak az ennyire a vonalon lévő pöttyöket használja a középre tartáshoz
DRIFT_WARN_M                = 1.0   # ennél nagyobb tartós eltérés: figyelmeztetés
DRIFT_SLOW_M                = 2.0   # ennél nagyobb: lassít, amíg vissza nem ér
DRIFT_RECOVER_M             = 0.7   # ez alatt újra normál sebesség
DRIFT_SLOW_FACTOR           = 0.3
WIND_WARN_MPS               = 0.6   # ekkora tartós ellentartás felett: "erős oldalszél"
HOVER_DRIFT_WARN_M          = 1.5   # lebegés közben ekkora elmozdulás: figyelmeztetés

# --- Visszatérés a felszállási ponthoz (alacsony akkumulátornál) ----------------
# A drón a bejárt úton (a megtanult pöttyláncon) megy vissza, mert ott végig lát
# pöttyöt, így a pozíciója pontos marad. Két feltétel közül az első indítja:
#   * az akkumulátor RTH_BATTERY_PCT alá esik, vagy
#   * a mért fogyasztás alapján már csak a hazaútra elég (+ BATTERY_FAILSAFE_PCT + RTH_RESERVE_PCT).
# A kritikus szint (BATTERY_FAILSAFE_PCT) alatt továbbra is azonnal leszáll, ahol van.
RTH_ENABLED            = True
RTH_BATTERY_PCT        = 35
RTH_RESERVE_PCT        = 5
RTH_SPEED_MPS          = 2.0
RTH_BREADCRUMB_M       = 3.0    # a bejárt út feljegyzése legalább ilyen sűrűn (a pöttyök mellett)
RTH_WAYPOINT_RADIUS_M  = 1.0    # ennyire kell megközelíteni egy útpontot
RTH_HOME_RADIUS_M      = 0.5    # a felszállási pontot ennyire
RTH_PATH_SIMPLIFY_M    = 1.5    # a hazaút kisimítása: ennél kisebb oldalirányú hullámzás nem számít kanyarnak
RTH_SEGMENT_END_M      = 0.5    # a hazaút egy szakaszának vége (kanyar) ennyire megközelítve jön a következő
RTH_TIME_MARGIN        = 1.25   # a hazaút idejének becslését ennyivel szorozza (biztonsági ráhagyás)
# TESZT: ha szám, ennyi másodperc pöttykövetés után akkumulátortól függetlenül visszafordul.
RTH_TEST_AFTER_S       = None

# --- Pötty-felismerés sebessége ------------------------------------------------
# A felismerés kicsinyített képen fut (0.5 -> 640x360), az eredményt visszaszámolja
# a teljes felbontásra. 10 m-ről a 0,6 m-es pötty így is ~24 px átmérőjű.
DETECT_SCALE = 0.5
# CLAHE fényerő-kiegyenlítés a felismerés előtt (valós, változó fénynél hasznos).
DETECT_CLAHE = True

# A felszállási pont ismert: innen indul a pozícióbecslés (a lefelé néző kamera
# a földön állva nem lát pöttyöt).
USE_KNOWN_START     = True
START_POSITION_NED  = [0.0, 0.0, 0.0]


# --- EKF origin GPS nélküli repüléshez -------------------------------------------
# Az ArduPilot GPS nélkül csak akkor enged élesíteni, ha megkapja a kiindulási koordinátát.
# A pontos érték mindegy (a navigáció relatív), csak legyen beállítva. A szimulált
# pálya koordinátái (sim/gazebo/worlds/nav_field.sdf):
SET_EKF_ORIGIN   = True
EKF_ORIGIN_LAT   = -35.363262
EKF_ORIGIN_LON   = 149.165237
EKF_ORIGIN_ALT_M = 584.0
