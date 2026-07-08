"""
===============================================================================
CanMV K230 智能人脸识别门禁/考勤系统 - 主程序
===============================================================================
"""

import os
import sys
import time
import gc
import ulab.numpy as np
from media.media import *
from media.sensor import *
from media.display import *
from machine import RTC

try:
    # FIX: import cp.xxx 需要父目录 /sdcard 在 sys.path；同时保留 /sdcard/cp 兼容直接 import。
    if "/sdcard" not in sys.path:
        sys.path.append("/sdcard")
    if "/sdcard/cp" not in sys.path:
        sys.path.append("/sdcard/cp")
except Exception:
    pass

try:
    import cp.recognition as recognition
except ImportError as e:
    raise ImportError("无法导入 recognition 模块: " + str(e))

try:
    import cp.registration as registration
except ImportError as e:
    raise ImportError("无法导入 registration 模块: " + str(e))

try:
    import cp.deletion as deletion
except ImportError as e:
    raise ImportError("无法导入 deletion 模块: " + str(e))

try:
    import cp.liveness as liveness
except ImportError as e:
    print("[警告] 活体检测模块导入失败，将禁止开门: " + str(e))
    liveness = None

try:
    import cp.hardware as hardware
except ImportError as e:
    print("[警告] 硬件控制模块导入失败，将禁用门锁/蜂鸣器/LED: " + str(e))
    hardware = None

try:
    import cp.iot as iot
except ImportError as e:
    print("[警告] IoT 日志模块导入失败，仅保留 SD 卡日志: " + str(e))
    iot = None

try:
    import cp.event_control as event_control
except ImportError as e:
    print("[警告] 事件控制模块导入失败，退化为主循环门控: " + str(e))
    event_control = None

try:
    import cp.gpio as gpio
except ImportError as e:
    print("[警告] GPIO/UART 模块导入失败: " + str(e))
    gpio = None


# ---------------------------------------------------------------------------
# 系统配置
# ---------------------------------------------------------------------------
CAM_RES_AI = (1280, 720)
CAM_RES_DISPLAY = (512, 320)
DISPLAY_W = CAM_RES_DISPLAY[0]
DISPLAY_H = CAM_RES_DISPLAY[1]

FACE_DET_MODEL = "/sdcard/examples/kmodel/face_detection_320.kmodel"
FACE_REG_MODEL = "/sdcard/examples/kmodel/face_recognition.kmodel"
FACE_LANDMARK_MODEL = "/sdcard/examples/kmodel/face_landmark.kmodel"
ANCHORS_PATH = "/sdcard/examples/utils/prior_data_320.bin"

DATABASE_DIR = "/sdcard/examples/utils/db/"
DATABASE_IMG_DIR = "/sdcard/examples/utils/db_img/"
ACCESS_LOG_PATH = "/sdcard/examples/utils/access_log.csv"
IOT_LOG_PATH = "/sdcard/examples/utils/iot_pending.log"
IOT_UPLOAD_ENABLED = False          # 有 Wi-Fi/服务器时改 True；默认不阻塞比赛现场演示
IOT_SERVER_URL = "http://38.182.122.194:8000/api/logs"
IOT_UPLOAD_INTERVAL_SEC = 30

FACE_DET_INPUT_SIZE = [320, 320]
FACE_REG_INPUT_SIZE = [112, 112]
FACE_LANDMARK_INPUT_SIZE = [192, 192]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2
ANCHOR_LEN = 4200
DET_DIM = 4
FACE_RECOGNITION_THRESHOLD = 0.75

GC_INTERVAL_MS = 1500  # FIX: 从3秒缩短到1.5秒，更频繁回收内存
WATCHDOG_TIMEOUT_MS = 5000  # FIX: 从10秒缩短到5秒，更快检测卡死
STRANGER_COOLDOWN_MS = 5000
DOOR_OPEN_COOLDOWN_MS = 8000
LIVENESS_TIMEOUT_SEC = 6
REQUIRED_BLINKS = 1
REGISTER_CAPTURE_COUNT = 30        # 每个用户默认采集 30 帧有效特征；可改 50，但注册时间会更长
REGISTER_CAPTURE_MAX_ATTEMPTS = 220 # 光照/姿态不好时给足重试机会
REGISTER_MIN_VALID_SAMPLES = 24       # 少于 24 帧有效特征不入库，保证注册质量
REGISTER_SAVE_SAMPLE_IMAGES = True   # 同步保存 30 张样本照片到 db_img/<编号>/，便于答辩展示
MAX_REGISTER_FACES = 50              # 竞赛要求支持 10-50 张/人脸库容量，本版上限按 50 人控制

# 竞赛安全策略：活体检测启用且要求 106 点模型；模型缺失时不允许开门。
LIVENESS_ENABLED = True
LIVENESS_STRICT = True
LIVENESS_LOG_LEVEL = 0  # 精简日志：仅保留启动、通过、超时关键日志，减少串口输出帧耗时

# 活体动作策略：
#   "any"    : 眨眼 / 上下点头 / 张嘴，任一动作通过，适合现场演示。
#   "random" : 每轮随机要求一个动作，更强抗预录视频，比赛答辩时可切换。
LIVENESS_ACTION_MODE = "random"
LIVENESS_ACTIONS = ("blink", "nod", "mouth")

MODE_RECOGNITION = 0
MODE_REGISTRATION = 1
MODE_DELETION = 2
MODE_PASSWORD_VERIFY = 3  # 密码验证模式：A/B键触发注册/删除前的密码校验

# 管理员密码配置
ADMIN_PASSWORD = "123456"

PIN_BUZZER = 27
PIN_RELAY = 63
PIN_LED = 61

# 全局事件控制器：safe_buzzer_beep/NonBlockingDoor 会优先通过它投递事件。
EVENT_CTRL = None


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def ticks_ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.time() * 1000)


def ticks_diff(a, b):
    return time.ticks_diff(a, b) if hasattr(time, "ticks_diff") else a - b


def sleep_ms(ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


def safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return 0


def check_file_exists(path, description, required=False):
    """检查文件是否存在，兼容 CanMV os.listdir。"""
    try:
        if "/" in path:
            parts = path.split("/")
            file_name = parts[-1]
            dir_path = "/".join(parts[:-1])
            if not dir_path:
                dir_path = "/"
        else:
            dir_path = "/"
            file_name = path
        ok = file_name in os.listdir(dir_path)
        if not ok:
            level = "错误" if required else "警告"
            print("[{}] {} 不存在: {}".format(level, description, path))
        return ok
    except Exception as e:
        level = "错误" if required else "警告"
        print("[{}] 无法检查 {}: {}".format(level, description, e))
        return False


def ensure_database_dirs():
    for d in [DATABASE_DIR, DATABASE_IMG_DIR]:
        try:
            os.mkdir(d)
        except Exception:
            pass


def validate_environment():
    errors = []
    if not check_file_exists(FACE_DET_MODEL, "人脸检测模型", required=True):
        errors.append(FACE_DET_MODEL)
    if not check_file_exists(FACE_REG_MODEL, "人脸识别模型", required=True):
        errors.append(FACE_REG_MODEL)
    if not check_file_exists(ANCHORS_PATH, "Anchor 文件", required=True):
        errors.append(ANCHORS_PATH)
    if LIVENESS_ENABLED and LIVENESS_STRICT:
        if not check_file_exists(FACE_LANDMARK_MODEL, "106点人脸关键点模型", required=True):
            errors.append(FACE_LANDMARK_MODEL)

    if errors:
        print("[自检] 以下关键文件缺失:")
        for e in errors:
            print("  - {}".format(e))
        return False
    print("[自检] 关键文件检查通过")
    return True


def load_anchors(anchors_path):
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((ANCHOR_LEN, DET_DIM))
    return anchors


def current_time_string():
    """生成日志时间。RTC 未校准时仍能给出单调 ticks，避免日志为空。"""
    try:
        t = time.localtime()
        return "%04d-%02d-%02d %02d:%02d:%02d" % (t[0], t[1], t[2], t[3], t[4], t[5])
    except Exception:
        return "ticks_%d" % ticks_ms()


def append_access_log(event_type, name="", score=0.0, message=""):
    """写 SD 卡考勤/事件日志，满足竞赛要求的本地记录能力。"""
    try:
        line = "%s,%s,%s,%.3f,%s\n" % (current_time_string(), event_type, str(name), float(score), str(message))
    except Exception:
        line = "%s,%s,%s,%s,%s\n" % (current_time_string(), event_type, str(name), str(score), str(message))
    try:
        # 首次创建时写表头，便于 Web/PC 查看。
        need_header = False
        try:
            stat = os.stat(ACCESS_LOG_PATH)
            need_header = stat[6] == 0
        except Exception:
            need_header = True
        with open(ACCESS_LOG_PATH, "a") as f:
            if need_header:
                f.write("time,event,name,score,message\n")
            f.write(line)
        print("[日志] " + line.strip())
        return True
    except Exception as e:
        print("[日志] 写入失败: {}".format(e))
        return False


def append_iot_pending(event_type, name="", score=0.0, message=""):
    if iot is None:
        return False
    try:
        line = "%s | %s | %s | %.3f | %s" % (current_time_string(), event_type, str(name), float(score), str(message))
        with open(IOT_LOG_PATH, "a") as f:
            f.write(line + "\n")
        return True
    except Exception:
        return False


# 全局 UART 日志发送器
UART_LOGGER = None

def uart_send(data_dict):
    """通过 UART 发送 JSON 数据，带硬件级别验证"""
    if UART_LOGGER is None or UART_LOGGER.uart is None:
        return
    try:
        import ujson
        json_str = ujson.dumps(data_dict)
        data = json_str + "\n"
        written = UART_LOGGER.uart.write(data)
        if written == len(data):
            # 验证数据是否真正发送完成（部分固件支持 txdone）
            if hasattr(UART_LOGGER.uart, 'txdone'):
                if UART_LOGGER.uart.txdone():
                    print("[UART] 发送成功: {}".format(json_str))
                else:
                    print("[UART] 发送中(未完成): {}".format(json_str))
            else:
                print("[UART] 发送成功: {}".format(json_str))
        else:
            print("[UART] 发送不完整: 期望{}字节 实际{}字节".format(len(data), written))
    except Exception as e:
        print("[UART] 发送失败: {}".format(e))

def log_event(event_type, name="", score=0.0, message=""):
    append_access_log(event_type, name, score, message)
    append_iot_pending(event_type, name, score, message)
    # 通过 UART 发送 JSON 日志
    uart_send({
        "device_id": "1",
        "device_name": "office_door_001",
        "type": "event",
        "event": event_type,
        "name": str(name),
        "score": float(score),
        "message": str(message),
        "time": current_time_string()
    })


def show_system_message(sensor, text, color=(255, 255, 255), hold_ms=500):
    """同步屏幕提示和终端输出，注册/删除结束时避免用户以为卡死。"""
    print(text)
    try:
        img_display = sensor.snapshot(chn=1)
        img_display.draw_rectangle(0, 0, DISPLAY_W, DISPLAY_H, color=(0, 0, 0), thickness=-1)
        img_display.draw_string_advanced(10, DISPLAY_H // 2 - 20, 22, text, color=color)
        Display.show_image(img_display)
        sleep_ms(hold_ms)
    except Exception:
        pass


def parse_recognition_result(recg_text):
    """返回 (is_known, name, score)。兼容 'name: xx, score:0.9' 与 'unknown'。"""
    text = str(recg_text)
    lower = text.lower()
    if "unknown" in lower:
        score = 0.0
        if "score:" in lower:
            try:
                score = float(text.split("score:")[1].strip())
            except Exception:
                score = 0.0
        return False, "unknown", score

    name = "User"
    score = 0.0
    try:
        if "name:" in text and "score:" in text:
            parts = text.split(",")
            name = parts[0].replace("name:", "").strip()
            score = float(parts[1].replace("score:", "").strip())
    except Exception:
        pass
    return True, name, score


def box_area(box):
    try:
        return abs(float(box[2]) * float(box[3]))
    except Exception:
        return 0.0


def select_best_known_face(det_boxes, recg_res):
    """选择本帧用于开门/活体的目标人脸。

    FIX: 原 main.py 在多脸场景下用循环最后一次 current_face_name，可能把 A 的识别
    结果和 B 的关键点混用。这里按分数优先、面积次之选择同一 index。
    """
    best = None
    unknown_count = 0
    count = safe_len(det_boxes)
    for i in range(count):
        recg_text = recg_res[i] if i < safe_len(recg_res) else "unknown"
        is_known, name, score = parse_recognition_result(recg_text)
        if not is_known:
            unknown_count += 1
            continue
        candidate = {
            "index": i,
            "name": name,
            "score": score,
            "box": det_boxes[i],
            "area": box_area(det_boxes[i])
        }
        if best is None:
            best = candidate
        elif candidate["score"] > best["score"]:
            best = candidate
        elif candidate["score"] == best["score"] and candidate["area"] > best["area"]:
            best = candidate
    return best, unknown_count


def init_sensor_display():
    sensor = Sensor()
    sensor.reset()
    sleep_ms(200)

    sensor.set_framesize(width=CAM_RES_AI[0], height=CAM_RES_AI[1], chn=0)
    sensor.set_pixformat(Sensor.RGB565, chn=0)
    sensor.set_framesize(width=DISPLAY_W, height=DISPLAY_H, chn=1)
    sensor.set_pixformat(Sensor.RGB565, chn=1)
    sensor.set_hmirror(True)
    sleep_ms(100)

    Display.init(Display.NT35516, DISPLAY_W, DISPLAY_H, 0, 0)
    sleep_ms(100)
    MediaManager.init()
    sleep_ms(100)
    sensor.run()
    sleep_ms(200)
    return sensor


def init_hardware():
    if hardware is None:
        print("[硬件] 硬件模块未加载，跳过初始化")
        return None, None, None
    try:
        door_ctrl = hardware.DoorAccessController(
            relay_pin=PIN_RELAY,
            buzzer_pin=PIN_BUZZER,
            open_duration=3
        )
        led = hardware.StatusLED(PIN_LED)
        uart_logger = None
        if gpio is not None and hasattr(gpio, 'UARTLogger'):
            uart_logger = gpio.UARTLogger(baudrate=115200)
        return door_ctrl, led, uart_logger
    except Exception as e:
        print("[警告] 硬件初始化失败: {}".format(e))
        return None, None, None


def create_recognition_instance(anchors):
    return recognition.create_face_recognition(
        FACE_DET_MODEL, FACE_REG_MODEL,
        FACE_DET_INPUT_SIZE, FACE_REG_INPUT_SIZE,
        DATABASE_DIR, anchors,
        CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
        FACE_RECOGNITION_THRESHOLD,
        CAM_RES_AI, CAM_RES_DISPLAY,
        use_kpu=True,
        face_landmark_kmodel_path=FACE_LANDMARK_MODEL,
        landmark_input_size=FACE_LANDMARK_INPUT_SIZE,
        enable_liveness_landmark=(LIVENESS_ENABLED and liveness is not None)
    )


def create_liveness_detector():
    if liveness is None or not LIVENESS_ENABLED:
        return None
    return liveness.LivenessDetector(
        required_blinks=REQUIRED_BLINKS,
        timeout_sec=LIVENESS_TIMEOUT_SEC,
        action_mode=LIVENESS_ACTION_MODE,
        allowed_actions=LIVENESS_ACTIONS,

        # 更敏感的眨眼参数：
        # 1) min_closed_ms=0：低帧率下只抓到闭眼瞬间也能通过；
        # 2) accept_blink_on_close=True：检测到闭眼下沿即计数，不必等下一帧睁眼；
        # 3) close_ratio/open_ratio 使用个人睁眼基线，适配小眼睛、眼镜、不同距离。
        min_closed_frames=1,
        min_open_frames=1,
        min_closed_ms=0,
        max_closed_ms=1200,
        close_ear_threshold=0.23,
        open_ear_threshold=0.25,
        close_ratio=0.82,
        open_ratio=0.90,
        blink_min_drop=0.025,
        accept_blink_on_close=True,

        # 头部动作参数：小幅左右摇头/上下点头即可触发，适合 K230 实时帧率。
        shake_threshold=0.10,
        shake_range_threshold=0.18,
        nod_threshold=0.10,
        nod_range_threshold=0.16,
        action_window_ms=2500,
        action_min_frames=2,

        # 允许头部动作，不再把正常摇头误判为“动作过大”。
        max_jump_ratio=0.60,
        log_level=LIVENESS_LOG_LEVEL,
        log_interval_ms=350
    )


# ---------------------------------------------------------------------------
# 按键、看门狗、FPS
# ---------------------------------------------------------------------------

class KeypadManager:
    _instance = None

    @classmethod
    def get_keypad(cls):
        if cls._instance is None:
            try:
                cls._instance = registration.MatrixKeypad()
            except Exception as e:
                print("[按键] 初始化失败: {}".format(e))
                return None
        return cls._instance

    @classmethod
    def scan(cls):
        kp = cls.get_keypad()
        if kp is None:
            return None
        try:
            return kp.scan()
        except Exception:
            return None


class SystemWatchdog:
    def __init__(self, timeout_ms=WATCHDOG_TIMEOUT_MS):
        self.timeout_ms = timeout_ms
        self.last_feed = ticks_ms()
        self.error_count = 0
        self.max_errors = 5

    def feed(self):
        self.last_feed = ticks_ms()

    def check(self):
        return ticks_diff(ticks_ms(), self.last_feed) > self.timeout_ms

    def record_error(self):
        self.error_count += 1
        return self.error_count < self.max_errors


class FrameRateMonitor:
    def __init__(self, window_size=30):
        self.frame_times = []
        self.window_size = window_size
        self.last_frame_time = ticks_ms()

    def tick(self):
        now = ticks_ms()
        elapsed = ticks_diff(now, self.last_frame_time)
        self.last_frame_time = now
        self.frame_times.append(elapsed)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        if len(self.frame_times) == 0:
            return 0, 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        return fps, elapsed


class NonBlockingDoor:
    """非阻塞门控代理。

    优先把开门命令投递给 EventControlProcess，保证 AI/显示主循环不直接执行
    GPIO、蜂鸣器和日志写入；事件线程不可用时自动退回 door_ctrl.update()。
    """

    def __init__(self, door_ctrl, open_duration_sec=3, event_ctrl=None):
        self.door_ctrl = door_ctrl
        self.open_duration_sec = open_duration_sec
        self.event_ctrl = event_ctrl

    def request_open(self, name="", score=0.0, extra=""):
        if self.event_ctrl is not None:
            return self.event_ctrl.request_open(name=name, score=score, extra=extra)
        if self.door_ctrl is None:
            return False
        try:
            if hasattr(self.door_ctrl, "request_open"):
                return self.door_ctrl.request_open(with_beep=True)
        except Exception:
            pass
        return False

    def update(self):
        try:
            if self.event_ctrl is not None:
                # 事件线程已启动时由线程推进；否则主循环协作推进。
                if not self.event_ctrl.is_thread_started():
                    self.event_ctrl.update()
                return
            if self.door_ctrl is not None and hasattr(self.door_ctrl, "update"):
                self.door_ctrl.update()
        except Exception:
            pass

    def is_open(self):
        try:
            if self.event_ctrl is not None:
                return self.event_ctrl.is_door_open()
            if self.door_ctrl is not None:
                return self.door_ctrl.is_door_open()
        except Exception:
            pass
        return False

    def cancel(self):
        try:
            if self.event_ctrl is not None:
                self.event_ctrl.stop()
            if self.door_ctrl is not None:
                if hasattr(self.door_ctrl, "close_door"):
                    self.door_ctrl.close_door()
                elif hasattr(self.door_ctrl, "relay") and self.door_ctrl.relay is not None:
                    self.door_ctrl.relay.lock()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# UI 与提示
# ---------------------------------------------------------------------------

def draw_ui_header(img_display, mode, fps, total_time, nb_door, face_count):
    try:
        img_display.draw_rectangle(0, 0, DISPLAY_W, 33, color=(30, 30, 30), thickness=-1)
        mode_names = {MODE_RECOGNITION: "识别", MODE_REGISTRATION: "注册", MODE_DELETION: "删除", MODE_PASSWORD_VERIFY: "验证"}
        img_display.draw_string_advanced(2, 0, 24, "[{}]".format(mode_names.get(mode, "?")), color=(0, 255, 0))
        img_display.draw_string_advanced(DISPLAY_W // 6, 0, 24,
                                         "FPS:{:.0f} {}ms".format(fps, total_time), color=(255, 255, 0))
        is_open = nb_door.is_open() if nb_door else False
        img_display.draw_string_advanced(DISPLAY_W // 2, 0, 24,
                                         "门:开" if is_open else "门:关",
                                         color=(255, 0, 0) if is_open else (0, 255, 0))
        img_display.draw_string_advanced(DISPLAY_W * 2 // 3, 0, 24,
                                         "库:{}".format(face_count), color=(255, 255, 255))
        img_display.draw_string_advanced(DISPLAY_W * 5 // 6, 0, 24, "A/B/C/D", color=(128, 128, 128))
    except Exception:
        pass


def draw_guide_text(img_display, text, y=None, color=(255, 255, 255)):
    try:
        if y is None:
            y = DISPLAY_H - 40
        img_display.draw_rectangle(0, DISPLAY_H - 45, DISPLAY_W, 45, color=(0, 0, 0), thickness=-1)
        img_display.draw_string_advanced(10, y, 18, text, color=color)
    except Exception:
        pass


def draw_password_screen(img_display, password_input, target_action):
    """绘制密码验证界面，覆盖整个屏幕。"""
    try:
        img_display.draw_rectangle(0, 0, DISPLAY_W, DISPLAY_H, color=(0, 0, 0), thickness=-1)
        # 标题
        action_name = "注册" if target_action == "register" else "删除"
        title = "[管理员验证]"
        img_display.draw_string_advanced(DISPLAY_W // 2 - 80, DISPLAY_H // 2 - 60, 24, title, color=(0, 255, 255))
        # 提示文字
        img_display.draw_string_advanced(DISPLAY_W // 2 - 100, DISPLAY_H // 2 - 20, 18,
                                         "请输入密码进行{}操作".format(action_name), color=(255, 255, 255))
        # 密码输入框（用*号显示已输入位数）
        masked = "*" * len(password_input)
        box_x = DISPLAY_W // 2 - 80
        box_y = DISPLAY_H // 2 + 10
        box_w = 160
        box_h = 36
        img_display.draw_rectangle(box_x, box_y, box_w, box_h, color=(50, 50, 50), thickness=-1)
        img_display.draw_rectangle(box_x, box_y, box_w, box_h, color=(0, 255, 0), thickness=2)
        img_display.draw_string_advanced(box_x + 10, box_y + 6, 22, masked if masked else "_", color=(0, 255, 0))
        # 底部提示
        img_display.draw_string_advanced(DISPLAY_W // 2 - 120, DISPLAY_H // 2 + 60, 16,
                                         "数字输入 D/#确认 C取消", color=(128, 128, 128))
    except Exception:
        pass


def draw_face_box(img_display, det_box, name, score, color=(0, 255, 0)):
    try:
        x1 = int(round(det_box[0], 0)) * DISPLAY_W // CAM_RES_AI[0]
        y1 = int(round(det_box[1], 0)) * DISPLAY_H // CAM_RES_AI[1]
        w = int(round(det_box[2], 0)) * DISPLAY_W // CAM_RES_AI[0]
        h = int(round(det_box[3], 0)) * DISPLAY_H // CAM_RES_AI[1]
        x1 = max(0, x1)
        y1 = max(22, y1)
        w = max(1, min(w, DISPLAY_W - x1))
        h = max(1, min(h, DISPLAY_H - y1))
        img_display.draw_rectangle(x1, y1, w, h, color=color, thickness=3)
        label_w = min(max(w, 120), DISPLAY_W - x1)
        label_y = max(22, y1 - 30)
        img_display.draw_rectangle(x1, label_y, label_w, 30, color=color, thickness=-1)
        label = "{} {:.2f}".format(name, score) if score else name
        img_display.draw_string_advanced(x1 + 2, label_y + 2, 24, label, color=(255, 255, 255))
    except Exception:
        pass


def safe_buzzer_beep(door_ctrl, beep_type="short"):
    global EVENT_CTRL
    if EVENT_CTRL is not None:
        try:
            EVENT_CTRL.beep(beep_type)
            return
        except Exception:
            pass
    if door_ctrl is None or getattr(door_ctrl, "buzzer", None) is None:
        return
    try:
        bz = door_ctrl.buzzer
        if beep_type == "success":
            bz.beep_success()
        elif beep_type == "stranger":
            bz.beep_stranger()
        elif beep_type == "warning":
            bz.beep_warning()
        elif beep_type == "door":
            bz.beep_door_open()
        else:
            if hasattr(bz, "start_pattern"):
                bz.start_pattern([(1, 50), (0, 1)])
            else:
                bz.beep(50)
    except Exception:
        pass


def update_led(status_led, nb_door, liveness_required, has_unknown_face):
    if status_led is None:
        return
    try:
        if nb_door is not None and nb_door.is_open():
            status_led.on()
        elif has_unknown_face:
            status_led.blink(120)
        elif liveness_required:
            status_led.blink(250)
        else:
            status_led.blink(800)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    global EVENT_CTRL
    print("=" * 56)
    print("  智能人脸识别门禁/考勤系统 V4.0 - 竞赛最终版")
    print("  平台: CanMV K230 / RT-Thread Smart / MicroPython")
    print("=" * 56)

    if not validate_environment():
        print("[致命] 关键文件缺失，系统安全退出。请检查 SD 卡 /sdcard/examples 资源。")
        return

    try:
        RTC()  # 不再强制写入固定日期，避免日志时间被覆盖。
    except Exception:
        pass

    ensure_database_dirs()

    print("[启动] 初始化摄像头与显示...")
    sensor = None
    for attempt in range(3):
        try:
            sensor = init_sensor_display()
            break
        except Exception as e:
            print("[错误] 摄像头初始化失败 (尝试 {}/3): {}".format(attempt + 1, e))
            sleep_ms(1000)
    if sensor is None:
        print("[致命] 摄像头初始化失败")
        return
    print("[启动] 摄像头就绪")

    print("[启动] 初始化门锁/蜂鸣器/LED/UART...")
    door_ctrl, status_led, uart_logger = init_hardware()
    global UART_LOGGER
    UART_LOGGER = uart_logger
    event_ctrl = None
    if event_control is not None:
        try:
            event_ctrl = event_control.EventControlProcess(door_ctrl, log_dir="/sdcard/cp/logs")
            event_ctrl.start()
            EVENT_CTRL = event_ctrl
        except Exception as e:
            print("[事件] 事件控制初始化失败，使用主循环门控: {}".format(e))
            event_ctrl = None
            EVENT_CTRL = None
    nb_door = NonBlockingDoor(door_ctrl, open_duration_sec=3, event_ctrl=event_ctrl)
    print("[启动] 硬件/事件控制就绪")

    print("[启动] 加载 anchors...")
    try:
        anchors = load_anchors(ANCHORS_PATH)
    except Exception as e:
        print("[致命] 加载 anchors 失败: {}".format(e))
        return

    print("[启动] 加载人脸识别/活体模型...")
    try:
        fr = create_recognition_instance(anchors)
    except Exception as e:
        print("[致命] 人脸识别模型加载失败: {}".format(e))
        return

    liveness_detector = create_liveness_detector()
    if LIVENESS_ENABLED and liveness_detector is None:
        print("[活体] 模块不可用，系统将禁止开门")

    log_uploader = None
    if IOT_UPLOAD_ENABLED and iot is not None:
        try:
            log_uploader = iot.LogUploader(IOT_SERVER_URL, IOT_LOG_PATH, upload_interval=IOT_UPLOAD_INTERVAL_SEC)
            log_uploader.start()
        except Exception as e:
            print("[IoT] 上传器初始化失败: {}".format(e)) 
    watchdog = SystemWatchdog()
    fps_monitor = FrameRateMonitor()

    current_mode = MODE_RECOGNITION
    last_gc_time = ticks_ms()
    last_stranger_alert = 0
    last_door_open = 0
    last_key_debounce = 0
    key_debounce_ms = 500
    stranger_frame_count = 0
    stranger_alert_threshold = 3
   
    # 密码验证状态 
    password_input = ""  
    password_error_msg = ""   

    liveness_required = False
    liveness_passed = False
    liveness_target_name = ""
    liveness_target_index = -1
    liveness_start_time = 0
    liveness_fail_beep_time = 0

    # 上层点头快速判定：双阈值 + 峰值追踪，响应速度对齐眨眼/张嘴
    # 单帧range≥0.18 或 连续2帧range≥0.14 即通过
    NOD_UPPER_SINGLE_THRESHOLD = 0.18
    NOD_UPPER_CONTINUOUS_THRESHOLD = 0.14
    NOD_UPPER_CONTINUOUS_FRAMES = 2
    nod_upper_consecutive_count = 0
    nod_upper_peak = 0.0  # 峰值追踪

    last_door_state = False
    door_notice_until = 0
    door_notice_text = ""
    last_ui_line = ""

    def set_ui_message(text, color=(255, 255, 255), hold_ms=0):
        """同步写终端和屏幕底部提示，避免注册/删除等阻塞流程让 UI 看起来卡住。"""
        try:
            print("[UI] " + str(text))
        except Exception:
            pass
        try:
            img_tip = sensor.snapshot(chn=1)
            draw_guide_text(img_tip, str(text), color=color)
            Display.show_image(img_tip)
            if hold_ms and hold_ms > 0:
                sleep_ms(hold_ms)
        except Exception:
            pass

    def registration_ui_callback(text):
        set_ui_message(text, color=(0, 255, 0))

    def deletion_ui_callback(text):
        set_ui_message(text, color=(255, 255, 0))

    frame_count = 0
    print("[系统] 进入人脸识别模式")
    log_event("system_start", "", 0.0, "进入人脸识别模式")
    safe_buzzer_beep(door_ctrl, "success")

    # FIX: 添加帧级超时检测，防止单次循环卡死超过阈值
    FRAME_TIMEOUT_MS = 5000  # 单帧超过5秒视为卡死
    last_frame_start_ms = ticks_ms()
    consecutive_timeout_count = 0
    max_consecutive_timeouts = 3  # 连续3次超时则触发恢复

    while True:
        try:
            # FIX: 记录帧开始时间，用于超时检测
            frame_start_ms = ticks_ms()
            frame_elapsed = ticks_diff(frame_start_ms, last_frame_start_ms)
            
            # 检测上一帧是否超时（说明卡在了推理或IO中）
            if frame_elapsed > FRAME_TIMEOUT_MS:
                consecutive_timeout_count += 1
                print("[看门狗] 帧超时: {}ms (连续{}次)".format(frame_elapsed, consecutive_timeout_count))
                if consecutive_timeout_count >= max_consecutive_timeouts:
                    print("[看门狗] 连续超时过多，尝试紧急恢复")
                    gc.collect()
                    if liveness_detector:
                        liveness_detector.reset()
                    consecutive_timeout_count = 0
            else:
                consecutive_timeout_count = 0
            
            last_frame_start_ms = frame_start_ms
            
            # 正常喂狗（放在循环开头，确保能执行到）
            watchdog.feed()
            nb_door.update()


            # 门状态变化由主循环统一打印、记录、提示，保证终端和 UI 一致。
            door_open_now = nb_door.is_open() if nb_door is not None else False
            if door_open_now != last_door_state:
                last_door_state = door_open_now
                if door_open_now:
                    door_notice_text = "[门禁] 门已打开，请通过"
                    door_notice_until = ticks_ms() + 1200
                    uart_send({
                        "device_id": "1",
                        "device_name": "office_door_001",
                        "type": "door",
                        "event": "open",
                        "name": str(liveness_target_name),
                        "time": current_time_string()
                    })
                else:
                    door_notice_text = "[门禁] 门已关闭，返回识别"
                    door_notice_until = ticks_ms() + 1800
                    liveness_passed = False
                    if liveness_detector:
                        liveness_detector.reset()
                    log_event("door_close", liveness_target_name, 0.0, "继电器已关门")
                    print(door_notice_text)

            # ------------------------ 按键处理 ------------------------
            key = KeypadManager.scan()
            if key:
                now_key = ticks_ms()
                if ticks_diff(now_key, last_key_debounce) > key_debounce_ms:
                    last_key_debounce = now_key
                    watchdog.feed()
                    if key == "A":
                        # 密码验证：注册操作前需要验证管理员密码
                        print("\n[密码] 请输入管理员密码进行注册操作")
                        password_target_action = "register"
                        password_input = ""
                        password_error_msg = ""
                        current_mode = MODE_PASSWORD_VERIFY

                    elif key == "B":
                        # 密码验证：删除操作前需要验证管理员密码
                        print("\n[密码] 请输入管理员密码进行删除操作")
                        password_target_action = "delete"
                        password_input = ""
                        password_error_msg = ""
                        current_mode = MODE_PASSWORD_VERIFY

                    elif key == "C":
                        current_mode = MODE_RECOGNITION
                        set_ui_message("[识别] 返回人脸识别模式", color=(0, 255, 0))
                        if liveness_detector:
                            liveness_detector.reset()
                        liveness_required = False
                        liveness_passed = False
                        show_system_message(sensor, "[模式] 返回人脸识别模式", color=(255, 255, 255), hold_ms=400)

                    elif key == "D":
                        face_count = len(fr.db_name) if hasattr(fr, "db_name") else 0
                        try:
                            mem_free = gc.mem_free()
                        except Exception:
                            mem_free = 0
                        live_ok = fr.has_liveness_landmark() if hasattr(fr, "has_liveness_landmark") else False
                        print("[系统] 已注册:{}人, 活体模型:{}, 内存:{} bytes".format(face_count, live_ok, mem_free))
                        set_ui_message("[系统] 库:{}人 活体:{} 内存:{}".format(face_count, "OK" if live_ok else "NO", mem_free), color=(255, 255, 0))
                        safe_buzzer_beep(door_ctrl, "short")

            # ------------------------ 密码验证模式 ------------------------
            if current_mode == MODE_PASSWORD_VERIFY:
                # 密码验证模式下，正常采集图像但只绘制密码界面
                frame_start = ticks_ms()
                frame_count += 1
                img = sensor.snapshot(chn=0)
                img_display = sensor.snapshot(chn=1)
                draw_password_screen(img_display, password_input, password_target_action)
                Display.show_image(img_display)
                frame_end = ticks_ms()
                frame_total = ticks_diff(frame_end, frame_start)
                fps, _ = fps_monitor.tick()
                face_count = len(fr.db_name) if hasattr(fr, "db_name") else 0
                # 修复：根据目标动作显示对应状态文字，而非硬编码"识别"
                header_mode = MODE_REGISTRATION if password_target_action == "register" else MODE_DELETION
                draw_ui_header(img_display, header_mode, fps, frame_total, nb_door, face_count)
                Display.show_image(img_display)
                last_frame_start_ms = frame_start
                consecutive_timeout_count = 0

                # 密码模式下的按键处理（独立于上面的按键处理）
                key = KeypadManager.scan()
                if key:
                    now_key = ticks_ms()
                    if ticks_diff(now_key, last_key_debounce) > key_debounce_ms:
                        last_key_debounce = now_key
                        if key == "C":
                            # 取消密码验证，返回识别
                            print("\n[密码] 取消验证，返回识别")
                            set_ui_message("[识别] 已取消，返回人脸识别", color=(0, 255, 0))
                            password_target_action = ""
                            password_input = ""
                            password_error_msg = ""
                            current_mode = MODE_RECOGNITION
                            if liveness_detector:
                                liveness_detector.reset()
                            liveness_required = False
                            liveness_passed = False
                        elif key == "D" or key == "#":
                            # 确认密码
                            if password_input == ADMIN_PASSWORD:
                                target = password_target_action  # 保存目标动作
                                print("\n[密码] 验证通过，执行{}操作".format(
                                    "注册" if target == "register" else "删除"))
                                set_ui_message("[密码] 验证通过", color=(0, 255, 0))
                                safe_buzzer_beep(door_ctrl, "success")
                                # 密码验证仅单次有效，清除状态
                                password_target_action = ""
                                password_input = ""
                                password_error_msg = ""
                                # 根据目标动作进入对应模式
                                if target == "register":
                                    current_mode = MODE_REGISTRATION
                                elif target == "delete":
                                    current_mode = MODE_DELETION
                                else:
                                    current_mode = MODE_RECOGNITION
                            else:
                                # 密码错误
                                print("\n[密码] 验证失败")
                                password_error_msg = "密码错误，请重试"
                                password_input = ""
                                safe_buzzer_beep(door_ctrl, "warning")
                                set_ui_message("[密码] 验证失败，请重试", color=(255, 0, 0))
                        elif key == "*":
                            # 退格
                            password_input = password_input[:-1]
                        else:
                            # 数字键输入
                            if len(password_input) < 6:
                                password_input += key

                continue  # 密码模式下不执行后续的识别逻辑

            # ------------------------ 注册模式 ------------------------
            if current_mode == MODE_REGISTRATION:
                if liveness_detector:
                    liveness_detector.reset()
                liveness_required = False
                liveness_passed = False

                face_count = len(fr.db_name) if hasattr(fr, "db_name") else 0
                if face_count >= MAX_REGISTER_FACES:
                    print("[注册] 人脸库已满 ({}人)，无法注册".format(MAX_REGISTER_FACES))
                    set_ui_message("[注册] 人脸库已满，返回识别", color=(255, 0, 0))
                    safe_buzzer_beep(door_ctrl, "warning")
                    current_mode = MODE_RECOGNITION
                else:
                    try:
                        fr.deinit()
                    except Exception:
                        pass
                    gc.collect()
                    reg_ok = False
                    try:
                        reg_ok = registration.do_face_registration(
                            sensor, DATABASE_DIR, DATABASE_IMG_DIR,
                            FACE_DET_MODEL, FACE_REG_MODEL, ANCHORS_PATH,
                            FACE_DET_INPUT_SIZE, FACE_REG_INPUT_SIZE,
                            CONFIDENCE_THRESHOLD, NMS_THRESHOLD,
                            keypad=KeypadManager.get_keypad(),
                            rgb888p_size=list(CAM_RES_AI),
                            display_size=list(CAM_RES_DISPLAY),
                            capture_count=REGISTER_CAPTURE_COUNT,
                            max_attempts=REGISTER_CAPTURE_MAX_ATTEMPTS,
                            min_valid_count=REGISTER_MIN_VALID_SAMPLES,
                            save_sample_images=REGISTER_SAVE_SAMPLE_IMAGES,
                            ui_callback=registration_ui_callback
                        )
                    except Exception as e:
                        print("[注册] 注册过程异常: {}".format(e))
                    log_event("register", "", 0.0, "success" if reg_ok else "failed_or_cancelled")
                    show_system_message(sensor, "[注册] 完成，返回识别" if reg_ok else "[注册] 未完成，返回识别",
                                        color=(0, 255, 0) if reg_ok else (255, 255, 0), hold_ms=800)
                    gc.collect()
                    try:
                        fr = create_recognition_instance(anchors)
                    except Exception as e:
                        print("[致命] 注册后重建识别模型失败: {}".format(e))
                        return
                    current_mode = MODE_RECOGNITION
                    show_system_message(sensor, "[模式] 返回人脸识别模式", color=(255, 255, 255), hold_ms=500)
                gc.collect()
                continue  # 注册完成后跳过本帧识别

            # ------------------------ 删除模式 ------------------------
            if current_mode == MODE_DELETION:
                face_list = fr.list_faces() if hasattr(fr, "list_faces") else []
                if len(face_list) == 0:
                    print("[删除] 数据库为空")
                    log_event("delete", "", 0.0, "database_empty")
                    safe_buzzer_beep(door_ctrl, "warning")
                    show_system_message(sensor, "[删除] 数据库为空，返回识别", color=(255, 255, 0), hold_ms=800)
                    current_mode = MODE_RECOGNITION
                else:
                    try:
                        delete_name = registration.get_filename_from_keypad(keypad=KeypadManager.get_keypad(), ui_callback=deletion_ui_callback)
                    except Exception:
                        delete_name = ""
                    del_ok = False
                    person_name = ""
                    if delete_name:
                        person_name = delete_name.split(".")[0]
                        if deletion.delete_face_by_name(DATABASE_DIR, person_name):
                            del_ok = True
                            print("[删除] 人脸 {} 删除成功".format(person_name))
                            set_ui_message("[删除] 删除成功，返回人脸识别", color=(0, 255, 0))
                            try:
                                fr.deinit()
                            except Exception:
                                pass
                            gc.collect()
                            try:
                                fr = create_recognition_instance(anchors)
                            except Exception as e:
                                print("[致命] 删除后重建识别模型失败: {}".format(e))
                                return
                        else:
                            print("[删除] 人脸 {} 删除失败".format(person_name))
                            safe_buzzer_beep(door_ctrl, "warning")
                    log_event("delete", person_name, 0.0, "success" if del_ok else "failed_or_cancelled")
                    show_system_message(sensor, "[删除] 完成，返回识别" if del_ok else "[删除] 未完成，返回识别",
                                        color=(0, 255, 0) if del_ok else (255, 255, 0), hold_ms=800)
                    current_mode = MODE_RECOGNITION
                gc.collect()
                continue  # 删除完成后跳过本帧识别

            # ------------------------ 图像采集与推理 ------------------------
            frame_start = ticks_ms()
            frame_count += 1

            img = sensor.snapshot(chn=0)
            img_display = sensor.snapshot(chn=1)

            format_start = ticks_ms()
            try:
                rgb888p_img_ndarry = fr.image2rgb888array(img)
            except Exception as e:
                print("[图像] 格式转换失败: {}".format(e))
                gc.collect()
                continue
            format_time = ticks_diff(ticks_ms(), format_start)

            infer_start = ticks_ms()
            try:
                det_boxes, recg_res = fr.run(rgb888p_img_ndarry)
            except Exception as e:
                print("[推理] 异常: {}".format(e))
                gc.collect()
                continue
            infer_time = ticks_diff(ticks_ms(), infer_start)

            # ------------------------ 结果解析与绘制 ------------------------
            draw_start = ticks_ms()
            face_count = len(fr.db_name) if hasattr(fr, "db_name") else 0
            det_count = safe_len(det_boxes)
            target, unknown_count = select_best_known_face(det_boxes, recg_res)
            has_known_face = target is not None
            has_unknown_face = unknown_count > 0

            if det_count > 0:
                for i in range(det_count):
                    recg_text = recg_res[i] if i < safe_len(recg_res) else "unknown"
                    is_known, name, score = parse_recognition_result(recg_text)
                    if is_known:
                        draw_face_box(img_display, det_boxes[i], name, score, color=(0, 255, 0))
                    else:
                        draw_face_box(img_display, det_boxes[i], "STRANGER", score, color=(255, 0, 0))
            else:
                if liveness_detector:
                    liveness_detector.reset()
                liveness_required = False
                liveness_passed = False
                liveness_target_name = ""
                liveness_target_index = -1

            # ------------------------ 活体检测 ------------------------
            liveness_msg_to_draw = ""
            liveness_status_to_draw = ""
            if LIVENESS_ENABLED and liveness_detector is not None and has_known_face:
                target_name = target["name"]
                target_index = target["index"]
                target_box = target["box"]

                if not hasattr(fr, "has_liveness_landmark") or not fr.has_liveness_landmark():
                    liveness_required = False
                    liveness_passed = False
                    liveness_msg_to_draw = "[活体] 缺少106点模型，禁止开门"
                    now_warn = ticks_ms()
                    if ticks_diff(now_warn, liveness_fail_beep_time) > 3000:
                        liveness_fail_beep_time = now_warn
                        print("[活体] face_landmark.kmodel 不可用，拒绝开门")
                        safe_buzzer_beep(door_ctrl, "warning")
                else:
                    # Bug3 fix: 只在活体未进行中才启动，避免同一次流程内频繁重复启动
                    if not liveness_required and not liveness_passed:
                        liveness_required = True
                        liveness_passed = False
                        liveness_target_name = target_name
                        liveness_target_index = target_index
                        liveness_start_time = ticks_ms()
                        nod_upper_consecutive_count = 0
                        nod_upper_peak = 0.0
                        liveness_detector.reset()
                        # 显式调用 begin() 触发随机动作选择
                        liveness_detector.begin()
                        challenge = liveness_detector.challenge_action
                        if challenge:
                            action_name_map = {"blink": "眨眼", "nod": "上下点头", "mouth": "张嘴"}
                            action_cn = action_name_map.get(challenge, challenge)
                            ui_msg = "[活体检测] 请完成：{}".format(action_cn)
                            log_msg = "[活体] 检测到 {}，指定动作：{}，开始验证".format(target_name, action_cn)
                        else:
                            ui_msg = "[活体检测] 请完成动作"
                            log_msg = "[活体] 检测到 {}，开始活体验证".format(target_name)
                        set_ui_message(ui_msg, color=(255, 255, 255))
                        print(log_msg)

                    if liveness_required and not liveness_passed:
                        now_ms = ticks_ms()
                        if ticks_diff(now_ms, liveness_start_time) > LIVENESS_TIMEOUT_SEC * 1000 + 500:
                            print("[活体] 主流程超时，重置挑战")
                            liveness_detector.reset()
                            liveness_required = False
                            liveness_passed = False
                            nod_upper_consecutive_count = 0
                            nod_upper_peak = 0.0
                        else:
                            # FIX: 活体检测前检查内存，不足时提前GC
                            try:
                                mem_before = gc.mem_free()
                                if mem_before < 50000:  # 低于50KB时强制GC
                                    print("[内存] 活体检测前内存不足: {} bytes，执行GC".format(mem_before))
                                    gc.collect()
                            except Exception:
                                pass
                            
                            landmarks106 = fr.extract_landmark106(rgb888p_img_ndarry, target_box)
                            
                            # FIX: 检查关键点提取是否成功，失败时跳过本帧
                            if landmarks106 is None:
                                liveness_msg_to_draw = "[活体] 关键点提取失败，重试中"
                                # 连续失败时重置挑战
                                if ticks_diff(now_ms, liveness_start_time) > 3000:
                                    print("[活体] 关键点连续提取失败，重置挑战")
                                    liveness_detector.reset()
                                    liveness_required = False
                                    liveness_passed = False
                                    nod_upper_consecutive_count = 0
                                    nod_upper_peak = 0.0
                            else:
                                status, msg = liveness_detector.update(
                                    landmarks106, now_ms, face_box=target_box, frame_id=frame_count
                                )
                                liveness_status_to_draw = status
                                liveness_msg_to_draw = msg

                                # 上层点头快速判定：双阈值 + 峰值追踪，响应速度对齐眨眼/张嘴
                                challenge = liveness_detector.challenge_action
                                if challenge == "nod" and status not in ("passed", "timeout", "failed"):
                                    # 直接读取属性，避免 get_debug_info() 创建字典的开销
                                    pitch_range = abs(liveness_detector.pitch_max - liveness_detector.pitch_min)
                                    # 峰值追踪：持续记录最大幅度
                                    if pitch_range > nod_upper_peak:
                                        nod_upper_peak = pitch_range
                                    # 双阈值判定
                                    if pitch_range >= NOD_UPPER_SINGLE_THRESHOLD:
                                        # 单帧达到高阈值，立即通过
                                        liveness_detector.detected_action = "nod"
                                        status = "passed"
                                        msg = "检测到上下点头，活体通过"
                                        liveness_status_to_draw = status
                                        liveness_msg_to_draw = msg
                                        print("[活体][upper_nod] 单帧range={:.3f} 峰值={:.3f} 快速通过".format(
                                            pitch_range, nod_upper_peak))
                                    elif pitch_range >= NOD_UPPER_CONTINUOUS_THRESHOLD:
                                        nod_upper_consecutive_count += 1
                                        if nod_upper_consecutive_count >= NOD_UPPER_CONTINUOUS_FRAMES:
                                            # 连续2帧达到低阈值，通过
                                            liveness_detector.detected_action = "nod"
                                            status = "passed"
                                            msg = "检测到上下点头，活体通过"
                                            liveness_status_to_draw = status
                                            liveness_msg_to_draw = msg
                                            print("[活体][upper_nod] 连续{}帧range≥{:.3f} 峰值={:.3f} 通过".format(
                                                nod_upper_consecutive_count, NOD_UPPER_CONTINUOUS_THRESHOLD, nod_upper_peak))
                                    else:
                                        # 低于低阈值，重置计数（峰值保留）
                                        nod_upper_consecutive_count = 0

                                if status == "passed":
                                    # Bug1 fix: 校验触发通过的动作是否与指定动作一致
                                    challenge = liveness_detector.challenge_action
                                    detected = liveness_detector.detected_action
                                    if challenge and detected and challenge != detected:
                                        # 非指定动作触发通过，静默忽略
                                        action_name_map = {"blink": "眨眼", "nod": "上下点头", "mouth": "张嘴"}
                                        det_cn = action_name_map.get(detected, detected)
                                        chal_cn = action_name_map.get(challenge, challenge)
                                        print("[活体] 检测到非指定动作：{}，继续等待指定动作：{}".format(det_cn, chal_cn))
                                        # 不重置状态，继续等待指定动作
                                        liveness_msg_to_draw = "[活体] 请完成：{}".format(chal_cn)
                                    else:
                                        liveness_passed = True
                                        liveness_required = False
                                        print("[活体] 验证通过 - {}".format(liveness_target_name))
                                        set_ui_message("[通过] {} 活体通过，正在开门".format(liveness_target_name), color=(0, 255, 0))
                                        now_door = ticks_ms()
                                        if ticks_diff(now_door, last_door_open) > DOOR_OPEN_COOLDOWN_MS:
                                            last_door_open = now_door
                                            safe_buzzer_beep(door_ctrl, "success")
                                            if nb_door.request_open(name=liveness_target_name, score=target.get("score", 0.0), extra="liveness_passed"):
                                                set_ui_message("[门禁] 门已打开，请通过", color=(0, 255, 0))
                                                print("[门禁] 门已打开，请通过 - 欢迎 {}".format(liveness_target_name))
                                                log_event("access_granted", liveness_target_name, target.get("score", 0.0), "liveness_passed")
                                                door_notice_text = "[门禁] 门已打开，请通过"
                                                door_notice_until = ticks_ms() + 1200
                                            else:
                                                set_ui_message("[门禁] 开门请求失败，请检查继电器", color=(255, 0, 0))
                                                print("[门禁] 开门请求失败，请检查继电器")
                                elif status == "timeout" or status == "failed":
                                    print("[活体] 验证未通过 - {}".format(msg))
                                    log_event("liveness_fail", liveness_target_name, target.get("score", 0.0), msg)
                                    liveness_required = False
                                    liveness_passed = False
                                    nod_upper_consecutive_count = 0
                                    nod_upper_peak = 0.0
                                    safe_buzzer_beep(door_ctrl, "warning")
                                else:
                                    liveness_msg_to_draw = "[活体] " + msg
            else:
                if not has_known_face:
                    if liveness_detector:
                        liveness_detector.reset()
                    liveness_required = False
                    liveness_passed = False
                    liveness_target_name = ""
                    liveness_target_index = -1
                    nod_upper_consecutive_count = 0
                    nod_upper_peak = 0.0

            # ------------------------ 陌生人报警 ------------------------
            if has_unknown_face:
                stranger_frame_count += 1
                if stranger_frame_count >= stranger_alert_threshold:
                    now_stranger = ticks_ms()
                    if ticks_diff(now_stranger, last_stranger_alert) > STRANGER_COOLDOWN_MS:
                        last_stranger_alert = now_stranger
                        safe_buzzer_beep(door_ctrl, "stranger")
                        set_ui_message("[警告] 陌生人! 请联系管理员", color=(255, 0, 0))
                        print("[报警] 检测到陌生人")
                        log_event("stranger", "unknown", 0.0, "连续检测到陌生人")
            else:
                stranger_frame_count = 0

            # ------------------------ UI 绘制 ------------------------
            fps, frame_time = fps_monitor.tick()
            total_time = ticks_diff(ticks_ms(), frame_start)
            draw_ui_header(img_display, current_mode, fps, total_time, nb_door, face_count)

            if nb_door.is_open():
                current_ui_line = "[门禁] 门已打开，请通过"
                draw_guide_text(img_display, current_ui_line, color=(0, 255, 0))
            elif ticks_diff(ticks_ms(), door_notice_until) < 0 and door_notice_text:
                current_ui_line = door_notice_text
                draw_guide_text(img_display, current_ui_line, color=(0, 255, 0))
            elif has_unknown_face:
                current_ui_line = "[警告] 陌生人! 请联系管理员"
                draw_guide_text(img_display, current_ui_line, color=(255, 0, 0))
            elif liveness_msg_to_draw:
                current_ui_line = liveness_msg_to_draw
                draw_guide_text(img_display, current_ui_line)
            elif liveness_passed:
                # 活体已通过，等待开门，不显示旧版全动作提示
                current_ui_line = "[通过] {} 活体通过".format(liveness_target_name)
                draw_guide_text(img_display, current_ui_line, color=(0, 255, 0))
            elif has_known_face and LIVENESS_ENABLED:
                # Bug2 fix: 活体进行中但尚未收到消息时，显示指定动作而非旧版全动作提示
                if liveness_required and liveness_detector.challenge_action:
                    action_name_map = {"blink": "眨眼", "nod": "上下点头", "mouth": "张嘴"}
                    chal_cn = action_name_map.get(liveness_detector.challenge_action, "动作")
                    current_ui_line = "[活体检测] 请完成：{}".format(chal_cn)
                else:
                    current_ui_line = "[活体检测] 请完成动作"
                draw_guide_text(img_display, current_ui_line)
            else:
                current_ui_line = "[提示] 请正脸对准摄像头"
                draw_guide_text(img_display, current_ui_line)

            if current_ui_line != last_ui_line:
                last_ui_line = current_ui_line
                print("[UI] " + current_ui_line)

            draw_time = ticks_diff(ticks_ms(), draw_start)
            update_led(status_led, nb_door, liveness_required, has_unknown_face)
            Display.show_image(img_display)

            # ------------------------ 日志与内存 ------------------------
            if log_uploader is not None:
                try:
                    log_uploader.check_and_upload()
                except Exception as e:
                    print("[IoT] 定时上传异常: {}".format(e))

            if frame_count % 30 == 0:
                try:
                    mem_free = gc.mem_free()
                except Exception:
                    mem_free = 0
                print("[性能] 帧#{} | 总:{}ms | 转换:{}ms | 推理:{}ms | 绘制:{}ms | FPS:{:.1f} | 内存:{}".format(
                    frame_count, total_time, format_time, infer_time, draw_time, fps, mem_free
                ))
                # UART 发送性能数据
                uart_send({
                    "device_id": "1",
                    "device_name": "office_door_001",
                    "type": "performance",
                    "frame": frame_count,
                    "total_ms": total_time,
                    "format_ms": format_time,
                    "infer_ms": infer_time,
                    "draw_ms": draw_time,
                    "fps": fps,
                    "mem_free": mem_free
                })
                if liveness_detector is not None and hasattr(liveness_detector, "get_debug_info"):
                    try:
                        info = liveness_detector.get_debug_info()
                        print("[活体DBG] status:{} blink:{}/{} ear:{:.3f} move:{:.3f}".format(
                            info.get("status"), info.get("blink_count"), info.get("required_blinks"),
                            info.get("ear"), info.get("motion_ratio")
                        ))
                    except Exception:
                        pass

            now = ticks_ms()
            if ticks_diff(now, last_gc_time) > GC_INTERVAL_MS:
                if liveness_detector is not None and hasattr(liveness_detector, "collect_garbage_if_needed"):
                    liveness_detector.collect_garbage_if_needed()
                gc.collect()
                last_gc_time = now

            if watchdog.check():
                print("[看门狗] 主循环可能卡死，执行软恢复")
                if watchdog.record_error():
                    gc.collect()
                    watchdog.feed()
                else:
                    print("[看门狗] 连续错误过多，退出主循环")
                    break

        except KeyboardInterrupt:
            print("\n[系统] 用户中断")
            break
        except MemoryError:
            print("[错误] 内存不足，尝试恢复")
            gc.collect()
            sleep_ms(100)
            watchdog.feed()
        except Exception as e:
            print("[错误] 主循环异常: {}".format(e))
            gc.collect()
            if watchdog.record_error():
                sleep_ms(300)
                watchdog.feed()
            else:
                print("[错误] 连续异常过多，退出主循环")
                break

    # ------------------------ 资源清理 ------------------------
    print("[系统] 正在关闭...")
    try:
        nb_door.cancel()
    except Exception:
        pass
    EVENT_CTRL = None
    try:
        fr.deinit()
    except Exception:
        pass
    try:
        sensor.stop()
    except Exception:
        pass
    try:
        Display.deinit()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    try:
        if door_ctrl:
            door_ctrl.deinit()
    except Exception:
        pass
    try:
        if status_led:
            status_led.deinit()
    except Exception:
        pass
    gc.collect()
    print("[系统] 已安全关闭")


if __name__ == "__main__":
    main()
