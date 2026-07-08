"""
===============================================================================
CanMV K230 人脸注册模块
===============================================================================
"""

from libs.PipeLine import ScopedTiming
from media.media import *
from media.sensor import *
from media.display import *
from machine import Pin

import os
import time
import gc
import image
import nncase_runtime as nn
import ulab.numpy as np

try:
    import cp.recognition as _recog
except Exception:
    import recognition as _recog

ALIGN_UP = _recog.ALIGN_UP
FaceDetApp = _recog.FaceDetApp
FaceRegistrationApp = _recog.FaceRegistrationApp


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return 0


def _sleep_ms(ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


def _ticks_ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.time() * 1000)


def _ticks_diff(a, b):
    return time.ticks_diff(a, b) if hasattr(time, "ticks_diff") else a - b


def _join_path(directory, filename):
    if directory.endswith("/"):
        return directory + filename
    return directory + "/" + filename


def _ensure_dir(path):
    try:
        os.mkdir(path)
    except Exception:
        pass


def _safe_person_name(name):
    if name is None:
        return ""
    name = str(name).strip()
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    out = ""
    for ch in name:
        if ch in allowed:
            out += ch
    return out


class MatrixKeypad:
    """4x4 矩阵键盘。scan() 为短阻塞消抖，适合主循环低频调用。"""

    def __init__(self):
        row_pins = [28, 29, 30, 31]
        col_pins = [18, 19, 33, 35]
        self.rows = []
        self.cols = []
        self.key_map = [
            ["1", "2", "3", "A"],
            ["4", "5", "6", "B"],
            ["7", "8", "9", "C"],
            ["*", "0", "#", "D"]
        ]
        try:
            for p in row_pins:
                pin = Pin(p, Pin.OUT, pull=Pin.PULL_NONE, drive=7)
                pin.high()
                self.rows.append(pin)
        except Exception as e:
            print("[警告] 矩阵键盘行引脚初始化失败: {}".format(e))
            self.rows = []
        try:
            for p in col_pins:
                self.cols.append(Pin(p, Pin.IN, Pin.PULL_UP))
        except Exception as e:
            print("[警告] 矩阵键盘列引脚初始化失败: {}".format(e))
            self.cols = []

    def release_all_rows(self):
        for row in self.rows:
            try:
                row.high()
            except Exception:
                pass

    def scan(self):
        if _safe_len(self.rows) == 0 or _safe_len(self.cols) == 0:
            return None
        for i in range(len(self.rows)):
            try:
                self.release_all_rows()
                self.rows[i].low()
                _sleep_ms(1)
                for j in range(len(self.cols)):
                    if self.cols[j].value() == 0:
                        _sleep_ms(20)
                        if self.cols[j].value() != 0:
                            continue
                        # FIX: 设置释放超时，避免按键/列线短路导致死等。
                        release_start = _ticks_ms()
                        while self.cols[j].value() == 0:
                            if _ticks_diff(_ticks_ms(), release_start) > 1200:
                                break
                            _sleep_ms(10)
                        self.release_all_rows()
                        return self.key_map[i][j]
            except Exception:
                pass
            finally:
                try:
                    self.rows[i].high()
                except Exception:
                    pass
        self.release_all_rows()
        return None


def get_filename_from_keypad(keypad=None, timeout_ms=30000, max_len=12, ui_callback=None):
    """读取编号。D/# 确认，* 退格，C 取消。返回 '编号.jpg' 或空字符串。"""
    try:
        kp = keypad if keypad is not None else MatrixKeypad()
    except Exception as e:
        print("[错误] 键盘初始化失败: {}".format(e))
        return ""

    print("等待输入编号... 数字键输入，D/#确认，*退格，C取消")
    def _notify_input():
        if ui_callback is not None:
            try:
                ui_callback("[输入] 编号:{}  D/#确认 C取消".format(name_str if name_str else "_"))
            except Exception:
                pass
    name_str = ""
    start_time = _ticks_ms()
    last_ui = 0
    _notify_input()
    while True:
        now_loop = _ticks_ms()
        if _ticks_diff(now_loop, start_time) > timeout_ms:
            print("\n[超时] 输入超时，取消操作")
            if ui_callback is not None:
                try:
                    ui_callback("[输入] 超时，取消操作")
                except Exception:
                    pass
            return ""
        if _ticks_diff(now_loop, last_ui) > 500:
            last_ui = now_loop
            _notify_input()
        try:
            key = kp.scan()
        except Exception:
            key = None
        if key:
            if key == "D" or key == "#":
                safe_name = _safe_person_name(name_str)
                if safe_name:
                    print("\n编号确认: {}".format(safe_name))
                    if ui_callback is not None:
                        try:
                            ui_callback("[输入] 编号确认:{}".format(safe_name))
                        except Exception:
                            pass
                    return safe_name + ".jpg"
                print("\n编号不能为空，请重新输入")
                _notify_input()
            elif key == "C":
                print("\n[取消] 用户取消输入")
                if ui_callback is not None:
                    try:
                        ui_callback("[输入] 已取消")
                    except Exception:
                        pass
                return ""
            elif key == "*":
                name_str = name_str[:-1]
                print("\r当前编号: {}   ".format(name_str), end="")
                _notify_input()
            else:
                if len(name_str) < max_len:
                    name_str += key
                print("\r当前编号: {}   ".format(name_str), end="")
                _notify_input()
        _sleep_ms(20)


class FaceRegistration:
    """注册专用 AI 封装。"""

    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size,
                 database_dir, anchors, confidence_threshold=0.25, nms_threshold=0.3,
                 rgb888p_size=[1280, 720], display_size=[512, 320], debug_mode=0):
        self.face_det_kmodel = face_det_kmodel
        self.face_reg_kmodel = face_reg_kmodel
        self.det_input_size = det_input_size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.face_det = FaceDetApp(
            self.face_det_kmodel, model_input_size=self.det_input_size,
            anchors=self.anchors, confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold, rgb888p_size=self.rgb888p_size,
            display_size=self.display_size, debug_mode=0
        )
        self.face_reg = FaceRegistrationApp(
            self.face_reg_kmodel, model_input_size=self.reg_input_size,
            rgb888p_size=self.rgb888p_size, display_size=self.display_size
        )

    def image2rgb888array(self, img):
        with ScopedTiming("registration image2rgb888array", self.debug_mode > 0):
            try:
                img_data_rgb888p = img.to_rgb888p()
                img_chw = img_data_rgb888p.to_numpy_ref()
                shape = img_chw.shape
                if len(shape) == 3 and shape[0] == 3:
                    return img_chw.reshape((1, shape[0], shape[1], shape[2]))
            except Exception:
                pass
            img_data_rgb888 = img.to_rgb888()
            img_hwc = img_data_rgb888.to_numpy_ref()
            shape = img_hwc.shape
            img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
            img_tmp_trans = img_tmp.transpose()
            img_res = img_tmp_trans.copy()
            return img_res.reshape((1, shape[2], shape[0], shape[1]))

    def extract_feature(self, input_np, landm):
        self.face_reg.config_preprocess(landm, input_image_size=[input_np.shape[3], input_np.shape[2]])
        feature = self.face_reg.run(input_np)
        try:
            return feature.flatten()
        except Exception:
            return feature

    def deinit(self):
        try:
            self.face_det.deinit()
        except Exception:
            pass
        try:
            self.face_reg.deinit()
        except Exception:
            pass
        gc.collect()


def capture_face(database_img_dir, capture_name="new_face.jpg"):
    """独立调试用拍照接口；主程序注册不依赖该函数。"""
    _ensure_dir(database_img_dir)
    print("初始化摄像头...")
    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(width=512, height=320)
    sensor.set_pixformat(Sensor.RGB565)
    Display.init(Display.NT35516, 512, 320, 0, 0)
    MediaManager.init()
    sensor.run()

    print("请对准摄像头，3秒后自动拍照...")
    for _ in range(30):
        img = sensor.snapshot()
        Display.show_image(img)
        _sleep_ms(100)

    img = sensor.snapshot()
    Display.show_image(img)
    save_path = _join_path(database_img_dir, capture_name)
    img.save(save_path)
    print("拍照成功，已保存至: {}".format(save_path))

    sensor.stop()
    Display.deinit()
    MediaManager.deinit()
    _sleep_ms(500)


def _draw_registration_tip(sensor, text, collected, total, ui_callback=None):
    try:
        img_display = sensor.snapshot(chn=1)
        img_display.draw_string_advanced(10, 30, 24, text, color=(0, 255, 0))
        img_display.draw_string_advanced(10, 62, 24, "采集: {}/{}".format(collected, total), color=(255, 255, 0))
        Display.show_image(img_display)
    except Exception:
        pass
    if ui_callback is not None:
        try:
            ui_callback("[注册] {} {}/{}".format(text, collected, total))
        except Exception:
            pass



def _face_quality_ok(det_box, frame_w=1280, frame_h=720, min_face_size=90):
    """注册质量门控：正脸、居中、尺寸足够才入库，提高 30/50 张样本平均特征质量。"""
    try:
        x, y, w, h = float(det_box[0]), float(det_box[1]), float(det_box[2]), float(det_box[3])
    except Exception:
        return False, "人脸框异常"
    if w < min_face_size or h < min_face_size:
        return False, "请靠近摄像头"
    if x < 2 or y < 2 or x + w > frame_w - 2 or y + h > frame_h - 2:
        return False, "请把脸放到画面中间"
    ratio = w / h if h > 1 else 0
    if ratio < 0.55 or ratio > 1.55:
        return False, "请保持正脸"
    cx = x + w / 2
    cy = y + h / 2
    if abs(cx - frame_w / 2) > frame_w * 0.36 or abs(cy - frame_h / 2) > frame_h * 0.36:
        return False, "请对准画面中心"
    return True, "质量合格"


def _save_sample_image(sensor, sample_dir, index):
    """保存注册样本照片。失败不影响特征入库，避免因为 SD 写入异常卡住注册。"""
    try:
        _ensure_dir(sample_dir)
        img_sample = sensor.snapshot(chn=1)
        save_path = _join_path(sample_dir, "%03d.jpg" % index)
        img_sample.save(save_path)
        return True
    except Exception as e:
        if index == 1:
            print("[注册] 样本照片保存失败，后续仅保存特征: {}".format(e))
        return False


def do_face_registration(sensor, database_dir, database_img_dir,
                         face_det_kmodel_path, face_reg_kmodel_path, anchors_path,
                         face_det_input_size, face_reg_input_size,
                         confidence_threshold, nms_threshold, keypad=None,
                         rgb888p_size=[1280, 720], display_size=[512, 320],
                         capture_count=30, max_attempts=220, min_valid_count=None,
                         save_sample_images=True, min_face_size=90,
                         ui_callback=None):
    """主程序调用的注册流程。

    注意：调用前 main.py 会释放识别模型，避免 KPU 资源竞争；本函数结束后会释放注册模型。
    """
    capture_name = get_filename_from_keypad(keypad=keypad, ui_callback=ui_callback)
    if not capture_name:
        print("[注册] 输入取消或超时，取消注册")
        return False
    person_name = _safe_person_name(capture_name.split(".")[0])
    if not person_name:
        print("[注册] 编号为空，取消注册")
        return False

    _ensure_dir(database_dir)
    _ensure_dir(database_img_dir)
    sample_dir = _join_path(database_img_dir, person_name)
    if save_sample_images:
        _ensure_dir(sample_dir)

    try:
        anchors = np.fromfile(anchors_path, dtype=np.float)
        anchors = anchors.reshape((4200, 4))
    except Exception as e:
        print("[错误] 加载 anchors 文件失败: {}, {}".format(anchors_path, e))
        return False

    fr_reg = None
    try:
        fr_reg = FaceRegistration(
            face_det_kmodel_path, face_reg_kmodel_path,
            det_input_size=face_det_input_size, reg_input_size=face_reg_input_size,
            database_dir=database_dir, anchors=anchors,
            confidence_threshold=confidence_threshold, nms_threshold=nms_threshold,
            rgb888p_size=rgb888p_size, display_size=display_size
        )
    except Exception as e:
        print("[错误] 人脸注册模块初始化失败: {}".format(e))
        return False

    if min_valid_count is None:
        min_valid_count = max(10, int(capture_count * 0.60))
    print("请正脸对准摄像头，系统将采集 {} 张有效样本照片/特征，至少需 {} 张有效数据...".format(capture_count, min_valid_count))
    for _ in range(15):
        _draw_registration_tip(sensor, "请正脸对准摄像头", 0, capture_count, ui_callback)
        _sleep_ms(80)

    collected = 0
    attempts = 0
    avg_feature = None
    feature_dim = 0
    last_log = _ticks_ms()

    try:
        while collected < capture_count and attempts < max_attempts:
            attempts += 1
            try:
                img = sensor.snapshot(chn=0)
                input_np = fr_reg.image2rgb888array(img)
                fr_reg.face_det.config_preprocess(input_image_size=[input_np.shape[3], input_np.shape[2]])
                det_boxes, landms = fr_reg.face_det.run(input_np)
                face_num = _safe_len(det_boxes)

                if face_num != 1 or _safe_len(landms) != 1:
                    if _ticks_diff(_ticks_ms(), last_log) > 1000:
                        if face_num == 0:
                            print("[注册] 未检测到人脸")
                        else:
                            print("[注册] 检测到多张人脸，请保持画面中只有本人")
                        last_log = _ticks_ms()
                    _draw_registration_tip(sensor, "请保持单人正脸", collected, capture_count, ui_callback)
                    _sleep_ms(60)
                    continue

                quality_ok, quality_msg = _face_quality_ok(det_boxes[0], input_np.shape[3], input_np.shape[2], min_face_size)
                if not quality_ok:
                    if _ticks_diff(_ticks_ms(), last_log) > 1000:
                        print("[注册] 样本质量不足: {}".format(quality_msg))
                        last_log = _ticks_ms()
                    _draw_registration_tip(sensor, quality_msg, collected, capture_count, ui_callback)
                    _sleep_ms(60)
                    continue

                feature = fr_reg.extract_feature(input_np, landms[0])
                if feature_dim == 0:
                    feature_dim = _safe_len(feature)
                    if feature_dim <= 0:
                        print("[注册] 特征维度异常")
                        continue
                    avg_feature = np.zeros(feature_dim, dtype=np.float)

                if _safe_len(feature) != feature_dim:
                    print("[注册] 特征维度变化，跳过本帧: {} != {}".format(_safe_len(feature), feature_dim))
                    continue

                # 每帧先做 L2 归一化再累加，减少距离/光照变化带来的特征尺度偏差。
                try:
                    f_norm = np.linalg.norm(feature)
                    if f_norm < 1e-8:
                        print("[注册] 特征范数异常，跳过本帧")
                        continue
                    feature = feature / f_norm
                except Exception:
                    pass

                avg_feature += feature
                collected += 1
                if save_sample_images:
                    _save_sample_image(sensor, sample_dir, collected)
                print("\r[注册] 已采集: {}/{}".format(collected, capture_count), end="")
                _draw_registration_tip(sensor, "采集中，请保持稳定", collected, capture_count, ui_callback)

                if collected % 5 == 0:
                    gc.collect()
                _sleep_ms(50)
            except MemoryError:
                print("\n[注册] 内存不足，执行 GC 后继续")
                gc.collect()
                _sleep_ms(100)
            except Exception as e:
                print("\n[注册] 采集过程出错: {}".format(e))
                gc.collect()
                _sleep_ms(80)

        print("\n[注册] 共采集到 {} 张有效样本".format(collected))
        if collected == 0 or avg_feature is None:
            print("[注册] 未采集到有效人脸，注册失败")
            if ui_callback is not None:
                try:
                    ui_callback("[注册] 未采集到有效人脸，失败")
                except Exception:
                    pass
            return False
        if collected < min_valid_count:
            print("[注册] 有效帧不足: {}/{}，为保证准确率，本次不入库".format(collected, min_valid_count))
            if ui_callback is not None:
                try:
                    ui_callback("[注册] 有效帧不足，不入库")
                except Exception:
                    pass
            return False

        avg_feature /= collected
        try:
            norm = np.linalg.norm(avg_feature)
            if norm > 1e-8:
                avg_feature /= norm
        except Exception:
            pass

        db_file_path = _join_path(database_dir, person_name + ".bin")
        tmp_file_path = db_file_path + ".tmp"
        with open(tmp_file_path, "wb") as file:
            file.write(avg_feature.tobytes())
        try:
            os.remove(db_file_path)
        except Exception:
            pass
        try:
            os.rename(tmp_file_path, db_file_path)
        except Exception:
            # 某些固件 os.rename 对覆盖/跨目录限制较多；退回直接写入。
            with open(db_file_path, "wb") as file:
                file.write(avg_feature.tobytes())
            try:
                os.remove(tmp_file_path)
            except Exception:
                pass
        print("[注册] 平均特征向量已保存至: {}".format(db_file_path))

        try:
            log_file = _join_path(database_dir, "registration_log.txt")
            timestamp = str(time.time()) if hasattr(time, "time") else str(_ticks_ms())
            with open(log_file, "a") as f:
                f.write(timestamp + " - 用户: " + person_name + ", 采集数量: " + str(collected) + ", 样本目录: " + sample_dir + "\n")
        except Exception as e:
            print("[注册] 写入日志失败: {}".format(e))

        print("[注册] 人脸注册完成")
        if ui_callback is not None:
            try:
                ui_callback("[注册] 成功，已保存 {} 帧特征".format(collected))
            except Exception:
                pass
        return True
    finally:
        if fr_reg is not None:
            fr_reg.deinit()
        gc.collect()


if __name__ == "__main__":
    face_det_kmodel_path = "/sdcard/examples/kmodel/face_detection_320.kmodel"
    face_reg_kmodel_path = "/sdcard/examples/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/examples/utils/prior_data_320.bin"
    database_dir = "/sdcard/examples/utils/db/"
    database_img_dir = "/sdcard/examples/utils/db_img/"
    face_det_input_size = [320, 320]
    face_reg_input_size = [112, 112]
    confidence_threshold = 0.5
    nms_threshold = 0.2
    print("registration.py 建议由 main.py 调用；独立运行仅用于调试。")
