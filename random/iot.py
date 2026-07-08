"""
===============================================================================
IoT 日志上传模块
===============================================================================
"""

import time
import os

try:
    import urequests
except Exception:
    urequests = None


def _ticks_ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.time() * 1000)


def _ticks_diff(a, b):
    return time.ticks_diff(a, b) if hasattr(time, "ticks_diff") else a - b


def _sleep_ms(ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


class LogUploader:
    def __init__(self, server_url, log_file_path, upload_interval=300,
                 max_upload_bytes=16 * 1024):
        self.server_url = server_url
        self.log_file_path = log_file_path
        self.upload_interval_ms = int(upload_interval * 1000)
        self.max_upload_bytes = max_upload_bytes
        self.last_upload_ms = 0

    def read_log(self):
        try:
            with open(self.log_file_path, "r") as f:
                content = f.read(self.max_upload_bytes)
            return content
        except Exception as e:
            print("[IoT] 读取日志失败: {}".format(e))
            return None

    def append_log(self, line):
        try:
            with open(self.log_file_path, "a") as f:
                f.write(str(line) + "\n")
            return True
        except Exception as e:
            print("[IoT] 写日志失败: {}".format(e))
            return False

    def clear_log(self):
        try:
            with open(self.log_file_path, "w") as f:
                f.write("")
            return True
        except Exception as e:
            print("[IoT] 清空日志失败: {}".format(e))
            return False

    def upload_log(self, content):
        if urequests is None:
            print("[IoT] urequests 不可用，跳过上传")
            return False
        if not content or content.strip() == "":
            return False

        response = None
        try:
            headers = {"Content-Type": "text/plain; charset=utf-8"}
            response = urequests.post(self.server_url, data=content, headers=headers)
            code = getattr(response, "status_code", 0)
            if code == 200 or code == 201 or code == 204:
                print("[IoT] 日志上传成功")
                return True
            print("[IoT] 日志上传失败，状态码: {}".format(code))
            return False
        except Exception as e:
            print("[IoT] 上传异常: {}".format(e))
            return False
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception:
                pass

    def check_and_upload(self):
        now = _ticks_ms()
        if _ticks_diff(now, self.last_upload_ms) < self.upload_interval_ms:
            return False
        self.last_upload_ms = now
        log_content = self.read_log()
        if not log_content:
            return False
        success = self.upload_log(log_content)
        if success:
            self.clear_log()
        return success

    def start(self):
        print("[IoT] 日志上传器已启动，上传间隔: {}ms".format(self.upload_interval_ms))


class DoorController:
    """兼容旧代码的简单门控制器。新工程推荐使用 hardware.DoorAccessController。"""

    def __init__(self, door_pin, open_duration=3):
        from machine import Pin
        self.door_pin = door_pin
        self.open_duration = open_duration
        self.pin = None
        self._open = False
        self._open_start_ms = 0
        try:
            self.pin = Pin(door_pin, Pin.OUT, pull=Pin.PULL_NONE)
            self.close_door()
            print("[门控] 已初始化，引脚: {}".format(door_pin))
        except Exception as e:
            print("[错误] 门控制器 Pin({}) 初始化失败: {}".format(door_pin, e))
            self.pin = None

    def open_door(self):
        if self.pin is not None:
            try:
                self.pin.high()
            except Exception:
                self.pin.value(1)
        self._open = True
        self._open_start_ms = _ticks_ms()
        print("[门控] 门已打开")

    def close_door(self):
        if self.pin is not None:
            try:
                self.pin.low()
            except Exception:
                self.pin.value(0)
        self._open = False
        print("[门控] 门已关闭")

    def request_open(self):
        self.open_door()

    def update(self):
        if self._open:
            if _ticks_diff(_ticks_ms(), self._open_start_ms) >= int(self.open_duration * 1000):
                self.close_door()

    def open_and_auto_close(self):
        self.open_door()
        _sleep_ms(int(self.open_duration * 1000))
        self.close_door()

    def is_open(self):
        if self.pin is None:
            return False
        try:
            return self.pin.value() == 1
        except Exception:
            return self._open

    def set_open_duration(self, duration):
        self.open_duration = duration
        print("[门控] 门保持时间已设置为: {} 秒".format(duration))

    def deinit(self):
        self.close_door()
        print("[门控] 门控制器已释放")
