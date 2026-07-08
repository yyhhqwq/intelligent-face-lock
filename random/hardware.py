"""
===============================================================================
CanMV K230 门禁硬件控制模块

控制对象：继电器（门锁）、蜂鸣器（提示音）、LED（状态指示）
默认 GPIO：继电器 63，蜂鸣器 62，LED 61。
===============================================================================
"""

import time
from machine import Pin


def _ticks_ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.time() * 1000)


def _ticks_diff(a, b):
    return time.ticks_diff(a, b) if hasattr(time, "ticks_diff") else a - b


def _sleep_ms(ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


class Buzzer:
    """蜂鸣器控制，支持阻塞 beep 和非阻塞 pattern。"""

    def __init__(self, pin=62, active_high=True):
        self.pin = None
        self.active_high = active_high
        self._pattern = []
        self._pattern_index = 0
        self._next_switch_ms = 0
        self._running = False
        try:
            self.pin = Pin(pin, Pin.OUT)
            self._write(False)
        except Exception as e:
            print("[错误] 蜂鸣器 Pin({}) 初始化失败: {}".format(pin, e))
            self.pin = None

    def _write(self, on):
        if self.pin is None:
            return
        try:
            v = 1 if on else 0
            if not self.active_high:
                v = 0 if on else 1
            self.pin.value(v)
        except Exception:
            pass

    def beep(self, duration_ms=100):
        """短促蜂鸣（同步阻塞）。旧接口保留，主循环中建议改用 start_pattern。"""
        if self.pin is None:
            return
        self._write(True)
        _sleep_ms(duration_ms)
        self._write(False)

    def start_pattern(self, pattern):
        """启动非阻塞蜂鸣。

        pattern: [(on, duration_ms), ...]
        需要在主循环中调用 update() 推进。
        """
        if self.pin is None:
            return False
        if not pattern:
            self.stop()
            return False
        self._pattern = pattern
        self._pattern_index = 0
        self._running = True
        self._next_switch_ms = _ticks_ms() + int(pattern[0][1])
        self._write(bool(pattern[0][0]))
        return True

    def update(self):
        if not self._running:
            return
        now = _ticks_ms()
        if _ticks_diff(now, self._next_switch_ms) < 0:
            return
        self._pattern_index += 1
        if self._pattern_index >= len(self._pattern):
            self.stop()
            return
        on, duration_ms = self._pattern[self._pattern_index]
        self._write(bool(on))
        self._next_switch_ms = now + int(duration_ms)

    def stop(self):
        self._running = False
        self._pattern = []
        self._pattern_index = 0
        self._write(False)

    def beep_on(self):
        self._running = False
        self._write(True)

    def beep_off(self):
        self.stop()

    def beep_success(self):
        """成功提示音：短-短-长，非阻塞。"""
        self.start_pattern([(1, 80), (0, 60), (1, 80), (0, 60), (1, 260), (0, 1)])

    def beep_warning(self):
        """警告提示音：长-短，非阻塞。"""
        self.start_pattern([(1, 420), (0, 80), (1, 120), (0, 1)])

    def beep_stranger(self):
        """陌生人报警音：急促三声，非阻塞。"""
        self.start_pattern([(1, 100), (0, 60), (1, 100), (0, 60), (1, 100), (0, 1)])

    def beep_door_open(self):
        self.start_pattern([(1, 180), (0, 1)])

    def deinit(self):
        self.stop()


class Relay:
    """继电器控制（门锁）。"""

    def __init__(self, pin=63, active_high=True):
        self.pin = None
        self.active_high = active_high
        try:
            self.pin = Pin(pin, Pin.OUT)
            self.lock()
        except Exception as e:
            print("[错误] 继电器 Pin({}) 初始化失败: {}".format(pin, e))
            self.pin = None

    def _write(self, on):
        if self.pin is None:
            return
        try:
            v = 1 if on else 0
            if not self.active_high:
                v = 0 if on else 1
            self.pin.value(v)
        except Exception:
            pass

    def unlock(self):
        self._write(True)

    def lock(self):
        self._write(False)

    def deinit(self):
        self.lock()


class StatusLED:
    """状态 LED 控制。"""

    def __init__(self, pin=61, active_high=True):
        self.pin = None
        self.active_high = active_high
        self._blink_state = False
        self._last_blink = 0
        self._blink_interval = 500
        try:
            self.pin = Pin(pin, Pin.OUT)
            self.off()
        except Exception as e:
            print("[错误] LED Pin({}) 初始化失败: {}".format(pin, e))
            self.pin = None

    def _write(self, on):
        if self.pin is None:
            return
        try:
            v = 1 if on else 0
            if not self.active_high:
                v = 0 if on else 1
            self.pin.value(v)
        except Exception:
            pass

    def on(self):
        self._write(True)

    def off(self):
        self._write(False)

    def toggle(self):
        self._blink_state = not self._blink_state
        self._write(self._blink_state)

    def blink(self, interval_ms=500):
        """非阻塞闪烁，需在主循环中调用。"""
        now = _ticks_ms()
        self._blink_interval = interval_ms
        if _ticks_diff(now, self._last_blink) >= self._blink_interval:
            self.toggle()
            self._last_blink = now

    def deinit(self):
        self.off()


class DoorAccessController:
    """门禁控制器（继电器 + 蜂鸣器）。"""

    def __init__(self, relay_pin=63, buzzer_pin=62, open_duration=3):
        self.relay = None
        self.buzzer = None
        self.open_duration = open_duration
        self._door_open = False
        self._open_start_ms = 0
        try:
            self.relay = Relay(relay_pin)
        except Exception as e:
            print("[错误] 继电器初始化失败: {}".format(e))
        try:
            self.buzzer = Buzzer(buzzer_pin, active_high=False)
        except Exception as e:
            print("[错误] 蜂鸣器初始化失败: {}".format(e))

    def is_door_open(self):
        return self._door_open

    def request_open(self, with_beep=True):
        """非阻塞开门请求。主循环需持续调用 update()。"""
        if self._door_open or self.relay is None:
            return False
        try:
            self.relay.unlock()
            self._door_open = True
            self._open_start_ms = _ticks_ms()
            if with_beep and self.buzzer is not None:
                self.buzzer.beep_door_open()
            return True
        except Exception as e:
            print("[门禁] 开门请求失败: {}".format(e))
            return False

    def update(self):
        if self.buzzer is not None:
            self.buzzer.update()
        if self._door_open:
            if _ticks_diff(_ticks_ms(), self._open_start_ms) >= int(self.open_duration * 1000):
                self.close_door()

    def close_door(self):
        try:
            if self.relay is not None:
                self.relay.lock()
        except Exception:
            pass
        self._door_open = False

    def open_door(self, with_beep=True):
        """阻塞式开门。保留旧接口，新主程序使用 request_open/update。"""
        if not self.request_open(with_beep=with_beep):
            return
        _sleep_ms(int(self.open_duration * 1000))
        self.close_door()

    def stranger_alert(self):
        if self.buzzer is not None:
            self.buzzer.beep_stranger()

    def access_granted_signal(self):
        if self.buzzer is not None:
            self.buzzer.beep_success()

    def warning_signal(self):
        if self.buzzer is not None:
            self.buzzer.beep_warning()

    def deinit(self):
        self.close_door()
        try:
            if self.buzzer is not None:
                self.buzzer.deinit()
        except Exception:
            pass
        try:
            if self.relay is not None:
                self.relay.deinit()
        except Exception:
            pass
