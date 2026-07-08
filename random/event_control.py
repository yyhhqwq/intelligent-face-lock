"""
===============================================================================
CanMV K230 事件控制模块
===============================================================================
"""

import os
import time
import gc

try:
    import _thread
except Exception:
    _thread = None


def _ticks_ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.time() * 1000)


def _ticks_diff(a, b):
    return time.ticks_diff(a, b) if hasattr(time, "ticks_diff") else a - b


def _sleep_ms(ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


def _ensure_dir(path):
    if not path:
        return
    # CanMV 的 os.mkdir 不能递归创建；逐级创建，失败忽略。
    parts = path.split("/")
    cur = ""
    for part in parts:
        if not part:
            continue
        cur += "/" + part
        try:
            os.mkdir(cur)
        except Exception:
            pass


def _safe_text(value):
    try:
        text = str(value)
    except Exception:
        text = ""
    # CSV 简单转义：避免换行破坏日志。
    text = text.replace("\r", " ").replace("\n", " ").replace(",", "|")
    return text


class _DummyLock:
    def acquire(self):
        return True

    def release(self):
        return True


class EventControlProcess:
    """事件控制“进程”。

    在 MicroPython 下实际优先使用 _thread；不可用时通过 main.py 每帧 update() 执行。
    队列里只传小对象，不传图像帧，不传 KPU tensor。
    """

    def __init__(self, door_ctrl=None, log_dir="/sdcard/cp/logs", log_file="attendance.csv",
                 max_queue=24, upload_enabled=False, upload_url=""):
        self.door_ctrl = door_ctrl
        self.log_dir = log_dir
        self.log_path = self._join(log_dir, log_file)
        self.max_queue = max_queue
        self.upload_enabled = upload_enabled
        self.upload_url = upload_url
        self._queue = []
        self._running = False
        self._thread_started = False
        self._lock = _thread.allocate_lock() if _thread is not None else _DummyLock()
        self._last_open_state = False
        self._door_closed_event = False
        self._last_upload_ms = 0
        self._upload_interval_ms = 300000
        _ensure_dir(log_dir)
        self._init_log_file()

    def _join(self, directory, filename):
        if directory.endswith("/"):
            return directory + filename
        return directory + "/" + filename

    def _init_log_file(self):
        try:
            # 文件不存在时写表头；存在则追加。
            exists = False
            try:
                parts = self.log_path.split("/")
                name = parts[-1]
                d = "/".join(parts[:-1]) or "/"
                exists = name in os.listdir(d)
            except Exception:
                exists = False
            if not exists:
                with open(self.log_path, "w") as f:
                    f.write("time_ms,event,name,score,extra\n")
        except Exception as e:
            print("[事件] 日志初始化失败: {}".format(e))

    def start(self):
        """启动事件控制线程；失败时返回 False，主循环继续协作 update。"""
        self._running = True
        if _thread is None:
            print("[事件] _thread 不可用，使用主循环协作事件控制")
            return False
        try:
            _thread.start_new_thread(self._thread_loop, ())
            self._thread_started = True
            print("[事件] 事件控制线程已启动")
            return True
        except Exception as e:
            self._thread_started = False
            print("[事件] 事件控制线程启动失败，退化为协作模式: {}".format(e))
            return False

    def is_thread_started(self):
        return self._thread_started

    def stop(self):
        self._running = False
        self.enqueue(("stop",))
        if self.door_ctrl is not None:
            try:
                self.door_ctrl.close_door()
            except Exception:
                pass

    def enqueue(self, item):
        try:
            self._lock.acquire()
            if len(self._queue) >= self.max_queue:
                # 丢弃最旧的低优先级事件，保证门禁命令不被堵住。
                try:
                    self._queue.pop(0)
                except Exception:
                    self._queue = []
            self._queue.append(item)
            return True
        except Exception:
            return False
        finally:
            try:
                self._lock.release()
            except Exception:
                pass

    def _pop_event(self):
        try:
            self._lock.acquire()
            if not self._queue:
                return None
            return self._queue.pop(0)
        except Exception:
            return None
        finally:
            try:
                self._lock.release()
            except Exception:
                pass

    def request_open(self, name="", score=0.0, extra=""):
        return self.enqueue(("open", _safe_text(name), _safe_text(score), _safe_text(extra)))

    def beep(self, kind="short"):
        return self.enqueue(("beep", _safe_text(kind)))

    def log_event(self, event, name="", score="", extra=""):
        return self.enqueue(("log", _safe_text(event), _safe_text(name), _safe_text(score), _safe_text(extra)))

    def is_door_open(self):
        try:
            if self.door_ctrl is not None:
                return self.door_ctrl.is_door_open()
        except Exception:
            pass
        return False

    def pop_door_closed_event(self):
        if self._door_closed_event:
            self._door_closed_event = False
            return True
        return False

    def update(self):
        """处理队列并推进门锁非阻塞状态。线程模式和协作模式都会调用。"""
        # 先处理有限数量事件，避免一帧内长期阻塞。
        for _ in range(8):
            item = self._pop_event()
            if item is None:
                break
            self._handle_event(item)

        try:
            if self.door_ctrl is not None and hasattr(self.door_ctrl, "update"):
                self.door_ctrl.update()
        except Exception as e:
            print("[事件] 门锁 update 异常: {}".format(e))

        now_open = self.is_door_open()
        if self._last_open_state and not now_open:
            self._door_closed_event = True
            self._write_log("door_closed", "", "", "auto_close")
        self._last_open_state = now_open

        if self.upload_enabled:
            self._try_upload()

    def _handle_event(self, item):
        if not item:
            return
        kind = item[0]
        if kind == "stop":
            self._running = False
            return
        if kind == "open":
            name = item[1] if len(item) > 1 else ""
            score = item[2] if len(item) > 2 else ""
            extra = item[3] if len(item) > 3 else ""
            ok = False
            try:
                if self.door_ctrl is not None:
                    ok = self.door_ctrl.request_open(with_beep=True)
            except Exception as e:
                print("[事件] 开门执行异常: {}".format(e))
            self._write_log("door_open" if ok else "door_open_failed", name, score, extra)
            return
        if kind == "beep":
            self._do_beep(item[1] if len(item) > 1 else "short")
            return
        if kind == "log":
            self._write_log(item[1] if len(item) > 1 else "event",
                            item[2] if len(item) > 2 else "",
                            item[3] if len(item) > 3 else "",
                            item[4] if len(item) > 4 else "")
            return

    def _do_beep(self, beep_type):
        if self.door_ctrl is None or getattr(self.door_ctrl, "buzzer", None) is None:
            return
        try:
            bz = self.door_ctrl.buzzer
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
        except Exception as e:
            print("[事件] 蜂鸣器异常: {}".format(e))

    def _write_log(self, event, name="", score="", extra=""):
        try:
            with open(self.log_path, "a") as f:
                f.write("{},{},{},{},{}\n".format(_ticks_ms(), _safe_text(event), _safe_text(name), _safe_text(score), _safe_text(extra)))
            return True
        except Exception as e:
            print("[事件] 写日志失败: {}".format(e))
            return False

    def _try_upload(self):
        # 可选功能：若后续接入 Wi-Fi 和服务器，可在这里调用 cp.iot.LogUploader。
        now = _ticks_ms()
        if _ticks_diff(now, self._last_upload_ms) < self._upload_interval_ms:
            return
        self._last_upload_ms = now
        if not self.upload_url:
            return
        try:
            import cp.iot as iot
            uploader = iot.LogUploader(self.upload_url, self.log_path, upload_interval=300)
            uploader.check_and_upload()
        except Exception as e:
            print("[事件] Wi-Fi日志上传跳过: {}".format(e))

    def _thread_loop(self):
        print("[事件] 事件控制循环运行中")
        while self._running:
            try:
                self.update()
                _sleep_ms(20)
            except Exception as e:
                print("[事件] 循环异常: {}".format(e))
                gc.collect()
                _sleep_ms(100)
        print("[事件] 事件控制循环退出")
