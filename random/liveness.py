"""
===============================================================================
CanMV K230 / RT-Thread Smart / MicroPython 多动作活体检测模块
===============================================================================
"""

import time

try:
    import gc
except ImportError:
    gc = None

try:
    import _thread
except ImportError:
    _thread = None


# ----------------------------- portable helpers ------------------------------

def _ticks_ms():
    if hasattr(time, "ticks_ms"):
        return time.ticks_ms()
    return int(time.time() * 1000)


def _ticks_diff(a, b):
    if hasattr(time, "ticks_diff"):
        return time.ticks_diff(a, b)
    return a - b


def _sleep_ms(ms):
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(ms)
    else:
        time.sleep(ms / 1000.0)


def _sqrt(v):
    return v ** 0.5


def _contains(seq, value):
    try:
        for item in seq:
            if item == value:
                return True
    except Exception:
        pass
    return False


# ------------------------- optional thread handoff ----------------------------

class LandmarkMailbox:
    """
    线程安全的“最新关键点邮箱”。

    说明：如果项目必须保留独立活体线程，只在识别线程 put() 关键点副本，
    活体线程 get_latest()，不要共享图像大缓冲、KPU 输出 Tensor 或 OSD 对象。
    """

    def __init__(self, max_landmark_values=212, use_lock=True):
        self.max_landmark_values = max_landmark_values
        self._landmarks = None
        self._face_box = None
        self._frame_id = None
        self._timestamp_ms = 0
        self._has_new = False
        self._lock = None
        if use_lock and _thread is not None:
            try:
                self._lock = _thread.allocate_lock()
            except Exception:
                self._lock = None

    def _acquire(self):
        if self._lock is not None:
            self._lock.acquire()

    def _release(self):
        if self._lock is not None:
            self._lock.release()

    def put(self, landmarks, face_box=None, frame_id=None, timestamp_ms=None):
        if landmarks is None:
            return False
        try:
            n = len(landmarks)
        except Exception:
            return False
        if n <= 0:
            return False
        if n > self.max_landmark_values:
            n = self.max_landmark_values

        data = [0.0] * n
        try:
            for i in range(n):
                data[i] = float(landmarks[i])
        except Exception:
            return False

        box = None
        if face_box is not None:
            try:
                box = (float(face_box[0]), float(face_box[1]),
                       float(face_box[2]), float(face_box[3]))
            except Exception:
                box = None

        ts = timestamp_ms if timestamp_ms is not None else _ticks_ms()
        self._acquire()
        try:
            self._landmarks = data
            self._face_box = box
            self._frame_id = frame_id
            self._timestamp_ms = ts
            self._has_new = True
        finally:
            self._release()
        return True

    def get_latest(self, only_new=False):
        self._acquire()
        try:
            if only_new and not self._has_new:
                return None, None, None, 0
            landmarks = self._landmarks
            face_box = self._face_box
            frame_id = self._frame_id
            timestamp_ms = self._timestamp_ms
            self._has_new = False
        finally:
            self._release()
        return landmarks, face_box, frame_id, timestamp_ms


# ----------------------------- liveness detector ------------------------------

class LivenessDetector:
    """
    CanMV K230 106 点关键点活体检测器。

    默认 action_mode="any"：眨眼、左右摇头、上下点头任一动作通过，适合比赛演示。
    若要增强抗预录视频攻击，可改为 action_mode="random"，系统每轮随机要求一个动作。
    """

    STATUS_IDLE = "idle"
    STATUS_PENDING = "pending"
    STATUS_PASSED = "passed"
    STATUS_FAILED = "failed"
    STATUS_TIMEOUT = "timeout"

    ACTION_BLINK = "blink"
    ACTION_SHAKE = "shake"
    ACTION_NOD = "nod"
    ACTION_MOUTH = "mouth"

    # CanMV K230 官方 106 点 face_landmark 眼部/鼻部/嘴部索引。
    CANMV106_LEFT_EYE = (35, 36, 33, 37, 39, 42, 40, 41)
    CANMV106_RIGHT_EYE = (89, 90, 87, 91, 93, 96, 94, 95)
    CANMV106_PUPILS = (34, 88)
    CANMV106_NOSE = (72, 73, 74, 77, 78, 79, 80, 83, 84, 85, 86)
    CANMV106_INNER_MOUTH = (65, 54, 60, 57, 69, 70, 62, 66)
    CANMV106_OUTER_MOUTH = (52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64)
    CANMV106_MOTION_POINTS = (33, 34, 35, 39, 42, 72, 77, 80, 83, 86, 88, 89, 93, 96)

    def __init__(self,
                 required_blinks=1,
                 timeout_sec=6,
                 action_mode="any",
                 allowed_actions=("blink", "nod", "mouth"),
                 left_eye=None,
                 right_eye=None,
                 pupils=None,
                 nose_points=None,
                 inner_mouth=None,
                 outer_mouth=None,
                 motion_points=None,
                 # blink sensitivity parameters
                 close_ear_threshold=0.23,
                 open_ear_threshold=0.25,
                 min_open_baseline=0.10,
                 min_close_ear=0.06,
                 max_close_ear=0.28,
                 close_ratio=0.82,
                 open_ratio=0.90,
                 hysteresis=0.015,
                 blink_min_drop=0.025,
                 accept_blink_on_close=True,
                 min_closed_frames=1,
                 min_open_frames=1,
                 min_closed_ms=0,
                 max_closed_ms=1200,
                 # head action parameters
                 shake_threshold=0.10,
                 shake_range_threshold=0.18,
                 nod_threshold=0.10,
                 nod_range_threshold=0.16,
                 action_window_ms=2500,
                 action_min_frames=2,
                 # optional mouth-open parameters
                 mouth_open_threshold=0.18,
                 mouth_open_delta=0.08,
                 # quality / thread parameters
                 max_missing_frames=4,
                 max_jump_ratio=0.60,
                 terminal_hold_ms=1200,
                 log_level=1,
                 log_interval_ms=300):
        self.required_blinks = required_blinks
        self.timeout_ms = int(timeout_sec * 1000)

        self.action_mode = action_mode
        self.allowed_actions = self._normalize_actions(allowed_actions)

        self.left_eye = left_eye if left_eye is not None else self.CANMV106_LEFT_EYE
        self.right_eye = right_eye if right_eye is not None else self.CANMV106_RIGHT_EYE
        self.pupils = pupils if pupils is not None else self.CANMV106_PUPILS
        self.nose_points = nose_points if nose_points is not None else self.CANMV106_NOSE
        self.inner_mouth = inner_mouth if inner_mouth is not None else self.CANMV106_INNER_MOUTH
        self.outer_mouth = outer_mouth if outer_mouth is not None else self.CANMV106_OUTER_MOUTH
        self.motion_points = motion_points if motion_points is not None else self.CANMV106_MOTION_POINTS

        self.close_ear_threshold = close_ear_threshold
        self.open_ear_threshold = open_ear_threshold
        self.min_open_baseline = min_open_baseline
        self.min_close_ear = min_close_ear
        self.max_close_ear = max_close_ear
        self.close_ratio = close_ratio
        self.open_ratio = open_ratio
        self.hysteresis = hysteresis
        self.blink_min_drop = blink_min_drop
        self.accept_blink_on_close = accept_blink_on_close
        self.min_closed_frames = min_closed_frames
        self.min_open_frames = min_open_frames
        self.min_closed_ms = min_closed_ms
        self.max_closed_ms = max_closed_ms

        self.shake_threshold = shake_threshold
        self.shake_range_threshold = shake_range_threshold
        self.nod_threshold = nod_threshold
        self.nod_range_threshold = nod_range_threshold
        self.action_window_ms = action_window_ms
        self.action_min_frames = action_min_frames

        self.mouth_open_threshold = mouth_open_threshold
        self.mouth_open_delta = mouth_open_delta

        self.max_missing_frames = max_missing_frames
        self.max_jump_ratio = max_jump_ratio
        self.terminal_hold_ms = terminal_hold_ms
        self.log_level = log_level
        self.log_interval_ms = log_interval_ms
        self._last_log_ms = 0

        self._required_values = self._calc_required_values()
        self._motion_prev = [0.0] * (len(self.motion_points) * 2)
        self.reset()

    # --------------------------- public interface ----------------------------

    def reset(self):
        self.started = False
        self.start_ms = 0
        self.terminal_ms = 0
        self.status = self.STATUS_IDLE
        self.message = ""
        self.challenge_action = None
        self.detected_action = ""

        # Blink state.
        self.blink_count = 0
        self.closed_frames = 0
        self.open_frames = 0
        self.in_closed = False
        self.closed_start_ms = 0
        self.blink_counted_in_candidate = False
        self.seen_open = False
        self.open_ear_baseline = None
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.ear = 0.0
        self.close_threshold_now = self.close_ear_threshold
        self.open_threshold_now = self.open_ear_threshold

        # Head pose state.
        self.shake_count = 0
        self.nod_count = 0
        self.yaw = 0.0
        self.pitch = 0.0
        self.yaw_delta = 0.0
        self.pitch_delta = 0.0
        self.pose_baseline_yaw = None
        self.pose_baseline_pitch = None
        self.pose_start_ms = 0
        self.pose_frames = 0
        self.yaw_min = 0.0
        self.yaw_max = 0.0
        self.pitch_min = 0.0
        self.pitch_max = 0.0
        self.yaw_left_seen = False
        self.yaw_right_seen = False
        self.pitch_up_seen = False
        self.pitch_down_seen = False

        # Mouth state, disabled unless allowed_actions contains "mouth".
        self.mouth_count = 0
        self.mouth_ratio = 0.0
        self.mouth_baseline = None

        # Quality / thread state.
        self.missing_frames = 0
        self.last_frame_id = None
        self.motion_ratio = 0.0
        self._motion_prev_valid = False

    def begin(self, current_time_ms=None):
        now = current_time_ms if current_time_ms is not None else _ticks_ms()
        mode = self.action_mode
        actions = self.allowed_actions
        self.reset()
        self.started = True
        self.start_ms = now
        self.status = self.STATUS_PENDING
        self.action_mode = mode
        self.allowed_actions = actions
        self.challenge_action = self._select_challenge(now)
        self.message = self._pending_message()
        self._log(1, now, "start", "mode={} action={} actions={}".format(
            self.action_mode, self.challenge_action, self.allowed_actions))

    def update(self, landmarks, current_time_ms=None, face_box=None, frame_id=None):
        """
        每帧更新。

        Args:
            landmarks: 106 点 flat 数组 [x0,y0,x1,y1,...]，共 212 个数值。
            current_time_ms: time.ticks_ms()。
            face_box: 同一目标人脸检测框 [x,y,w,h]。
            frame_id: 主循环单调递增帧号，避免多线程重复消费同一帧。
        """
        now = current_time_ms if current_time_ms is not None else _ticks_ms()

        if self.status in (self.STATUS_PASSED, self.STATUS_FAILED, self.STATUS_TIMEOUT):
            if self.terminal_hold_ms > 0 and _ticks_diff(now, self.terminal_ms) > self.terminal_hold_ms:
                self.reset()
            else:
                return self.status, self.message

        if frame_id is not None and frame_id == self.last_frame_id:
            return self.status, self.message
        self.last_frame_id = frame_id

        ok, reason = self._validate_landmarks(landmarks)
        if not ok:
            return self._handle_invalid(now, reason)

        if not self.started:
            self.begin(now)

        elapsed = _ticks_diff(now, self.start_ms)
        if elapsed > self.timeout_ms:
            self._finish(self.STATUS_TIMEOUT, "活体验证超时，请重新对准人脸", now)
            return self.status, self.message

        left_ratio, left_width = self._calc_open_ratio_by_points(landmarks, self.left_eye)
        right_ratio, right_width = self._calc_open_ratio_by_points(landmarks, self.right_eye)
        if left_ratio < 0.0 or right_ratio < 0.0:
            return self._handle_invalid(now, "眼部关键点异常")

        self.left_ear = left_ratio
        self.right_ear = right_ratio
        self.ear = (left_ratio + right_ratio) * 0.5

        # 第一次有效帧直接建立“睁眼基线”，使小眼睛/眼镜用户不被固定阈值卡死。
        if self.open_ear_baseline is None:
            self.open_ear_baseline = self.ear
            if self.ear >= self.min_open_baseline:
                self.seen_open = True

        self.close_threshold_now, self.open_threshold_now = self._thresholds()

        face_scale = self._estimate_face_scale(landmarks, face_box, left_width, right_width)
        self.motion_ratio = self._calc_motion_ratio(landmarks, face_scale)
        if self.motion_ratio > self.max_jump_ratio:
            # 不直接失败：摇头/点头时会有运动。这里只重置眨眼候选，避免抖动误计眨眼。
            self._reset_blink_candidate()

        # 动作 OR 逻辑：任一允许动作通过。
        if self._update_blink_action(now):
            return self.status, self.message
        if self._update_head_actions(landmarks, now):
            return self.status, self.message
        if self._update_mouth_action(landmarks, now):
            return self.status, self.message

        self.status = self.STATUS_PENDING
        self.message = self._pending_message()

        if self.log_level >= 2 and _ticks_diff(now, self._last_log_ms) >= self.log_interval_ms:
            self._last_log_ms = now
            self._log(2, now, "state",
                      "ear={:.3f} L={:.3f} R={:.3f} close={:.3f} open={:.3f} blink={} yaw={:.3f}/{:.3f} pitch={:.3f}/{:.3f} move={:.3f}".format(
                          self.ear, self.left_ear, self.right_ear,
                          self.close_threshold_now, self.open_threshold_now,
                          self.blink_count, self.yaw, self.yaw_delta,
                          self.pitch, self.pitch_delta, self.motion_ratio))
        return self.status, self.message

    def get_debug_info(self):
        return {
            "status": self.status,
            "message": self.message,
            "mode": self.action_mode,
            "challenge_action": self.challenge_action,
            "detected_action": self.detected_action,
            "allowed_actions": self.allowed_actions,
            "blink_count": self.blink_count,
            "shake_count": self.shake_count,
            "nod_count": self.nod_count,
            "mouth_count": self.mouth_count,
            "ear": self.ear,
            "left_ear": self.left_ear,
            "right_ear": self.right_ear,
            "close_threshold": self.close_threshold_now,
            "open_threshold": self.open_threshold_now,
            "baseline": self.open_ear_baseline,
            "yaw": self.yaw,
            "yaw_delta": self.yaw_delta,
            "pitch": self.pitch,
            "pitch_delta": self.pitch_delta,
            "mouth_ratio": self.mouth_ratio,
            "motion_ratio": self.motion_ratio,
            "started": self.started,
        }

    # ----------------------------- action config -----------------------------

    def _normalize_actions(self, actions):
        result = []
        if actions is None:
            actions = (self.ACTION_BLINK,)
        try:
            for a in actions:
                s = str(a).lower()
                if s in (self.ACTION_BLINK, self.ACTION_SHAKE, self.ACTION_NOD, self.ACTION_MOUTH):
                    if not _contains(result, s):
                        result.append(s)
        except Exception:
            result = []
        if len(result) == 0:
            result.append(self.ACTION_BLINK)
        return tuple(result)

    def _select_challenge(self, now):
        if self.action_mode != "random":
            return None
        n = len(self.allowed_actions)
        if n <= 0:
            return self.ACTION_BLINK
        # 使用系统时间作为随机种子，适配 MicroPython 原生语法。
        # 结合 now 的低 16 位和当前秒数，增加随机性。
        seed = (now & 0xFFFF) ^ (int(time.time()) & 0xFFFF)
        idx = int((seed * 31 + 7) % n)
        return self.allowed_actions[idx]

    def _action_can_pass(self, action):
        if not _contains(self.allowed_actions, action):
            return False
        if self.action_mode == "random":
            return action == self.challenge_action
        return True

    def _action_is_enabled(self, action):
        return _contains(self.allowed_actions, action)

    def _action_words(self):
        words = []
        if _contains(self.allowed_actions, self.ACTION_BLINK):
            words.append("眨眼")
        if _contains(self.allowed_actions, self.ACTION_SHAKE):
            words.append("左右摇头")
        if _contains(self.allowed_actions, self.ACTION_NOD):
            words.append("上下点头")
        if _contains(self.allowed_actions, self.ACTION_MOUTH):
            words.append("张嘴")
        if len(words) == 0:
            return "眨眼"
        text = words[0]
        for i in range(1, len(words)):
            text += "或" + words[i]
        return text

    def _action_name(self, action):
        if action == self.ACTION_BLINK:
            return "眨眼"
        if action == self.ACTION_SHAKE:
            return "左右摇头"
        if action == self.ACTION_NOD:
            return "上下点头"
        if action == self.ACTION_MOUTH:
            return "张嘴"
        return "动作"

    def _pending_message(self):
        if self.action_mode == "random" and self.challenge_action is not None:
            return "请完成动作：{}".format(self._action_name(self.challenge_action))
        if self.in_closed:
            return "检测到闭眼，请睁眼或继续{}".format(self._action_words())
        return "请{}，任一动作通过".format(self._action_words())

    # ----------------------------- blink action ------------------------------

    def _update_blink_action(self, now):
        if not self._action_is_enabled(self.ACTION_BLINK):
            return False

        ear = self.ear
        close_thr = self.close_threshold_now
        open_thr = self.open_threshold_now
        baseline = self.open_ear_baseline if self.open_ear_baseline is not None else ear
        drop = baseline - ear

        # 相对下降 + 绝对闭合双判据，比固定 EAR 更适配不同眼型和眼镜反光。
        is_close = False
        if self.seen_open:
            if ear <= close_thr and drop >= self.blink_min_drop:
                is_close = True
            elif baseline > 0.0 and ear <= baseline * self.close_ratio and drop >= self.blink_min_drop:
                is_close = True

        is_open = False
        if ear >= open_thr:
            is_open = True
        elif baseline > 0.0 and ear >= baseline * self.open_ratio:
            is_open = True

        if is_open:
            self.seen_open = True
            if not self.in_closed:
                self._update_open_baseline(ear)

        if not self.in_closed:
            if is_close:
                if self.closed_frames == 0:
                    self.closed_start_ms = now
                self.closed_frames += 1
                self.open_frames = 0
                if self.closed_frames >= self.min_closed_frames:
                    self.in_closed = True
                    self.blink_counted_in_candidate = False
                    self._log(1, now, "eye", "closed ear={:.3f} base={:.3f}".format(ear, baseline))
                    if self.accept_blink_on_close:
                        return self._count_blink(now, "close-edge")
            else:
                self.closed_frames = 0
                if is_open:
                    self.open_frames += 1
                else:
                    self.open_frames = 0
            return False

        # 已经进入闭眼候选，等待睁眼以解除候选；如果没有启用 close-edge，则在睁眼时计数。
        closed_ms = _ticks_diff(now, self.closed_start_ms)
        if is_close:
            self.closed_frames += 1
            self.open_frames = 0
            if closed_ms > self.max_closed_ms:
                self._reset_blink_candidate()
                self.message = "闭眼时间过长，请自然快速眨眼"
                self.status = self.STATUS_PENDING
                self._log(1, now, "eye", "closed_too_long={}ms".format(closed_ms))
                return False
        elif is_open:
            self.open_frames += 1
            if self.open_frames >= self.min_open_frames:
                if (not self.blink_counted_in_candidate) and closed_ms >= self.min_closed_ms:
                    passed = self._count_blink(now, "reopen")
                    self._reset_blink_candidate()
                    return passed
                self._reset_blink_candidate()
        return False

    def _count_blink(self, now, source):
        if self.blink_counted_in_candidate:
            return False
        self.blink_counted_in_candidate = True
        self.blink_count += 1
        self._log(1, now, "blink", "count={} source={}".format(self.blink_count, source))
        if self._action_can_pass(self.ACTION_BLINK) and self.blink_count >= self.required_blinks:
            return self._finish_action(self.ACTION_BLINK, now)
        return False

    def _reset_blink_candidate(self):
        self.in_closed = False
        self.closed_frames = 0
        self.open_frames = 0
        self.closed_start_ms = 0
        self.blink_counted_in_candidate = False

    # ----------------------------- head actions ------------------------------

    def _update_head_actions(self, landmarks, now):
        if (not self._action_is_enabled(self.ACTION_SHAKE)) and (not self._action_is_enabled(self.ACTION_NOD)):
            return False

        ok, yaw, pitch = self._calc_head_pose_ratios(landmarks)
        if not ok:
            return False

        self.yaw = yaw
        self.pitch = pitch

        if self.pose_baseline_yaw is None or self.pose_baseline_pitch is None:
            self.pose_baseline_yaw = yaw
            self.pose_baseline_pitch = pitch
            self._reset_pose_window(now, 0.0, 0.0)
            return False

        self.yaw_delta = yaw - self.pose_baseline_yaw
        self.pitch_delta = pitch - self.pose_baseline_pitch

        if self.pose_start_ms == 0 or _ticks_diff(now, self.pose_start_ms) > self.action_window_ms:
            self._reset_pose_window(now, self.yaw_delta, self.pitch_delta)
        else:
            self.pose_frames += 1
            if self.yaw_delta < self.yaw_min:
                self.yaw_min = self.yaw_delta
            if self.yaw_delta > self.yaw_max:
                self.yaw_max = self.yaw_delta
            if self.pitch_delta < self.pitch_min:
                self.pitch_min = self.pitch_delta
            if self.pitch_delta > self.pitch_max:
                self.pitch_max = self.pitch_delta

        # 人脸基本稳定时缓慢更新姿态基线，避免站位轻微偏移导致动作阈值漂移。
        if abs(self.yaw_delta) < self.shake_threshold * 0.45 and abs(self.pitch_delta) < self.nod_threshold * 0.45:
            self.pose_baseline_yaw = self.pose_baseline_yaw * 0.98 + yaw * 0.02
            self.pose_baseline_pitch = self.pose_baseline_pitch * 0.98 + pitch * 0.02

        if self._action_is_enabled(self.ACTION_SHAKE):
            if self.yaw_delta <= -self.shake_threshold:
                self.yaw_left_seen = True
            if self.yaw_delta >= self.shake_threshold:
                self.yaw_right_seen = True
            yaw_range = self.yaw_max - self.yaw_min
            if self.pose_frames >= self.action_min_frames:
                if (self.yaw_left_seen and self.yaw_right_seen) or yaw_range >= self.shake_range_threshold:
                    self.shake_count += 1
                    self._log(1, now, "shake", "range={:.3f}".format(yaw_range))
                    if self._action_can_pass(self.ACTION_SHAKE):
                        return self._finish_action(self.ACTION_SHAKE, now)
                    self._reset_pose_window(now, self.yaw_delta, self.pitch_delta)

        if self._action_is_enabled(self.ACTION_NOD):
            if self.pitch_delta <= -self.nod_threshold:
                self.pitch_up_seen = True
            if self.pitch_delta >= self.nod_threshold:
                self.pitch_down_seen = True
            pitch_range = self.pitch_max - self.pitch_min
            if self.pose_frames >= self.action_min_frames:
                if (self.pitch_up_seen and self.pitch_down_seen) or pitch_range >= self.nod_range_threshold:
                    self.nod_count += 1
                    self._log(1, now, "nod", "range={:.3f}".format(pitch_range))
                    if self._action_can_pass(self.ACTION_NOD):
                        return self._finish_action(self.ACTION_NOD, now)
                    self._reset_pose_window(now, self.yaw_delta, self.pitch_delta)

        return False

    def _reset_pose_window(self, now, yaw_delta, pitch_delta):
        self.pose_start_ms = now
        self.pose_frames = 1
        self.yaw_min = yaw_delta
        self.yaw_max = yaw_delta
        self.pitch_min = pitch_delta
        self.pitch_max = pitch_delta
        self.yaw_left_seen = yaw_delta <= -self.shake_threshold
        self.yaw_right_seen = yaw_delta >= self.shake_threshold
        self.pitch_up_seen = pitch_delta <= -self.nod_threshold
        self.pitch_down_seen = pitch_delta >= self.nod_threshold

    # ----------------------------- mouth action ------------------------------

    def _update_mouth_action(self, landmarks, now):
        if not self._action_is_enabled(self.ACTION_MOUTH):
            return False
        ratio, _ = self._calc_open_ratio_by_points(landmarks, self.inner_mouth)
        if ratio < 0.0:
            return False
        self.mouth_ratio = ratio
        if self.mouth_baseline is None:
            self.mouth_baseline = ratio
            return False
        if ratio < self.mouth_baseline:
            self.mouth_baseline = self.mouth_baseline * 0.95 + ratio * 0.05
        threshold = self.mouth_baseline + self.mouth_open_delta
        if threshold < self.mouth_open_threshold:
            threshold = self.mouth_open_threshold
        if ratio >= threshold:
            self.mouth_count += 1
            self._log(1, now, "mouth", "ratio={:.3f} threshold={:.3f}".format(ratio, threshold))
            if self._action_can_pass(self.ACTION_MOUTH):
                return self._finish_action(self.ACTION_MOUTH, now)
        return False

    # ------------------------- landmark and geometry -------------------------

    def _calc_required_values(self):
        max_idx = 0
        groups = (self.left_eye, self.right_eye, self.pupils, self.nose_points,
                  self.inner_mouth, self.outer_mouth, self.motion_points)
        for group in groups:
            if group is None:
                continue
            for idx in group:
                if idx > max_idx:
                    max_idx = idx
        return (max_idx + 1) * 2

    def _validate_landmarks(self, landmarks):
        if landmarks is None:
            return False, "未检测到人脸关键点"
        try:
            n = len(landmarks)
        except Exception:
            return False, "关键点对象不可读取长度"
        if n < self._required_values:
            return False, "关键点长度不足 {}/{}".format(n, self._required_values)
        return True, ""

    def _handle_invalid(self, now, reason):
        if not self.started:
            self.status = self.STATUS_IDLE
            self.message = "请对准人脸"
            return self.status, self.message
        self.missing_frames += 1
        if self.missing_frames > self.max_missing_frames:
            self._log(1, now, "reset", "face lost: {}".format(reason))
            self.reset()
            self.status = self.STATUS_IDLE
            self.message = "人脸丢失，请重新对准"
            return self.status, self.message
        self.status = self.STATUS_PENDING
        self.message = "请保持人脸在框内"
        return self.status, self.message

    def _point(self, landmarks, idx):
        base = idx * 2
        return float(landmarks[base]), float(landmarks[base + 1])

    def _center_of(self, landmarks, indices):
        sx = 0.0
        sy = 0.0
        count = 0
        try:
            for idx in indices:
                x, y = self._point(landmarks, idx)
                sx += x
                sy += y
                count += 1
        except Exception:
            return False, 0.0, 0.0
        if count <= 0:
            return False, 0.0, 0.0
        return True, sx / count, sy / count

    def _calc_open_ratio_by_points(self, landmarks, indices):
        """
        通过点集主方向宽度和法向高度计算开合比例。
        优点：不依赖眼睛点顺序，适合 CanMV 106 点左右眼 8 点轮廓。
        """
        try:
            n = len(indices)
            max_d2 = 0.0
            ax = ay = bx = by = 0.0
            for i in range(n):
                xi, yi = self._point(landmarks, indices[i])
                for j in range(i + 1, n):
                    xj, yj = self._point(landmarks, indices[j])
                    dx = xi - xj
                    dy = yi - yj
                    d2 = dx * dx + dy * dy
                    if d2 > max_d2:
                        max_d2 = d2
                        ax, ay, bx, by = xi, yi, xj, yj
            if max_d2 <= 1.0:
                return -1.0, 0.0
            width = _sqrt(max_d2)
            ux = (bx - ax) / width
            uy = (by - ay) / width
            nx = -uy
            ny = ux
            min_p = 1000000000.0
            max_p = -1000000000.0
            for idx in indices:
                x, y = self._point(landmarks, idx)
                p = (x - ax) * nx + (y - ay) * ny
                if p < min_p:
                    min_p = p
                if p > max_p:
                    max_p = p
            vertical = max_p - min_p
            if vertical < 0.0:
                vertical = 0.0
            return vertical / width, width
        except Exception as e:
            self._log(1, _ticks_ms(), "ratio_error", str(e))
            return -1.0, 0.0

    def _calc_head_pose_ratios(self, landmarks):
        try:
            # 两个瞳孔优先；异常时退化为两眼轮廓中心。
            if self.pupils is not None and len(self.pupils) >= 2:
                lx, ly = self._point(landmarks, self.pupils[0])
                rx, ry = self._point(landmarks, self.pupils[1])
            else:
                ok_l, lx, ly = self._center_of(landmarks, self.left_eye)
                ok_r, rx, ry = self._center_of(landmarks, self.right_eye)
                if not ok_l or not ok_r:
                    return False, 0.0, 0.0

            dx = rx - lx
            dy = ry - ly
            pupil_dist = _sqrt(dx * dx + dy * dy)
            if pupil_dist <= 2.0:
                return False, 0.0, 0.0
            eye_cx = (lx + rx) * 0.5
            eye_cy = (ly + ry) * 0.5

            ok_n, nx, ny = self._center_of(landmarks, self.nose_points)
            if not ok_n:
                return False, 0.0, 0.0
            yaw = (nx - eye_cx) / pupil_dist
            pitch = (ny - eye_cy) / pupil_dist
            return True, yaw, pitch
        except Exception as e:
            self._log(1, _ticks_ms(), "pose_error", str(e))
            return False, 0.0, 0.0

    def _estimate_face_scale(self, landmarks, face_box, left_width, right_width):
        if face_box is not None:
            try:
                w = abs(float(face_box[2]))
                h = abs(float(face_box[3]))
                if w > 2.0 and h > 2.0:
                    return (w + h) * 0.5
            except Exception:
                pass
        try:
            a = self.pupils[0] * 2
            b = self.pupils[1] * 2
            dx = float(landmarks[a]) - float(landmarks[b])
            dy = float(landmarks[a + 1]) - float(landmarks[b + 1])
            d = _sqrt(dx * dx + dy * dy)
            if d > 2.0:
                return d * 2.2
        except Exception:
            pass
        w = left_width if left_width > right_width else right_width
        if w > 2.0:
            return w * 4.0
        return 100.0

    def _calc_motion_ratio(self, landmarks, face_scale):
        if face_scale <= 1.0:
            face_scale = 100.0
        total = 0.0
        count = len(self.motion_points)
        ok = True
        for i in range(count):
            idx = self.motion_points[i] * 2
            try:
                x = float(landmarks[idx])
                y = float(landmarks[idx + 1])
            except Exception:
                ok = False
                x = 0.0
                y = 0.0
            p = i * 2
            if self._motion_prev_valid:
                dx = x - self._motion_prev[p]
                dy = y - self._motion_prev[p + 1]
                total += _sqrt(dx * dx + dy * dy)
            self._motion_prev[p] = x
            self._motion_prev[p + 1] = y
        if not ok or count <= 0:
            self._motion_prev_valid = False
            return 0.0
        if not self._motion_prev_valid:
            self._motion_prev_valid = True
            return 0.0
        return (total / count) / face_scale

    # ----------------------------- thresholds --------------------------------

    def _thresholds(self):
        if self.open_ear_baseline is None or self.open_ear_baseline <= 0.0:
            return self.close_ear_threshold, self.open_ear_threshold
        close_thr = self.open_ear_baseline * self.close_ratio
        if close_thr < self.min_close_ear:
            close_thr = self.min_close_ear
        if close_thr > self.max_close_ear:
            close_thr = self.max_close_ear
        open_thr = self.open_ear_baseline * self.open_ratio
        if open_thr < close_thr + self.hysteresis:
            open_thr = close_thr + self.hysteresis
        # 不要求回到 100% 基线，避免低帧率下睁眼帧不够导致卡住。
        if open_thr > self.open_ear_baseline * 0.98:
            open_thr = self.open_ear_baseline * 0.98
        return close_thr, open_thr

    def _update_open_baseline(self, ear):
        if ear <= 0.0:
            return
        if self.open_ear_baseline is None:
            self.open_ear_baseline = ear
            return
        # 向上跟踪快，向下跟踪慢：眼睛变大时快速适配，短暂眨眼不会把基线拉低。
        if ear > self.open_ear_baseline:
            alpha = 0.18
        else:
            alpha = 0.035
        self.open_ear_baseline = self.open_ear_baseline * (1.0 - alpha) + ear * alpha

    # ------------------------------- finishing -------------------------------

    def _finish_action(self, action, now):
        self.detected_action = action
        msg = "检测到{}，活体通过".format(self._action_name(action))
        return self._finish(self.STATUS_PASSED, msg, now)

    def _finish(self, status, message, now):
        self.status = status
        self.message = message
        self.terminal_ms = now
        self._log(1, now, status, message)
        return True

    # ------------------------------- logging ---------------------------------

    def _log(self, level, now, tag, msg):
        if self.log_level < level:
            return
        try:
            print("[活体][{}][{}ms] {}".format(tag, now, msg))
        except Exception:
            pass

    def collect_garbage_if_needed(self, min_free_bytes=256 * 1024):
        if gc is None or not hasattr(gc, "mem_free"):
            return
        try:
            if gc.mem_free() >= 0 and gc.mem_free() < min_free_bytes:
                self._log(1, _ticks_ms(), "gc", "mem_free low, collect")
                gc.collect()
        except Exception:
            pass


# ------------------------- optional worker-loop helper ------------------------

def liveness_worker_loop(mailbox, detector, shared_result=None, sleep_ms=20):
    while True:
        landmarks, face_box, frame_id, ts = mailbox.get_latest(only_new=True)
        if landmarks is not None:
            status, msg = detector.update(landmarks, ts, face_box=face_box, frame_id=frame_id)
            if shared_result is not None:
                shared_result["status"] = status
                shared_result["message"] = msg
                shared_result["blink_count"] = detector.blink_count
                shared_result["detected_action"] = detector.detected_action
        _sleep_ms(sleep_ms)


__all__ = ["LivenessDetector", "LandmarkMailbox", "liveness_worker_loop"]
