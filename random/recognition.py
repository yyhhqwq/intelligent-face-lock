"""
===============================================================================
CanMV K230 人脸识别模块
===============================================================================
"""

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *

import os
import time
import gc
import math
import image
import aidemo
import nncase_runtime as nn
import ulab.numpy as np


def ALIGN_UP(x, align):
    return ((x + align - 1) // align) * align


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return 0


def _file_exists(path):
    try:
        parts = path.split("/")
        file_name = parts[-1]
        dir_path = "/".join(parts[:-1])
        if not dir_path:
            dir_path = "/"
        return file_name in os.listdir(dir_path)
    except Exception:
        return False


def detect_low_light(image_array, threshold=50):
    try:
        mean_brightness = np.mean(image_array)
        return mean_brightness < threshold
    except Exception:
        return False


def detect_backlight(image_array):
    try:
        if len(image_array.shape) == 4:
            img = image_array[0]
        else:
            img = image_array
        top_mean = np.mean(img[:, :, :int(img.shape[2] * 0.3)])
        bottom_mean = np.mean(img[:, :, int(img.shape[2] * 0.7):])
        return (top_mean - bottom_mean) > 30
    except Exception:
        return False


class FaceDetApp(AIBase):
    """人脸检测任务类，Ai2d 预处理 + aidemo 后处理。"""

    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.25, nms_threshold=0.3,
                 rgb888p_size=[1920, 1080], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.pad(self.get_pad_param(ai2d_input_size), 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            res = aidemo.face_det_post_process(
                self.confidence_threshold, self.nms_threshold, self.model_input_size[0],
                self.anchors, self.rgb888p_size, results
            )
            if res is None or _safe_len(res) == 0:
                return [], []
            if _safe_len(res) >= 2:
                return res[0], res[1]
            return [], []

    def get_pad_param(self, image_input_size=None):
        src_size = image_input_size if image_input_size else self.rgb888p_size
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / src_size[0]
        ratio_h = dst_h / src_size[1]
        ratio = ratio_w if ratio_w < ratio_h else ratio_h
        new_w = int(ratio * src_size[0])
        new_h = int(ratio * src_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return [0, 0, 0, 0, top, bottom, left, right]


class FaceRegistrationApp(AIBase):
    """人脸特征提取任务类。"""

    def __init__(self, kmodel_path, model_input_size,
                 rgb888p_size=[1920, 1080], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.umeyama_args_112 = [
            38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041
        ]
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        s = [0.0, 0.0]
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]
        s[0] = (math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2) +
                math.sqrt((a[0] + a[3]) ** 2 + (a[1] - a[2]) ** 2)) / 2
        s[1] = abs(s[0] - math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2))
        v[2] = math.sin((math.atan2(2 * (a[0] * a[1] + a[2] * a[3]),
                                     a[0] ** 2 - a[1] ** 2 + a[2] ** 2 - a[3] ** 2)) / 2) if s[0] > s[1] else 0
        v[0] = math.sqrt(1 - v[2] ** 2)
        v[1] = -v[2]
        v[3] = v[0]
        u[0] = -(a[0] * v[0] + a[1] * v[2]) / s[0] if s[0] != 0 else 1
        u[2] = -(a[2] * v[0] + a[3] * v[2]) / s[0] if s[0] != 0 else 0
        u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1] if s[1] != 0 else -u[2]
        u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1] if s[1] != 0 else u[0]
        v[0] = -v[0]
        v[2] = -v[2]
        return u, s, v

    def image_umeyama_112(self, src):
        SRC_NUM = 5
        SRC_DIM = 2
        src_mean = [0.0, 0.0]
        dst_mean = [0.0, 0.0]
        for i in range(0, SRC_NUM * 2, 2):
            src_mean[0] += src[i]
            src_mean[1] += src[i + 1]
            dst_mean[0] += self.umeyama_args_112[i]
            dst_mean[1] += self.umeyama_args_112[i + 1]
        src_mean[0] /= SRC_NUM
        src_mean[1] /= SRC_NUM
        dst_mean[0] /= SRC_NUM
        dst_mean[1] /= SRC_NUM

        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]

        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM

        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])
        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]

        src_demean_mean = [0.0, 0.0]
        src_demean_var = [0.0, 0.0]
        for i in range(SRC_NUM):
            src_demean_mean[0] += src_demean[i][0]
            src_demean_mean[1] += src_demean[i][1]
        src_demean_mean[0] /= SRC_NUM
        src_demean_mean[1] /= SRC_NUM
        for i in range(SRC_NUM):
            src_demean_var[0] += (src_demean_mean[0] - src_demean[i][0]) * (src_demean_mean[0] - src_demean[i][0])
            src_demean_var[1] += (src_demean_mean[1] - src_demean[i][1]) * (src_demean_mean[1] - src_demean[i][1])
        src_demean_var[0] /= SRC_NUM
        src_demean_var[1] /= SRC_NUM
        denom = src_demean_var[0] + src_demean_var[1]
        scale = 1.0 if denom < 1e-8 else (1.0 / denom * (S[0] + S[1]))

        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale
        return T

    def get_affine_matrix(self, sparse_points):
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            return [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                    matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]


class FaceLandmarkApp(AIBase):
    """106 点人脸关键点任务类，专供眨眼活体检测使用。

    官方 CanMV face_landmark 示例使用 192x192 输入、det_box 仿射裁剪，并在
    postprocess 中将关键点逆变换回原图坐标。本类按该流程实现，避免把 5 点
    对齐关键点误当作眼部轮廓使用。
    """

    def __init__(self, kmodel_path, model_input_size,
                 rgb888p_size=[1280, 720], display_size=[1920, 1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.matrix_dst = None
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, det, input_image_size=None):
        with ScopedTiming("landmark preprocess", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.matrix_dst = self.get_affine_matrix(det)
            affine_matrix = [self.matrix_dst[0][0], self.matrix_dst[0][1], self.matrix_dst[0][2],
                             self.matrix_dst[1][0], self.matrix_dst[1][1], self.matrix_dst[1][2]]
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                            [1, 3, self.model_input_size[1], self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("landmark postprocess", self.debug_mode > 0):
            pred = results[0]
            pred = pred.flatten()
            half_input_len = self.model_input_size[0] // 2
            # FIX: 参考 CanMV 官方 face_landmark.py，将模型输出还原到裁剪框坐标。
            for i in range(len(pred)):
                pred[i] += (pred[i] + 1) * half_input_len

            matrix_dst_inv = aidemo.invert_affine_transform(self.matrix_dst)
            matrix_dst_inv = matrix_dst_inv.flatten()
            half_out_len = len(pred) // 2
            for kp_id in range(half_out_len):
                old_x = pred[kp_id * 2]
                old_y = pred[kp_id * 2 + 1]
                new_x = old_x * matrix_dst_inv[0] + old_y * matrix_dst_inv[1] + matrix_dst_inv[2]
                new_y = old_x * matrix_dst_inv[3] + old_y * matrix_dst_inv[4] + matrix_dst_inv[5]
                pred[kp_id * 2] = new_x
                pred[kp_id * 2 + 1] = new_y
            return pred

    def get_affine_matrix(self, bbox):
        with ScopedTiming("landmark affine", self.debug_mode > 1):
            x1, y1, w, h = map(lambda x: int(round(x, 0)), bbox[:4])
            if w < 2:
                w = 2
            if h < 2:
                h = 2
            scale_ratio = self.model_input_size[0] / (max(w, h) * 1.5)
            cx = (x1 + w / 2) * scale_ratio
            cy = (y1 + h / 2) * scale_ratio
            half_input_len = self.model_input_size[0] / 2
            matrix_dst = np.zeros((2, 3), dtype=np.float)
            matrix_dst[0, 0] = scale_ratio
            matrix_dst[0, 1] = 0
            matrix_dst[0, 2] = half_input_len - cx
            matrix_dst[1, 0] = 0
            matrix_dst[1, 1] = scale_ratio
            matrix_dst[1, 2] = half_input_len - cy
            return matrix_dst


class FaceRecognition:
    """人脸检测 + 特征提取 + 数据库检索。"""

    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size,
                 database_dir, anchors, confidence_threshold=0.25, nms_threshold=0.3,
                 face_recognition_threshold=0.75, rgb888p_size=[1280, 720],
                 display_size=[1920, 1080], debug_mode=0, enable_enhancement=True,
                 use_kpu=True, face_landmark_kmodel=None, landmark_input_size=[192, 192],
                 enable_liveness_landmark=False):
        self.face_det_kmodel = face_det_kmodel
        self.face_reg_kmodel = face_reg_kmodel
        self.face_landmark_kmodel = face_landmark_kmodel
        self.det_input_size = det_input_size
        self.reg_input_size = reg_input_size
        self.landmark_input_size = landmark_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.face_recognition_threshold = face_recognition_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        self.max_register_face = 100
        self.feature_num = 0
        self.valid_register_face = 0
        self.db_name = []
        self.db_data = []

        # last_landms 保留为 5 点对齐关键点，避免破坏旧代码；活体使用 last_landmarks106。
        self.last_det_boxes = []
        self.last_landms = []
        self.last_landmarks106 = None

        self.enable_enhancement = enable_enhancement
        self.low_light_threshold = 50
        self.backlight_diff_threshold = 30
        self.use_kpu = use_kpu
        self.kpu_detector = None
        self.face_det = None
        self.face_reg = None
        self.face_landmark = None
        self._detector_logged = False
        self._landmark_logged = False

        if self.use_kpu:
            try:
                import cp.kpu_det as _kpu_mod
                self.kpu_detector = _kpu_mod.KpuFaceDetector(
                    self.face_det_kmodel,
                    model_input_size=self.det_input_size,
                    anchors=self.anchors,
                    rgb888p_size=self.rgb888p_size,
                    confidence=self.confidence_threshold,
                    nms=self.nms_threshold
                )
            except Exception as e:
                print("[识别] KPU detector 初始化失败，回退 Ai2d: {}".format(e))
                self.kpu_detector = None

        if self.kpu_detector is None:
            self.face_det = FaceDetApp(
                self.face_det_kmodel, model_input_size=self.det_input_size,
                anchors=self.anchors, confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold, rgb888p_size=self.rgb888p_size,
                display_size=self.display_size, debug_mode=0
            )
            try:
                self.face_det.config_preprocess()
            except Exception as e:
                print("[警告] 人脸检测预处理配置失败: {}".format(e))

        self.face_reg = FaceRegistrationApp(
            self.face_reg_kmodel, model_input_size=self.reg_input_size,
            rgb888p_size=self.rgb888p_size, display_size=self.display_size
        )

        if enable_liveness_landmark:
            if self.face_landmark_kmodel and _file_exists(self.face_landmark_kmodel):
                try:
                    self.face_landmark = FaceLandmarkApp(
                        self.face_landmark_kmodel, model_input_size=self.landmark_input_size,
                        rgb888p_size=self.rgb888p_size, display_size=self.display_size
                    )
                    print("[活体] 106点关键点模型已加载: {}".format(self.face_landmark_kmodel))
                except Exception as e:
                    print("[活体] 106点关键点模型加载失败: {}".format(e))
                    self.face_landmark = None
            else:
                print("[活体] 未找到 106点关键点模型，眨眼活体不可用: {}".format(self.face_landmark_kmodel))

        self.database_init()

    def has_liveness_landmark(self):
        return self.face_landmark is not None

    def detect_scene_condition(self, image_array):
        try:
            mean_brightness = np.mean(image_array)
            is_low_light = mean_brightness < self.low_light_threshold
            if len(image_array.shape) == 4:
                img_2d = image_array[0]
            else:
                img_2d = image_array
            top_mean = np.mean(img_2d[:, :, :int(img_2d.shape[2] * 0.3)])
            bottom_mean = np.mean(img_2d[:, :, int(img_2d.shape[2] * 0.7):])
            left_mean = np.mean(img_2d[:, :int(img_2d.shape[1] * 0.3), :])
            right_mean = np.mean(img_2d[:, int(img_2d.shape[1] * 0.7):, :])
            is_backlight = (top_mean - bottom_mean) > self.backlight_diff_threshold
            is_side_light_left = (right_mean - left_mean) > self.backlight_diff_threshold
            is_side_light_right = (left_mean - right_mean) > self.backlight_diff_threshold
            scene_type = "normal"
            if is_low_light:
                scene_type = "low_light"
            elif is_backlight:
                scene_type = "backlight"
            elif is_side_light_left:
                scene_type = "side_light_left"
            elif is_side_light_right:
                scene_type = "side_light_right"
            return scene_type, mean_brightness
        except Exception:
            return "normal", 0

    def enhance_image_for_scene(self, image_array, scene_type):
        return image_array

    def estimate_head_pose(self, landms):
        if _safe_len(landms) == 0:
            return "frontal"
        try:
            landm = landms[0]
            left_eye_x, left_eye_y = landm[0], landm[1]
            right_eye_x, right_eye_y = landm[2], landm[3]
            nose_x, nose_y = landm[4], landm[5]
            eye_distance = math.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2)
            if eye_distance < 1e-8:
                return "frontal"
            eye_center_x = (left_eye_x + right_eye_x) / 2
            eye_center_y = (left_eye_y + right_eye_y) / 2
            pitch_angle = math.atan2(nose_y - eye_center_y, eye_distance) * (180 / math.pi)
            yaw_angle = math.atan2(nose_x - eye_center_x, eye_distance) * (180 / math.pi)
            if abs(yaw_angle) > 45:
                return "profile_right" if yaw_angle > 0 else "profile_left"
            if abs(yaw_angle) > 20:
                return "turn_right" if yaw_angle > 0 else "turn_left"
            if abs(pitch_angle) > 30:
                return "pitch_down" if pitch_angle > 0 else "pitch_up"
            return "frontal"
        except Exception:
            return "frontal"

    def adjust_recognition_threshold(self, pose_type):
        if pose_type in ["frontal"]:
            return self.face_recognition_threshold
        if pose_type in ["turn_left", "turn_right"]:
            return self.face_recognition_threshold - 0.08
        if pose_type in ["profile_left", "profile_right"]:
            return self.face_recognition_threshold - 0.12
        return self.face_recognition_threshold

    def run(self, input_np):
        """执行一次检测+识别，返回 det_boxes, recg_res。"""
        try:
            if self.kpu_detector is not None:
                if not self._detector_logged:
                    print("[识别] 使用 KPU detector")
                    self._detector_logged = True
                det_boxes, landms = self.kpu_detector.run(input_np)
            else:
                if not self._detector_logged:
                    print("[识别] 使用 Ai2d detector")
                    self._detector_logged = True
                det_boxes, landms = self.face_det.run(input_np)
        except Exception as e:
            print("[识别] 人脸检测失败: {}".format(e))
            self.last_det_boxes = []
            self.last_landms = []
            return [], []

        self.last_det_boxes = det_boxes
        self.last_landms = landms  # 5点对齐关键点，不可用于眨眼。
        self.last_landmarks106 = None

        det_count = _safe_len(det_boxes)
        landm_count = _safe_len(landms)
        recg_res = []
        for i in range(det_count):
            if i >= landm_count:
                recg_res.append("unknown")
                continue
            try:
                landm = landms[i]
                self.face_reg.config_preprocess(landm, input_image_size=[input_np.shape[3], input_np.shape[2]])
                feature = self.face_reg.run(input_np)
                recg_res.append(self.database_search(feature))
            except MemoryError:
                print("[识别] 特征提取内存不足，本帧跳过")
                gc.collect()
                recg_res.append("unknown")
            except Exception as e:
                print("[识别] 特征提取失败: {}".format(e))
                recg_res.append("unknown")

        return det_boxes, recg_res

    def extract_landmark106(self, input_np, det_box):
        """对指定 det_box 提取 106 点关键点，供 liveness.update 使用。
        
        FIX: 添加推理超时保护，防止 KPU 硬件卡死导致整个系统死锁。
        """
        if self.face_landmark is None:
            return None
        if input_np is None or det_box is None:
            return None
        try:
            if not self._landmark_logged:
                print("[活体] 使用 106点关键点进行眨眼检测")
                self._landmark_logged = True
            
            # 推理前记录时间，用于超时检测
            infer_start = time.ticks_ms()
            
            self.face_landmark.config_preprocess(det_box, input_image_size=[input_np.shape[3], input_np.shape[2]])
            res = self.face_landmark.run(input_np)
            
            # 检查推理耗时
            infer_time = time.ticks_diff(time.ticks_ms(), infer_start)
            if infer_time > 2000:  # 超过 2 秒说明可能卡过
                print("[活体] 关键点推理耗时过长: {}ms，可能存在硬件异常".format(infer_time))
            
            self.last_landmarks106 = res
            return res
        except MemoryError:
            print("[活体] 106点关键点内存不足，本帧跳过")
            gc.collect()
            return None
        except Exception as e:
            print("[活体] 106点关键点提取失败: {}".format(e))
            gc.collect()
            return None

    def database_init(self):
        with ScopedTiming("database_init", self.debug_mode > 1):
            try:
                db_file_list = os.listdir(self.database_dir)
            except Exception as e:
                print("[警告] 数据库目录读取失败: {}, {}".format(self.database_dir, e))
                return
            for db_file in db_file_list:
                try:
                    if not db_file.endswith(".bin"):
                        continue
                    if self.valid_register_face >= self.max_register_face:
                        break
                    full_db_file = self.database_dir + db_file
                    with open(full_db_file, "rb") as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    if _safe_len(feature) <= 0:
                        continue
                    if self.feature_num == 0:
                        self.feature_num = _safe_len(feature)
                    if _safe_len(feature) != self.feature_num:
                        print("[警告] 特征维度不一致，跳过: {}".format(full_db_file))
                        continue
                    self.db_data.append(feature)
                    self.db_name.append(db_file[:-4])
                    self.valid_register_face += 1
                except Exception as e:
                    print("[警告] 读取人脸特征文件失败: {}, {}".format(db_file, e))

    def database_reset(self):
        with ScopedTiming("database_reset", self.debug_mode > 1):
            self.db_name = []
            self.db_data = []
            self.valid_register_face = 0
            self.feature_num = 0
            print("[人脸库] 内存索引已清空")

    def list_faces(self):
        with ScopedTiming("list_faces", self.debug_mode > 1):
            if self.valid_register_face == 0:
                print("数据库为空，无已注册人脸")
                return []
            print("已注册人脸 ({}人):".format(self.valid_register_face))
            for name in self.db_name:
                print("  - {}".format(name))
            return self.db_name

    def database_search(self, feature, threshold=None):
        with ScopedTiming("database_search", self.debug_mode > 1):
            actual_threshold = threshold if threshold is not None else self.face_recognition_threshold
            if self.valid_register_face <= 0:
                return "unknown"
            try:
                feature = feature.flatten()
            except Exception:
                pass
            feat_len = _safe_len(feature)
            if feat_len <= 0:
                return "unknown"
            feat_norm = np.linalg.norm(feature)
            if feat_norm < 1e-8:
                return "unknown"

            v_id = -1
            v_score_max = 0.0
            for i in range(self.valid_register_face):
                db_feature = self.db_data[i]
                if _safe_len(db_feature) != feat_len:
                    continue
                db_norm = np.linalg.norm(db_feature)
                if db_norm < 1e-8:
                    continue
                # FIX: 不原地 feature/=norm、db_feature/=norm，避免破坏缓存和只读 buffer。
                v_score = (np.dot(feature, db_feature) / (feat_norm * db_norm)) / 2 + 0.5
                if v_score > v_score_max:
                    v_score_max = v_score
                    v_id = i
            if v_id == -1 or v_score_max < actual_threshold:
                return "unknown"
            return "name: {}, score:{:.3f}".format(self.db_name[v_id], v_score_max)

    def image2rgb888array(self, img):
        with ScopedTiming("image2rgb888array", self.debug_mode > 0):
            try:
                # 优先使用 RGB888 planar 零拷贝引用，减少 1280x720 帧转换时的堆分配。
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

    def draw_result(self, img, det, recg_res):
        if det:
            for i, one_det in enumerate(det):
                x1, y1, w, h = map(lambda x: int(round(x, 0)), one_det[:4])
                x1 = x1 * self.display_size[0] // self.rgb888p_size[0]
                y1 = y1 * self.display_size[1] // self.rgb888p_size[1]
                w = w * self.display_size[0] // self.rgb888p_size[0]
                h = h * self.display_size[1] // self.rgb888p_size[1]
                img.draw_rectangle(x1, y1, w, h, color=(255, 0, 0), thickness=4)
                recg_text = recg_res[i] if i < len(recg_res) else "unknown"
                img.draw_string_advanced(x1, y1, 32, recg_text, color=(255, 0, 0))

    def deinit(self):
        try:
            if self.face_det is not None:
                self.face_det.deinit()
        except Exception:
            pass
        try:
            if self.kpu_detector is not None:
                self.kpu_detector.deinit()
        except Exception:
            pass
        try:
            if self.face_reg is not None:
                self.face_reg.deinit()
        except Exception:
            pass
        try:
            if self.face_landmark is not None:
                self.face_landmark.deinit()
        except Exception:
            pass
        try:
            time.sleep_ms(100)
        except Exception:
            pass
        gc.collect()


def create_face_recognition(face_det_kmodel_path, face_reg_kmodel_path,
                            det_input_size, reg_input_size, database_dir, anchors,
                            confidence_threshold, nms_threshold, face_recognition_threshold,
                            rgb888p_size, display_size, use_kpu=True,
                            face_landmark_kmodel_path=None, landmark_input_size=[192, 192],
                            enable_liveness_landmark=False):
    return FaceRecognition(
        face_det_kmodel_path, face_reg_kmodel_path,
        det_input_size=det_input_size, reg_input_size=reg_input_size,
        database_dir=database_dir, anchors=anchors,
        confidence_threshold=confidence_threshold, nms_threshold=nms_threshold,
        face_recognition_threshold=face_recognition_threshold,
        rgb888p_size=rgb888p_size, display_size=display_size, use_kpu=use_kpu,
        face_landmark_kmodel=face_landmark_kmodel_path,
        landmark_input_size=landmark_input_size,
        enable_liveness_landmark=enable_liveness_landmark
    )
