"""
===============================================================================
K230 KPU 人脸检测封装
===============================================================================
"""

import gc
import nncase_runtime as nn
import ulab.numpy as np
import aidemo


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return 0


class KpuFaceDetector:
    """KPU accelerated face detector wrapper.

    Input: numpy array from image2rgb888array, shape (1,3,H,W), dtype uint8.
    Returns: (det_boxes, landms), compatible with aidemo.face_det_post_process.
    """

    def __init__(self, kmodel_path, model_input_size=(320, 320), anchors=None,
                 rgb888p_size=(1280, 720), confidence=0.5, nms=0.2):
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.anchors = anchors
        self.rgb888p_size = list(rgb888p_size)
        self.confidence = confidence
        self.nms = nms

        self.kpu = nn.kpu()
        self.ai2d = nn.ai2d()
        self.ai2d_builder = None
        self._last_input_hw = None

        try:
            self.kpu.load_kmodel(self.kmodel_path)
        except Exception as e:
            print("[错误] KPU 模型加载失败: {}, {}".format(self.kmodel_path, e))
            raise

        # FIX: 预先绑定 KPU 输入 tensor，后续 ai2d 输出直接写入该 tensor，避免每帧重新分配。
        data = np.zeros((1, 3, self.model_input_size[1], self.model_input_size[0]), dtype=np.uint8)
        self.kpu.set_input_tensor(0, nn.from_numpy(data))

        self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
        self.ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.set_pad_param(True, [0, 0, 0, 0, 0, 0, 0, 0], 0, [104, 117, 123])

    def _calc_pad(self, input_w, input_h):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / input_w
        ratio_h = dst_h / input_h
        ratio = ratio_w if ratio_w < ratio_h else ratio_h
        new_w = int(ratio * input_w)
        new_h = int(ratio * input_h)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return [0, 0, 0, 0, top, bottom, left, right]

    def _ensure_builder(self, input_size_hw):
        # input_size_hw: (H, W)
        if self._last_input_hw != input_size_hw:
            # FIX: 不同通道/不同分辨率切换时必须重建 builder，否则 ai2d.run 可能崩溃。
            self.ai2d_builder = None
            self._last_input_hw = input_size_hw

        if self.ai2d_builder is not None:
            return

        in_shape = [1, 3, input_size_hw[0], input_size_hw[1]]
        out_shape = [1, 3, self.model_input_size[1], self.model_input_size[0]]
        pad_param = self._calc_pad(input_w=input_size_hw[1], input_h=input_size_hw[0])
        self.ai2d.set_pad_param(True, pad_param, 0, [104, 117, 123])
        try:
            self.ai2d_builder = self.ai2d.build(in_shape, out_shape)
        except Exception as e:
            print("[错误] ai2d build 失败: {}".format(e))
            raise

    def run(self, input_np):
        """Run one detection. Return (det_boxes, landms)."""
        if input_np is None:
            return [], []
        try:
            h = int(input_np.shape[2])
            w = int(input_np.shape[3])
            self._ensure_builder((h, w))

            ai2d_input_tensor = nn.from_numpy(input_np)
            ai2d_out = self.kpu.get_input_tensor(0)
            self.ai2d_builder.run(ai2d_input_tensor, ai2d_out)

            self.kpu.run()

            results = []
            for i in range(self.kpu.outputs_size()):
                results.append(self.kpu.get_output_tensor(i).to_numpy())

            res = aidemo.face_det_post_process(
                self.confidence, self.nms, self.model_input_size[0],
                self.anchors, self.rgb888p_size, results
            )
            if res is None or _safe_len(res) == 0:
                return [], []
            if _safe_len(res) >= 2:
                return res[0], res[1]
            return [], []
        except MemoryError:
            print("[KPU] 内存不足，触发 GC 后跳过本帧")
            gc.collect()
            return [], []
        except Exception as e:
            print("[KPU] 检测异常: {}".format(e))
            return [], []

    def deinit(self):
        try:
            self.ai2d_builder = None
            del self.ai2d_builder
        except Exception:
            pass
        try:
            del self.ai2d
        except Exception:
            pass
        try:
            del self.kpu
        except Exception:
            pass
        try:
            gc.collect()
            nn.shrink_memory_pool()
        except Exception:
            pass


__all__ = ["KpuFaceDetector"]
