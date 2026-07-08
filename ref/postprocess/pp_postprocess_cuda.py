"""
CUB-based CUDA post-processor for PointPillars raw-head output.

Compiles pp_post_cuda.cu via torch.utils.cpp_extension on first import (~30s).
Subsequent runs load the cached .so — no recompile.

Usage:
    from pp_postprocess_cuda import PPPostProcessorCUDA
    post = PPPostProcessorCUDA('path/to/backbone_rawhead_fp32.onnx')
    # cls/box/dir are CUDA torch.Tensor or numpy arrays
    boxes, scores, class_ids = post(cls_preds, box_preds, dir_preds)

Expected latency: ~0.4 ms  (vs ~2.8 ms for PyTorch nonzero version)
"""

from __future__ import annotations
import os
import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_kernel = None


def _load_kernel():
    global _kernel
    if _kernel is None:
        from torch.utils.cpp_extension import load
        _kernel = load(
            name="pp_post_cuda",
            sources=[os.path.join(_SRC_DIR, "pp_post_cuda.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                # Match your GPU — uncomment as needed:
                # "-gencode", "arch=compute_89,code=sm_89",  # Ada  (RTX 40xx)
                # "-gencode", "arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
                # "-gencode", "arch=compute_80,code=sm_80",  # A100
            ],
            verbose=False,
        )
    return _kernel


def _extract_proto(onnx_path: str) -> np.ndarray:
    import onnx
    import onnx.numpy_helper as nph
    m = onnx.load(onnx_path)
    for init in m.graph.initializer:
        arr = nph.to_array(init)
        if arr.size == 266:
            return arr.reshape(38, 7).astype(np.float32)
    raise ValueError(
        f"Proto table not found in {onnx_path}. "
        "Pass a decoded-head ONNX (one with [500,11] output) or a [38,7] numpy array."
    )


class PPPostProcessorCUDA:
    """
    Fast CUB-based post-processor. Requires CUDA + CUDA toolkit for compilation.

    Parameters
    ----------
    proto_source : str | np.ndarray
        Path to a decoded-head ONNX (proto extracted automatically), or
        a [38, 7] float32 array with anchor prototypes [cx0 cy0 cz w l h yaw0].
    score_threshold : float
        Pre-NMS confidence threshold (sigmoid).
    topk : int
        Maximum detections returned.
    """

    def __init__(
        self,
        proto_source: "str | np.ndarray",
        score_threshold: float = 0.1,
        topk: int = 500,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.topk = int(topk)

        if isinstance(proto_source, str):
            proto_np = _extract_proto(proto_source)
        else:
            proto_np = np.asarray(proto_source, dtype=np.float32)

        assert proto_np.shape == (38, 7), f"Expected proto [38,7], got {proto_np.shape}"
        self._proto = torch.from_numpy(proto_np).cuda().contiguous()

        # Trigger JIT compile at construction time, not at first inference
        _load_kernel()

    def __call__(
        self,
        cls_preds: "torch.Tensor | np.ndarray",  # [1, 722, H, W]
        box_preds: "torch.Tensor | np.ndarray",  # [1, 266, H, W]
        dir_preds: "torch.Tensor | np.ndarray",  # [1,  76, H, W]
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Returns
        -------
        boxes     : [K, 7]  float32  x y z w l h yaw
        scores    : [K]     float32  sigmoid confidence
        class_ids : [K]     int32    0-indexed
        """
        def _cuda(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return x.float().cuda().contiguous()

        cls_preds = _cuda(cls_preds)
        box_preds = _cuda(box_preds)
        dir_preds = _cuda(dir_preds)

        return _load_kernel().forward(
            cls_preds, box_preds, dir_preds,
            self._proto,
            self.score_threshold,
            self.topk,
        )
