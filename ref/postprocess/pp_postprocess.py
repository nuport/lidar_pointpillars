"""
Custom GPU post-processor for PointPillars raw-head TRT output.

Replaces the TRT-internal TopK+decode (5.9 ms) with a CUDA-friendly pipeline:
  1. sigmoid + argmax over classes  →  scores [N], class_ids [N]     (N = H*W*n_anc ≈ 2.5M)
  2. score threshold                →  candidate_indices (~25K)
  3. topk on ~25K candidates        →  top-K global indices (~0.1 ms)
  4. on-the-fly anchor decode       →  reconstructed cx/cy from grid position
  5. box regression decode          →  decoded 3-D boxes

Input layout (raw-head TRT output):
  cls_preds : [1, n_cls*n_anc, H, W]   e.g. [1, 722, 200, 328]
  box_preds : [1, 7*n_anc,     H, W]   e.g. [1, 266, 200, 328]
  dir_preds : [1, 2*n_anc,     H, W]   e.g. [1,  76, 200, 328]

Output:
  boxes    : [K, 7]   x y z w l h yaw
  scores   : [K]
  class_ids: [K]      0-indexed
"""

from __future__ import annotations
import math
import numpy as np
import torch


# ── model-specific constants (pointpillar_combined_v2, ep20) ─────────────────
# point cloud range: x[-60, 71.2], y[-40, 40]
# voxel size: 0.2 × 0.2 × 10 → backbone stride 2 → grid cell = 0.4 m
_PC_RANGE_X_MIN = -60.0
_PC_RANGE_Y_MIN = -40.0
_VOXEL_SIZE     = 0.2
_BACKBONE_STRIDE = 2
_GRID_STEP      = _VOXEL_SIZE * _BACKBONE_STRIDE   # 0.4 m per cell
_W_FEAT         = 328
_H_FEAT         = 200
_N_ANC          = 38    # anchor types (n_classes * n_rotations summed across all sizes)
_N_CLS          = 19
# ─────────────────────────────────────────────────────────────────────────────


class PPPostProcessor:
    """
    Threshold-first GPU post-processor.  Call as a callable after raw-head TRT.

    Parameters
    ----------
    proto_source : str | np.ndarray | torch.Tensor
        Path to a decoded-head ONNX (proto extracted automatically), or a
        [38, 7] array/tensor with anchor prototype rows:
        [cx0, cy0, cz, w, l, h, yaw0]  (values at grid cell (0, 0)).
    score_threshold : float
        Pre-NMS confidence threshold (sigmoid).  Lower = more candidates.
    topk : int
        Max detections returned after threshold + topk step.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        proto_source: "str | np.ndarray | torch.Tensor",
        score_threshold: float = 0.1,
        topk: int = 500,
        device: str = "cuda",
    ) -> None:
        self.score_threshold = score_threshold
        self.topk = topk
        self.device = torch.device(device)

        if isinstance(proto_source, str):
            proto_np = _extract_proto_from_onnx(proto_source)
        elif isinstance(proto_source, np.ndarray):
            proto_np = proto_source
        else:
            proto_np = proto_source.cpu().numpy()

        assert proto_np.shape == (38, 7), f"Expected proto [38,7], got {proto_np.shape}"
        self.proto = torch.from_numpy(proto_np).float().to(self.device)  # [38, 7]

    # ------------------------------------------------------------------
    def __call__(
        self,
        cls_preds: "np.ndarray | torch.Tensor",
        box_preds: "np.ndarray | torch.Tensor",
        dir_preds: "np.ndarray | torch.Tensor",
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Returns
        -------
        boxes     : [K, 7]   float32  x y z w l h yaw
        scores    : [K]      float32  (post-sigmoid)
        class_ids : [K]      int64    0-indexed
        """
        dev = self.device

        def _t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(dev)
            return x.to(dev)

        cls_preds = _t(cls_preds).float()   # [1, 722, H, W]
        box_preds = _t(box_preds).float()   # [1, 266, H, W]
        dir_preds = _t(dir_preds).float()   # [1,  76, H, W]

        H = cls_preds.shape[2]
        W = cls_preds.shape[3]
        n_anc = _N_ANC
        n_cls = _N_CLS
        N_total = H * W * n_anc            # ≈ 2.5 M

        # ── 1. sigmoid + argmax ──────────────────────────────────────────────
        # Reshape to [N_total, n_cls]
        cls_flat = (
            cls_preds.permute(0, 2, 3, 1)   # [1, H, W, 722]
            .contiguous()
            .view(H * W, n_anc, n_cls)
            .view(N_total, n_cls)
        )
        scores_raw, class_ids = torch.sigmoid(cls_flat).max(dim=1)  # [N_total]

        # ── 2. threshold → candidate compaction ─────────────────────────────
        cand_mask = scores_raw > self.score_threshold
        cand_idx = cand_mask.nonzero(as_tuple=False).squeeze(1)  # [M]

        if cand_idx.numel() == 0:
            empty_f = torch.zeros(0, 7, device=dev)
            empty_s = torch.zeros(0, device=dev)
            empty_c = torch.zeros(0, dtype=torch.long, device=dev)
            return empty_f, empty_s, empty_c

        # ── 3. topk on ~M candidates ─────────────────────────────────────────
        cand_scores = scores_raw[cand_idx]
        k = min(self.topk, cand_idx.numel())
        topk_scores, topk_local = torch.topk(cand_scores, k)
        sel_idx = cand_idx[topk_local]       # global anchor indices [K]
        sel_class = class_ids[sel_idx]       # [K]  0-indexed

        # ── 4. on-the-fly anchor decode ───────────────────────────────────────
        anc_type  = sel_idx % n_anc          # [K]  which prototype row
        grid_pos  = sel_idx // n_anc         # [K]  flattened (row * W + col)
        grid_col  = (grid_pos % W).float()   # x direction
        grid_row  = (grid_pos // W).float()  # y direction

        sel_proto = self.proto[anc_type]     # [K, 7]  cx0 cy0 cz w l h yaw0
        sel_cx = sel_proto[:, 0] + grid_col * _GRID_STEP
        sel_cy = sel_proto[:, 1] + grid_row * _GRID_STEP

        # ── 5. box regression decode ─────────────────────────────────────────
        box_flat = (
            box_preds.permute(0, 2, 3, 1)   # [1, H, W, 266]
            .contiguous()
            .view(H * W, n_anc, 7)
            .view(N_total, 7)
        )
        sel_box = box_flat[sel_idx]          # [K, 7]

        anc_w = sel_proto[:, 3]
        anc_l = sel_proto[:, 4]
        anc_h = sel_proto[:, 5]
        anc_cz = sel_proto[:, 2]
        anc_yaw = sel_proto[:, 6]

        diag  = torch.sqrt(anc_w ** 2 + anc_l ** 2)
        pred_x = sel_box[:, 0] * diag + sel_cx
        pred_y = sel_box[:, 1] * diag + sel_cy
        pred_z = sel_box[:, 2] * anc_h + anc_cz
        pred_w = torch.exp(sel_box[:, 3]) * anc_w
        pred_l = torch.exp(sel_box[:, 4]) * anc_l
        pred_h = torch.exp(sel_box[:, 5]) * anc_h
        pred_yaw = sel_box[:, 6] + anc_yaw

        # ── 6. direction correction ───────────────────────────────────────────
        dir_flat = (
            dir_preds.permute(0, 2, 3, 1)   # [1, H, W, 76]
            .contiguous()
            .view(H * W, n_anc, 2)
            .view(N_total, 2)
        )
        sel_dir = dir_flat[sel_idx]          # [K, 2]
        dir_label = (sel_dir[:, 1] > sel_dir[:, 0]).float()  # 0 or 1

        pi = math.pi
        # limit_period(val, offset=0.5, period=pi): maps to [-pi/2, pi/2]
        yaw_norm = pred_yaw - torch.floor(pred_yaw / pi + 0.5) * pi
        # shift by pi if dir=1 (forward hemisphere)
        pred_yaw_final = yaw_norm + dir_label * pi

        boxes = torch.stack(
            [pred_x, pred_y, pred_z, pred_w, pred_l, pred_h, pred_yaw_final],
            dim=1,
        )  # [K, 7]

        return boxes, topk_scores, sel_class.long()


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_proto_from_onnx(onnx_path: str) -> np.ndarray:
    """Pull the 38×7 prototype table from a decoded-head backbone ONNX."""
    import onnx
    import onnx.numpy_helper as nph

    m = onnx.load(onnx_path)
    for init in m.graph.initializer:
        arr = nph.to_array(init)
        if arr.size == 266:
            return arr.reshape(38, 7).astype(np.float32)
    raise ValueError(
        f"Could not find 38×7 proto initializer in {onnx_path}. "
        "Pass a decoded-head ONNX or a [38,7] numpy array directly."
    )


def load_proto_from_decoded_head_onnx(onnx_path: str) -> np.ndarray:
    """Public helper: load proto [38,7] from a decoded-head ONNX."""
    return _extract_proto_from_onnx(onnx_path)
