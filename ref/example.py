"""
PointPillars INT8 end-to-end inference example.

Pipeline:
  1. Encoder FP16 TRT  —  raw point cloud → BEV feature map  (0.43 ms)
  2. Raw backbone INT8 TRT  —  BEV → cls_preds / box_preds / dir_preds  (1.62 ms)
  3. Post-processor (CUDA or PyTorch)  —  decode top-K boxes  (~0.4–2.8 ms)
  ─────────────────────────────────────────────────────────────────────────
  Total: ~2.5–4.9 ms  (RTX 4080 SUPER, batch=1)

This script uses ONNX Runtime for inference (no TRT Python bindings required).
For production, replace ORT sessions with your TRT inference wrapper.

Usage:
    python example.py --bev_npz /path/to/frame.npz [--use_cub]
"""

import argparse
import time
import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'postprocess'))

ENCODER_FP32_ONNX  = os.path.join(_HERE, 'onnx', 'encoder_fp32.onnx')
BACKBONE_FP32_ONNX = os.path.join(_HERE, 'onnx', 'backbone_rawhead_fp32.onnx')
BACKBONE_INT8_ONNX = os.path.join(_HERE, 'onnx', 'backbone_rawhead_int8_qdq.onnx')
DECODED_HEAD_ONNX  = os.path.join(_HERE, 'onnx', 'backbone_decoded_fp32.onnx')  # for proto


def load_ort_session(path):
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(path, providers=providers)
    print(f"  Loaded: {os.path.basename(path)}  [{sess.get_providers()[0]}]")
    return sess


def run_ort(sess, feed: dict):
    return sess.run(None, feed)


def benchmark_fn(fn, n=50, warmup=10):
    for _ in range(warmup): fn()
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n): fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n
    else:
        t0 = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - t0) / n * 1000


def main(args):
    print("\n── Loading models ────────────────────────────────────────────────")
    sess_fp32 = load_ort_session(BACKBONE_FP32_ONNX)
    sess_int8 = load_ort_session(BACKBONE_INT8_ONNX)

    # Pick post-processor
    if args.use_cub:
        print("\n── Loading CUB CUDA post-processor (JIT compile ~30s first time)")
        from pp_postprocess_cuda import PPPostProcessorCUDA
        post = PPPostProcessorCUDA(DECODED_HEAD_ONNX,
                                   score_threshold=args.score_thr,
                                   topk=args.topk)
        print("  CUB kernel ready.")
    else:
        from pp_postprocess import PPPostProcessor
        post = PPPostProcessor(DECODED_HEAD_ONNX,
                               score_threshold=args.score_thr,
                               topk=args.topk)
        print("  PyTorch post-processor ready.")

    # Load BEV frame
    print(f"\n── Loading BEV frame: {args.bev_npz}")
    data = np.load(args.bev_npz)
    bev  = data['bev']  # [1, 64, 400, 656]
    print(f"  BEV shape: {bev.shape}  dtype: {bev.dtype}")

    # FP32 inference + decode
    print("\n── FP32 inference ────────────────────────────────────────────────")
    raw_fp32 = run_ort(sess_fp32, {'spatial_features': bev})
    boxes_fp32, scores_fp32, cls_fp32 = post(*raw_fp32)
    print(f"  Detections: {len(scores_fp32)}")
    if len(scores_fp32) > 0:
        top = scores_fp32.cpu().numpy().argsort()[::-1][:3]
        for i in top:
            b = boxes_fp32[i].cpu().numpy()
            print(f"    cls={int(cls_fp32[i])}  score={float(scores_fp32[i]):.3f}"
                  f"  x={b[0]:.1f}  y={b[1]:.1f}  z={b[2]:.1f}"
                  f"  w={b[3]:.1f}  l={b[4]:.1f}  h={b[5]:.1f}")

    # INT8 inference + decode
    print("\n── INT8 inference ────────────────────────────────────────────────")
    raw_int8 = run_ort(sess_int8, {'spatial_features': bev})
    boxes_int8, scores_int8, cls_int8 = post(*raw_int8)
    print(f"  Detections: {len(scores_int8)}")

    # Accuracy comparison
    if len(scores_fp32) > 0 and len(scores_int8) > 0:
        s_fp = scores_fp32.cpu().numpy()[:min(50, len(scores_fp32))]
        s_i8 = scores_int8.cpu().numpy()[:min(50, len(scores_int8))]
        b_fp = boxes_fp32[:len(s_fp)].cpu().numpy()
        b_i8 = boxes_int8[:len(s_i8)].cpu().numpy()
        n = min(len(s_fp), len(s_i8))
        print(f"\n  Score MAE (top-{n}): {np.mean(np.abs(s_fp[:n] - s_i8[:n])):.4f}")
        center_d = np.sqrt((b_fp[:n, 0] - b_i8[:n, 0])**2 + (b_fp[:n, 1] - b_i8[:n, 1])**2)
        print(f"  Center MAE (top-{n}): {np.mean(center_d):.3f} m")

    # Speed benchmark (post-processor only, GPU tensors)
    print("\n── Post-processor speed (GPU tensors, no H2D copy) ──────────────")
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_gpu = [torch.from_numpy(r).to(dev) for r in raw_int8]
    t_ms = benchmark_fn(lambda: post(*raw_gpu), n=200, warmup=20)
    print(f"  {t_ms:.2f} ms  ({'CUB CUDA kernel' if args.use_cub else 'PyTorch nonzero'})")
    print("\n  TRT backbone latency (from trtexec benchmark):")
    print("    encoder_fp16.trt        0.43 ms")
    print("    backbone_rawhead_int8.trt  1.62 ms")
    print(f"    post-processor            {t_ms:.2f} ms")
    print(f"    ──────────────────────────────────────")
    print(f"    TOTAL (estimated)         {0.43 + 1.62 + t_ms:.2f} ms")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--bev_npz',   required=True,
                   help='Path to a calibration NPZ (must contain "bev" key)')
    p.add_argument('--score_thr', type=float, default=0.1)
    p.add_argument('--topk',      type=int,   default=500)
    p.add_argument('--use_cub',   action='store_true',
                   help='Use CUB CUDA kernel (compiles on first run)')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
