/*
 * pp_post_cuda.cu  —  PointPillars raw-head post-processor (CUB CUDA kernel)
 *
 * Pipeline:
 *   1. sigmoid_argmax_kernel  : fused sigmoid + argmax over N_CLS classes per anchor
 *   2. CUB DeviceSelect       : stream-compact 2.5M anchors → ~25K candidates
 *   3. CUB DeviceRadixSort    : sort candidates by score (descending)
 *   4. decode_boxes_kernel    : gather + on-the-fly anchor decode for top-K
 *
 * Expected speedup vs PyTorch nonzero():  ~0.3–0.5 ms vs ~2.8 ms
 *
 * Build (via torch.utils.cpp_extension — see pp_postprocess_cuda.py):
 *   Requires: CUDA ≥ 11.0, CUB (bundled with CUDA toolkit)
 *   Arch flags: -gencode arch=compute_89,code=sm_89  (RTX 4080 Super / Ada)
 *              -gencode arch=compute_86,code=sm_86  (RTX 3000 / Ampere)
 *              -gencode arch=compute_80,code=sm_80  (A100)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <stdexcept>

/* ── model constants ──────────────────────────────────────────────────────── */
static constexpr int   N_ANC  = 38;
static constexpr int   N_CLS  = 19;
static constexpr float GRID   = 0.4f;   /* voxel_size(0.2) × backbone_stride(2) */
static constexpr float PI     = 3.14159265358979f;

/* ── helpers ──────────────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess)                                                  \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(e) + " at " __FILE__ ":"                  \
                + std::to_string(__LINE__));                                    \
    } while (0)

/* ── kernel 1: fused sigmoid + argmax over classes ─────────────────────────
 *
 *  cls_preds layout: NCHW  [1, N_ANC*N_CLS, H, W]
 *  Channel ch = anc_type * N_CLS + class_idx
 *
 *  flat index i ∈ [0, H*W*N_ANC):
 *    anc_type = i % N_ANC
 *    hw_pos   = i / N_ANC
 *    row      = hw_pos / W
 *    col      = hw_pos % W
 */
__global__ void sigmoid_argmax_kernel(
    const float* __restrict__ cls_preds,   /* [N_ANC*N_CLS, H*W]  (NCHW, batch=0 dropped) */
    float*       __restrict__ out_scores,  /* [H*W*N_ANC] */
    int*         __restrict__ out_cls,     /* [H*W*N_ANC] */
    int H, int W
) {
    const int HW      = H * W;
    const int N_total = HW * N_ANC;
    const int i       = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_total) return;

    const int anc_type = i % N_ANC;
    const int hw_pos   = i / N_ANC;
    const int row      = hw_pos / W;
    const int col      = hw_pos % W;
    const int hw_idx   = row * W + col;

    float max_logit = -1e20f;
    int   max_c     = 0;

    #pragma unroll
    for (int c = 0; c < N_CLS; c++) {
        float v = cls_preds[(anc_type * N_CLS + c) * HW + hw_idx];
        if (v > max_logit) { max_logit = v; max_c = c; }
    }

    out_scores[i] = __fdividef(1.0f, 1.0f + __expf(-max_logit));
    out_cls[i]    = max_c;
}

/* ── kernel 2: build iota (sequential index array) ─────────────────────── */
__global__ void iota_kernel(int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = i;
}

/* ── kernel 3: flag anchors above threshold ───────────────────────────── */
__global__ void threshold_flags_kernel(
    const float* __restrict__ scores,
    uint8_t*     __restrict__ flags,
    int N, float thr
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) flags[i] = (scores[i] > thr) ? 1 : 0;
}

/* ── kernel 4: gather scores at selected indices (for sort keys) ──────── */
__global__ void gather_scores_kernel(
    const float* __restrict__ all_scores,
    const int*   __restrict__ sel_idx,
    float*       __restrict__ sel_scores,
    int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) sel_scores[i] = all_scores[sel_idx[i]];
}

/* ── kernel 5: decode top-K boxes on the fly ──────────────────────────── */
__global__ void decode_boxes_kernel(
    const float* __restrict__ box_preds,   /* [7*N_ANC, H*W] NCHW */
    const float* __restrict__ dir_preds,   /* [2*N_ANC, H*W] NCHW */
    const float* __restrict__ proto,       /* [N_ANC, 7]  cx0 cy0 cz w l h yaw0 */
    const float* __restrict__ all_scores,
    const int*   __restrict__ all_cls,
    const int*   __restrict__ topk_idx,    /* global anchor indices, sorted desc by score */
    float*       __restrict__ out_boxes,   /* [K, 7] */
    float*       __restrict__ out_scores,  /* [K] */
    int*         __restrict__ out_cls,     /* [K] */
    int K, int H, int W
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int HW  = H * W;
    const int gi  = topk_idx[k];

    const int anc_type = gi % N_ANC;
    const int hw_pos   = gi / N_ANC;
    const int col      = hw_pos % W;
    const int row      = hw_pos / W;
    const int hw_idx   = row * W + col;

    /* anchor prototype */
    const float cx0    = proto[anc_type * 7 + 0];
    const float cy0    = proto[anc_type * 7 + 1];
    const float cz0    = proto[anc_type * 7 + 2];
    const float anc_w  = proto[anc_type * 7 + 3];
    const float anc_l  = proto[anc_type * 7 + 4];
    const float anc_h  = proto[anc_type * 7 + 5];
    const float anc_yaw = proto[anc_type * 7 + 6];

    const float cx   = cx0 + col * GRID;
    const float cy   = cy0 + row * GRID;
    const float diag = sqrtf(anc_w * anc_w + anc_l * anc_l);

    /* box regression  [ch = anc_type*7 + coord] */
    const float bx = box_preds[(anc_type * 7 + 0) * HW + hw_idx];
    const float by = box_preds[(anc_type * 7 + 1) * HW + hw_idx];
    const float bz = box_preds[(anc_type * 7 + 2) * HW + hw_idx];
    const float bw = box_preds[(anc_type * 7 + 3) * HW + hw_idx];
    const float bl = box_preds[(anc_type * 7 + 4) * HW + hw_idx];
    const float bh = box_preds[(anc_type * 7 + 5) * HW + hw_idx];
    const float ba = box_preds[(anc_type * 7 + 6) * HW + hw_idx];

    out_boxes[k * 7 + 0] = bx * diag + cx;
    out_boxes[k * 7 + 1] = by * diag + cy;
    out_boxes[k * 7 + 2] = bz * anc_h + cz0;
    out_boxes[k * 7 + 3] = __expf(bw) * anc_w;
    out_boxes[k * 7 + 4] = __expf(bl) * anc_l;
    out_boxes[k * 7 + 5] = __expf(bh) * anc_h;

    /* direction correction: limit_period(ba + anc_yaw, 0.5, π) + dir*π */
    float pred_yaw = ba + anc_yaw;
    pred_yaw = pred_yaw - floorf(pred_yaw / PI + 0.5f) * PI;
    const float d0 = dir_preds[(anc_type * 2 + 0) * HW + hw_idx];
    const float d1 = dir_preds[(anc_type * 2 + 1) * HW + hw_idx];
    out_boxes[k * 7 + 6] = pred_yaw + ((d1 > d0) ? PI : 0.0f);

    out_scores[k] = all_scores[gi];
    out_cls[k]    = all_cls[gi];
}

/* ── main entry point ─────────────────────────────────────────────────────── */
std::vector<torch::Tensor> pp_post_forward(
    torch::Tensor cls_preds,      /* [1, 722, H, W]  CUDA float32 */
    torch::Tensor box_preds,      /* [1, 266, H, W]  CUDA float32 */
    torch::Tensor dir_preds,      /* [1,  76, H, W]  CUDA float32 */
    torch::Tensor proto,          /* [38, 7]          CUDA float32 */
    float score_threshold,
    int   topk
) {
    TORCH_CHECK(cls_preds.is_cuda(),  "cls_preds must be a CUDA tensor");
    TORCH_CHECK(cls_preds.is_contiguous(), "cls_preds must be contiguous");
    TORCH_CHECK(box_preds.is_contiguous());
    TORCH_CHECK(dir_preds.is_contiguous());
    TORCH_CHECK(proto.is_contiguous());

    const int H = cls_preds.size(2);
    const int W = cls_preds.size(3);
    const int N_total = H * W * N_ANC;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* ── step 1: sigmoid + argmax → scores[N], cls[N] ─────────────────────── */
    auto scores_all = torch::empty({N_total},     cls_preds.options());
    auto cls_all    = torch::empty({N_total},     cls_preds.options().dtype(torch::kInt32));

    const int T1 = 256;
    sigmoid_argmax_kernel<<<(N_total + T1 - 1) / T1, T1, 0, stream>>>(
        cls_preds.data_ptr<float>(),
        scores_all.data_ptr<float>(),
        cls_all.data_ptr<int>(),
        H, W
    );

    /* ── step 2: threshold flags ─────────────────────────────────────────── */
    auto flags = torch::empty({N_total}, cls_preds.options().dtype(torch::kUInt8));
    threshold_flags_kernel<<<(N_total + T1 - 1) / T1, T1, 0, stream>>>(
        scores_all.data_ptr<float>(),
        flags.data_ptr<uint8_t>(),
        N_total, score_threshold
    );

    /* ── step 3: CUB stream compaction → selected_indices[M] ──────────────── */
    auto iota_buf = torch::empty({N_total}, cls_preds.options().dtype(torch::kInt32));
    iota_kernel<<<(N_total + T1 - 1) / T1, T1, 0, stream>>>(
        iota_buf.data_ptr<int>(), N_total
    );

    /* num_selected lives on device; read back after sync */
    auto num_sel_dev = torch::zeros({1}, cls_preds.options().dtype(torch::kInt32));
    auto cand_idx    = torch::empty({N_total}, cls_preds.options().dtype(torch::kInt32));

    /* query temp storage size */
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(
        d_temp, temp_bytes,
        iota_buf.data_ptr<int>(),
        flags.data_ptr<uint8_t>(),
        cand_idx.data_ptr<int>(),
        num_sel_dev.data_ptr<int>(),
        N_total, stream
    );

    auto temp_buf = torch::empty({(int64_t)temp_bytes},
                                  cls_preds.options().dtype(torch::kUInt8));
    /* actual compaction */
    cub::DeviceSelect::Flagged(
        temp_buf.data_ptr<void>(), temp_bytes,
        iota_buf.data_ptr<int>(),
        flags.data_ptr<uint8_t>(),
        cand_idx.data_ptr<int>(),
        num_sel_dev.data_ptr<int>(),
        N_total, stream
    );

    /* read back M (host sync required) */
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int M = num_sel_dev.item<int>();

    if (M == 0) {
        return {
            torch::zeros({0, 7}, cls_preds.options()),
            torch::zeros({0},    cls_preds.options()),
            torch::zeros({0},    cls_preds.options().dtype(torch::kInt32)),
        };
    }

    /* ── step 4: gather scores for M candidates, then sort descending ──────── */
    auto cand_scores = torch::empty({M}, cls_preds.options());
    const int T4 = 256;
    gather_scores_kernel<<<(M + T4 - 1) / T4, T4, 0, stream>>>(
        scores_all.data_ptr<float>(),
        cand_idx.data_ptr<int>(),
        cand_scores.data_ptr<float>(),
        M
    );

    auto sorted_scores = torch::empty({M}, cls_preds.options());
    auto sorted_idx    = torch::empty({M}, cls_preds.options().dtype(torch::kInt32));

    /* CUB sort pairs descending: (score, global_index) */
    void*  d_temp2 = nullptr;
    size_t temp2_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        d_temp2, temp2_bytes,
        cand_scores.data_ptr<float>(),   sorted_scores.data_ptr<float>(),
        cand_idx.data_ptr<int>(),        sorted_idx.data_ptr<int>(),
        M, 0, sizeof(float) * 8, stream
    );
    auto temp2_buf = torch::empty({(int64_t)temp2_bytes},
                                   cls_preds.options().dtype(torch::kUInt8));
    cub::DeviceRadixSort::SortPairsDescending(
        temp2_buf.data_ptr<void>(), temp2_bytes,
        cand_scores.data_ptr<float>(),   sorted_scores.data_ptr<float>(),
        cand_idx.data_ptr<int>(),        sorted_idx.data_ptr<int>(),
        M, 0, sizeof(float) * 8, stream
    );

    /* ── step 5: decode top-K boxes ─────────────────────────────────────────── */
    const int K = std::min(topk, M);

    auto out_boxes  = torch::empty({K, 7}, cls_preds.options());
    auto out_scores = torch::empty({K},    cls_preds.options());
    auto out_cls    = torch::empty({K},    cls_preds.options().dtype(torch::kInt32));

    const int T5 = 256;
    decode_boxes_kernel<<<(K + T5 - 1) / T5, T5, 0, stream>>>(
        box_preds.data_ptr<float>(),
        dir_preds.data_ptr<float>(),
        proto.data_ptr<float>(),
        scores_all.data_ptr<float>(),
        cls_all.data_ptr<int>(),
        sorted_idx.data_ptr<int>(),
        out_boxes.data_ptr<float>(),
        out_scores.data_ptr<float>(),
        out_cls.data_ptr<int>(),
        K, H, W
    );

    return {out_boxes, out_scores, out_cls};
}

/* ── pybind11 binding ────────────────────────────────────────────────────── */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pp_post_forward,
          "PointPillars raw-head post-processor (CUB CUDA)",
          py::arg("cls_preds"),
          py::arg("box_preds"),
          py::arg("dir_preds"),
          py::arg("proto"),
          py::arg("score_threshold") = 0.1f,
          py::arg("topk")            = 500);
}
