#include "pp_postprocess_cuda.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "nms_trt.h"
#include "utils.h"

namespace {

struct Candidate {
    float score;
    int class_id;
    int anc_type;
    int row;
    int col;
};

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline size_t idxNCHW(int c, int h, int w, int H, int W) {
    return static_cast<size_t>(c) * static_cast<size_t>(H) * static_cast<size_t>(W)
         + static_cast<size_t>(h) * static_cast<size_t>(W)
         + static_cast<size_t>(w);
}

}  // namespace

bool decodeRawHeadsCUDA(const std::vector<float>& cls_output,
                        const std::vector<float>& box_output,
                        const std::vector<float>& dir_output,
                        const nvinfer1::Dims& cls_dims,
                        const nvinfer1::Dims& box_dims,
                        const nvinfer1::Dims& dir_dims,
                        const RawHeadPostprocessConfig& cfg,
                        std::vector<Box3dfull>& bboxes_full,
                        std::string* error_message) {
    auto setError = [&](const std::string& msg) {
        if (error_message) {
            *error_message = msg;
        }
    };

    bboxes_full.clear();

    if (cls_dims.nbDims != 4 || box_dims.nbDims != 4 || dir_dims.nbDims != 4) {
        setError("Expected 4D raw heads for cls/box/dir.");
        return false;
    }

    if (cls_dims.d[0] != 1 || box_dims.d[0] != 1 || dir_dims.d[0] != 1) {
        setError("Only batch size 1 is supported in raw-head decoder.");
        return false;
    }

    const int H = cls_dims.d[2];
    const int W = cls_dims.d[3];
    if (H <= 0 || W <= 0 || box_dims.d[2] != H || box_dims.d[3] != W || dir_dims.d[2] != H || dir_dims.d[3] != W) {
        setError("Spatial shape mismatch between cls/box/dir heads.");
        return false;
    }

    const int cls_ch = cls_dims.d[1];
    const int box_ch = box_dims.d[1];
    const int dir_ch = dir_dims.d[1];
    if (box_ch % 7 != 0 || dir_ch % 2 != 0) {
        setError("Invalid channel layout: box_ch must be multiple of 7 and dir_ch multiple of 2.");
        return false;
    }

    const int n_anc = box_ch / 7;
    if (n_anc <= 0 || dir_ch != n_anc * 2 || cls_ch % n_anc != 0) {
        setError("Inconsistent anchor channels among cls/box/dir heads.");
        return false;
    }
    const int n_cls = cls_ch / n_anc;
    if (n_cls <= 0) {
        setError("Derived class count is invalid.");
        return false;
    }

    const size_t expected_cls = static_cast<size_t>(cls_ch) * static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t expected_box = static_cast<size_t>(box_ch) * static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t expected_dir = static_cast<size_t>(dir_ch) * static_cast<size_t>(H) * static_cast<size_t>(W);
    if (cls_output.size() != expected_cls || box_output.size() != expected_box || dir_output.size() != expected_dir) {
        std::ostringstream oss;
        oss << "Raw-head element count mismatch. got(cls/box/dir)="
            << cls_output.size() << "/" << box_output.size() << "/" << dir_output.size()
            << " expected=" << expected_cls << "/" << expected_box << "/" << expected_dir;
        setError(oss.str());
        return false;
    }

    if (cfg.anchor_prototypes.size() != static_cast<size_t>(n_anc) * 7U) {
        std::ostringstream oss;
        oss << "anchor_prototypes size mismatch. got=" << cfg.anchor_prototypes.size()
            << " expected=" << (static_cast<size_t>(n_anc) * 7U)
            << " (n_anc=" << n_anc << ")";
        setError(oss.str());
        return false;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(H) * static_cast<size_t>(W));

    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            for (int anc = 0; anc < n_anc; ++anc) {
                float best_score = -1.0f;
                int best_cls = 0;
                for (int c = 0; c < n_cls; ++c) {
                    const int ch = anc * n_cls + c;
                    const float score = sigmoid(cls_output[idxNCHW(ch, row, col, H, W)]);
                    if (score > best_score) {
                        best_score = score;
                        best_cls = c;
                    }
                }
                if (best_score > cfg.score_threshold) {
                    candidates.push_back({best_score, best_cls, anc, row, col});
                }
            }
        }
    }

    if (candidates.empty()) {
        return true;
    }

    const int topk = std::min(cfg.max_detections, static_cast<int>(candidates.size()));
    std::partial_sort(candidates.begin(), candidates.begin() + topk, candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    std::vector<Box2d> boxes2d;
    std::vector<Box3d> boxes3d;
    std::vector<float> scores;
    std::vector<int> classes;
    std::vector<float> directions;
    boxes2d.reserve(topk);
    boxes3d.reserve(topk);
    scores.reserve(topk);
    classes.reserve(topk);
    directions.reserve(topk);

    for (int i = 0; i < topk; ++i) {
        const Candidate& cand = candidates[i];
        const int anc = cand.anc_type;
        const float* proto = &cfg.anchor_prototypes[anc * 7];
        const float cx0 = proto[0];
        const float cy0 = proto[1];
        const float anc_cz = proto[2];
        const float anc_w = proto[3];
        const float anc_l = proto[4];
        const float anc_h = proto[5];
        const float anc_yaw = proto[6];

        const float cx = cx0 + static_cast<float>(cand.col) * cfg.grid_step;
        const float cy = cy0 + static_cast<float>(cand.row) * cfg.grid_step;
        const float diag = std::sqrt(anc_w * anc_w + anc_l * anc_l);

        const int box_base = anc * 7;
        const float bx = box_output[idxNCHW(box_base + 0, cand.row, cand.col, H, W)];
        const float by = box_output[idxNCHW(box_base + 1, cand.row, cand.col, H, W)];
        const float bz = box_output[idxNCHW(box_base + 2, cand.row, cand.col, H, W)];
        const float bw = box_output[idxNCHW(box_base + 3, cand.row, cand.col, H, W)];
        const float bl = box_output[idxNCHW(box_base + 4, cand.row, cand.col, H, W)];
        const float bh = box_output[idxNCHW(box_base + 5, cand.row, cand.col, H, W)];
        const float ba = box_output[idxNCHW(box_base + 6, cand.row, cand.col, H, W)];

        const float pred_x = bx * diag + cx;
        const float pred_y = by * diag + cy;
        const float pred_z = bz * anc_h + anc_cz;
        const float pred_w = std::exp(bw) * anc_w;
        const float pred_l = std::exp(bl) * anc_l;
        const float pred_h = std::exp(bh) * anc_h;
        const float pred_yaw = ba + anc_yaw;

        const int dir_base = anc * 2;
        const float d0 = dir_output[idxNCHW(dir_base + 0, cand.row, cand.col, H, W)];
        const float d1 = dir_output[idxNCHW(dir_base + 1, cand.row, cand.col, H, W)];
        const float direction = (d1 > d0) ? 1.0f : 0.0f;

        Box2d box2d;
        box2d.x1 = pred_x - pred_l * 0.5f;
        box2d.y1 = pred_y - pred_w * 0.5f;
        box2d.x2 = pred_x + pred_l * 0.5f;
        box2d.y2 = pred_y + pred_w * 0.5f;
        box2d.theta = pred_yaw;
        boxes2d.push_back(box2d);

        Box3d box3d;
        box3d.x = pred_x;
        box3d.y = pred_y;
        box3d.z = pred_z;
        box3d.w = pred_w;
        box3d.l = pred_l;
        box3d.h = pred_h;
        box3d.theta = pred_yaw;
        boxes3d.push_back(box3d);

        scores.push_back(cand.score);
        classes.push_back(cand.class_id + cfg.class_id_offset);
        directions.push_back(direction);
    }

    std::vector<int> nms_filter_inds;
    float nms_thr = cfg.nms_threshold;
    nms(boxes2d, scores, nms_thr, nms_filter_inds);

    std::vector<Box3dfull> nms_boxes;
    nms_boxes.reserve(nms_filter_inds.size());
    for (const int ind : nms_filter_inds) {
        Box3dfull box;
        box.x = boxes3d[ind].x;
        box.y = boxes3d[ind].y;
        box.z = boxes3d[ind].z;
        box.w = boxes3d[ind].w;
        box.l = boxes3d[ind].l;
        box.h = boxes3d[ind].h;

        float limited_theta = boxes3d[ind].theta;
        limited_theta = limitPeriod(limited_theta);
        box.theta = (1.0f - directions[ind]) * M_PI + limited_theta;
        box.score = scores[ind];
        box.label = classes[ind];
        nms_boxes.push_back(box);
    }

    int max_num = cfg.max_detections;
    getTopkBoxes(nms_boxes, max_num, bboxes_full);
    return true;
}
