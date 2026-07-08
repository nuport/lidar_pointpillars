#ifndef PP_POSTPROCESS_CUDA_H
#define PP_POSTPROCESS_CUDA_H

#include <string>
#include <vector>

#include <NvInfer.h>

#include "common_trt.h"

struct RawHeadPostprocessConfig {
    float score_threshold = 0.1f;
    float grid_step = 0.4f;
    float nms_threshold = 0.2f;
    int max_detections = 100;
    int class_id_offset = 0;
    std::vector<float> anchor_prototypes;
};

// Returns true if decode succeeds. The first implementation pass provides an
// integration hook and will be replaced with the CUB decode path.
bool decodeRawHeadsCUDA(const std::vector<float>& cls_output,
                        const std::vector<float>& box_output,
                        const std::vector<float>& dir_output,
                        const nvinfer1::Dims& cls_dims,
                        const nvinfer1::Dims& box_dims,
                        const nvinfer1::Dims& dir_dims,
                        const RawHeadPostprocessConfig& cfg,
                        std::vector<Box3dfull>& bboxes_full,
                        std::string* error_message = nullptr);

#endif  // PP_POSTPROCESS_CUDA_H
