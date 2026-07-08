#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nuport_perception_msgs/ObjectDetectionArray.h>
#include <nuport_perception_msgs/ObjectDetection.h>
#include <nuport_perception_msgs/ObjectClassification.h>
#include <nuport_perception_msgs/ObjectKinematics.h>
#include <nuport_perception_msgs/Shape.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/Vector3.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <xmlrpcpp/XmlRpcValue.h>

#include "common_trt.h"
#include "io_trt.hpp"
#include "voxelization_trt.h"
#include "utils.h"
#include "nms_trt.h"
#include "pp_postprocess_cuda.h"

class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        switch(severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
            case Severity::kWARNING:
                ROS_WARN("%s", msg);
                break;
            default:
                break;
        }
    }
};

#define CHECK_CUDA(status) \
    if (status != 0) \
    { \
        ROS_FATAL("CUDA failure: %d", status); \
        ros::shutdown(); \
    }

class PointPillarsNodelet : public nodelet::Nodelet
{
public:
        PointPillarsNodelet()
                : encoder_runtime_(nullptr), encoder_engine_(nullptr), encoder_context_(nullptr),
                    backbone_runtime_(nullptr), backbone_engine_(nullptr), backbone_context_(nullptr) {}

    ~PointPillarsNodelet()
    {
                delete backbone_context_;
                delete backbone_engine_;
                delete backbone_runtime_;
                delete encoder_context_;
                delete encoder_engine_;
                delete encoder_runtime_;
    }

    void onInit() override
    {
        ros::NodeHandle& nh  = getNodeHandle();
        ros::NodeHandle& pnh = getPrivateNodeHandle();

        // Parameters
        std::string encoder_engine_path;
        std::string backbone_engine_path;
        pnh.param<std::string>("encoder_engine_path", encoder_engine_path, "");
        pnh.param<std::string>("backbone_engine_path", backbone_engine_path, "");
        pnh.param<std::string>("input_topic", input_topic_, "/points");
        pnh.param<std::string>("output_topic", output_topic_, "/detections");
        pnh.param<float>("score_threshold", score_thr_, 0.1f);
        pnh.param<float>("nms_threshold", nms_thr_, 0.01f);
        pnh.param<int>("max_detections", max_num_, 50);
        pnh.param<int>("num_classes", num_class_, 19);
        pnh.param<int>("num_boxes", num_box_, 1000);
        pnh.param<int>("max_points_per_voxel", max_points_, 32);
        pnh.param<int>("max_voxels", max_voxels_, 40000);
        pnh.param<std::string>("tensor_name_encoder_pillars", tensor_name_encoder_pillars_, "voxels");
        pnh.param<std::string>("tensor_name_encoder_coords", tensor_name_encoder_coords_, "coords");
        pnh.param<std::string>("tensor_name_encoder_num_points", tensor_name_encoder_num_points_, "voxel_num_points");
        pnh.param<std::string>("tensor_name_encoder_output", tensor_name_encoder_output_, "spatial_features");
        pnh.param<std::string>("tensor_name_backbone_input", tensor_name_backbone_input_, "spatial_features");
        pnh.param<std::string>("tensor_name_backbone_cls_output", tensor_name_backbone_cls_output_, "cls_preds");
        pnh.param<std::string>("tensor_name_backbone_box_output", tensor_name_backbone_box_output_, "box_preds");
        pnh.param<std::string>("tensor_name_backbone_dir_output", tensor_name_backbone_dir_output_, "dir_preds");
        pnh.param<int>("point_feature_dim", point_feature_dim_, 4);
        pnh.param<float>("grid_step", grid_step_, 0.4f);
        pnh.param<std::string>("anchor_prototypes_file", anchor_prototypes_file_, "");
        pnh.param<int>("class_id_offset", class_id_offset_, 0);
        pnh.param<float>("z_shift", z_shift_, 0.0f);
        pnh.param<bool>("xy_center", xy_center_, false);

        loadAnchorPrototypes(pnh);

        if (!pnh.getParam("voxel_size", voxel_size_)) {
            voxel_size_ = {0.2f, 0.2f, 10.0f};
            NODELET_INFO("Using default voxel_size_: [0.2, 0.2, 10.0]");
        }
        if (!pnh.getParam("coors_range", coors_range_)) {
            coors_range_ = {-60.0f, -40.0f, -2.0f, 71.2f, 60.8f, 8.0f};
            NODELET_INFO("Using default coors_range_: [-60.0, -40.0, -2.0, 71.2, 60.8, 8.0]");
        }
        if (!pnh.getParam("class_names", class_names_)) {
            class_names_ = {
                "Car", "Truck", "Bus", "Construction", "Motorcycle", "Bicycle", "Trailer",
                "EmergencyVehicle", "Train", "OtherVehicle", "EgoTrailer", "Pedestrian", "Animal",
                "TrafficCone", "Barrier", "PushablePullable", "Debris", "BicycleRack", "TrafficSign"
            };
            NODELET_INFO("Using default class_names for legacy mantruck model");
        }

        initClassMap();
        NDim_ = 3;

        if (encoder_engine_path.empty() || backbone_engine_path.empty()) {
            NODELET_FATAL("Parameters 'encoder_engine_path' and 'backbone_engine_path' are required");
            return;
        }

        if (point_feature_dim_ != static_cast<int>(sizeof(Point) / sizeof(float))) {
            NODELET_FATAL("point_feature_dim=%d does not match Point layout=%zu. Update Point struct or model contract first.",
                          point_feature_dim_, sizeof(Point) / sizeof(float));
            return;
        }

        bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
        if (!didInitPlugins) {
            NODELET_WARN("Failed to initialize TensorRT plugins");
        }

        if (!initTRT(encoder_engine_path, encoder_runtime_, encoder_engine_, encoder_context_, "encoder")) {
            NODELET_FATAL("Failed to initialize encoder TensorRT engine");
            return;
        }
        if (!initTRT(backbone_engine_path, backbone_runtime_, backbone_engine_, backbone_context_, "backbone")) {
            NODELET_FATAL("Failed to initialize backbone TensorRT engine");
            return;
        }
        if (!validateTensorBindings()) {
            NODELET_FATAL("Tensor name validation failed for split architecture");
            return;
        }

        // Publishers and subscribers
        det_pub_ = nh.advertise<nuport_perception_msgs::ObjectDetectionArray>(output_topic_, 1);
        pc_sub_ = nh.subscribe(input_topic_, 1, &PointPillarsNodelet::pointCloudCallback, this);

        NODELET_INFO("PointPillars nodelet initialized. Subscribing to: %s, Publishing to: %s",
                     input_topic_.c_str(), output_topic_.c_str());
    }

private:
    bool initTRT(const std::string& engine_path,
                 nvinfer1::IRuntime*& runtime,
                 nvinfer1::ICudaEngine*& engine,
                 nvinfer1::IExecutionContext*& context,
                 const std::string& engine_name)
    {
        std::vector<char> engineData;
        try {
            engineData = readEngineFile(engine_path);
        } catch (const std::runtime_error& e) {
            NODELET_ERROR("Failed to read %s engine file: %s", engine_name.c_str(), e.what());
            return false;
        }

        runtime = nvinfer1::createInferRuntime(trt_logger_);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine) {
            NODELET_ERROR("Failed to deserialize %s CUDA engine", engine_name.c_str());
            return false;
        }

        context = engine->createExecutionContext();
        if (!context) {
            NODELET_ERROR("Failed to create %s execution context", engine_name.c_str());
            return false;
        }

        return true;
    }

    bool validateTensorBinding(const nvinfer1::ICudaEngine* engine,
                               const std::string& tensor_name,
                               nvinfer1::TensorIOMode expected_mode,
                               const std::string& engine_name) const
    {
        if (!engine) {
            NODELET_ERROR("%s engine is null while validating tensor '%s'", engine_name.c_str(), tensor_name.c_str());
            return false;
        }
        const int io_count = engine->getNbIOTensors();
        bool found = false;
        for (int i = 0; i < io_count; ++i) {
            const char* io_name = engine->getIOTensorName(i);
            if (io_name && tensor_name == io_name) {
                found = true;
                break;
            }
        }
        if (!found) {
            NODELET_ERROR("Tensor '%s' not found in %s engine", tensor_name.c_str(), engine_name.c_str());
            return false;
        }
        const auto mode = engine->getTensorIOMode(tensor_name.c_str());
        if (mode != expected_mode) {
            NODELET_ERROR("Tensor '%s' in %s engine has wrong IO mode", tensor_name.c_str(), engine_name.c_str());
            return false;
        }
        return true;
    }

    bool validateTensorBindings() const
    {
        bool ok = true;
        ok &= validateTensorBinding(encoder_engine_, tensor_name_encoder_pillars_, nvinfer1::TensorIOMode::kINPUT, "encoder");
        ok &= validateTensorBinding(encoder_engine_, tensor_name_encoder_coords_, nvinfer1::TensorIOMode::kINPUT, "encoder");
        ok &= validateTensorBinding(encoder_engine_, tensor_name_encoder_num_points_, nvinfer1::TensorIOMode::kINPUT, "encoder");
        ok &= validateTensorBinding(encoder_engine_, tensor_name_encoder_output_, nvinfer1::TensorIOMode::kOUTPUT, "encoder");

        ok &= validateTensorBinding(backbone_engine_, tensor_name_backbone_input_, nvinfer1::TensorIOMode::kINPUT, "backbone");
        ok &= validateTensorBinding(backbone_engine_, tensor_name_backbone_cls_output_, nvinfer1::TensorIOMode::kOUTPUT, "backbone");
        ok &= validateTensorBinding(backbone_engine_, tensor_name_backbone_box_output_, nvinfer1::TensorIOMode::kOUTPUT, "backbone");
        ok &= validateTensorBinding(backbone_engine_, tensor_name_backbone_dir_output_, nvinfer1::TensorIOMode::kOUTPUT, "backbone");
        return ok;
    }

    size_t numElements(const nvinfer1::Dims& dims) const
    {
        if (dims.nbDims <= 0) {
            return 0;
        }
        size_t count = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0) {
                return 0;
            }
            count *= static_cast<size_t>(dims.d[i]);
        }
        return count;
    }

    static bool xmlValueToFloat(const XmlRpc::XmlRpcValue& value, float& out)
    {
        if (value.getType() == XmlRpc::XmlRpcValue::TypeInt) {
            out = static_cast<float>(static_cast<int>(value));
            return true;
        }
        if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
            out = static_cast<float>(static_cast<double>(value));
            return true;
        }
        return false;
    }

    bool loadAnchorsFromParam(const ros::NodeHandle& pnh, std::vector<float>& out) const
    {
        std::vector<double> flat;
        if (pnh.getParam("anchor_prototypes", flat) && !flat.empty()) {
            out.assign(flat.begin(), flat.end());
            return true;
        }

        XmlRpc::XmlRpcValue xml;
        if (!pnh.getParam("anchor_prototypes", xml)) {
            return false;
        }
        if (xml.getType() != XmlRpc::XmlRpcValue::TypeArray) {
            return false;
        }

        for (int i = 0; i < xml.size(); ++i) {
            if (xml[i].getType() == XmlRpc::XmlRpcValue::TypeArray) {
                for (int j = 0; j < xml[i].size(); ++j) {
                    float v = 0.0f;
                    if (!xmlValueToFloat(xml[i][j], v)) {
                        return false;
                    }
                    out.push_back(v);
                }
            } else {
                float v = 0.0f;
                if (!xmlValueToFloat(xml[i], v)) {
                    return false;
                }
                out.push_back(v);
            }
        }
        return !out.empty();
    }

    bool loadAnchorsFromFile(const std::string& file_path, std::vector<float>& out) const
    {
        if (file_path.empty()) {
            return false;
        }

        std::ifstream ifs(file_path.c_str());
        if (!ifs.is_open()) {
            NODELET_WARN("Could not open anchor_prototypes_file: %s", file_path.c_str());
            return false;
        }

        std::string line;
        while (std::getline(ifs, line)) {
            for (char& ch : line) {
                if (ch == ',') {
                    ch = ' ';
                }
            }
            std::istringstream iss(line);
            float v = 0.0f;
            while (iss >> v) {
                out.push_back(v);
            }
        }
        return !out.empty();
    }

    void loadAnchorPrototypes(const ros::NodeHandle& pnh)
    {
        anchor_prototypes_.clear();

        if (loadAnchorsFromParam(pnh, anchor_prototypes_)) {
            NODELET_INFO("Loaded anchor_prototypes from ROS params: %zu values", anchor_prototypes_.size());
            return;
        }

        if (loadAnchorsFromFile(anchor_prototypes_file_, anchor_prototypes_)) {
            NODELET_INFO("Loaded anchor_prototypes from file %s: %zu values",
                         anchor_prototypes_file_.c_str(), anchor_prototypes_.size());
            return;
        }

        NODELET_WARN("No anchor prototypes loaded. Set 'anchor_prototypes' or 'anchor_prototypes_file'.");
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        // Convert sensor_msgs/PointCloud2 to vector of Points
        std::vector<Point> points_ori;
        points_ori.reserve(msg->width * msg->height);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        // Check if intensity field exists
        bool has_intensity = false;
        for (const auto& field : msg->fields) {
            if (field.name == "intensity") { has_intensity = true; break; }
        }

        if (has_intensity) {
            sensor_msgs::PointCloud2ConstIterator<float> iter_i(*msg, "intensity");
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
                Point p;
                p.x = *iter_x; p.y = *iter_y; p.z = *iter_z; p.feature = *iter_i;
                points_ori.push_back(p);
            }
        } else {
            for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                Point p;
                p.x = *iter_x; p.y = *iter_y; p.z = *iter_z; p.feature = 0.0f;
                points_ori.push_back(p);
            }
        }

        if (z_shift_ != 0.0f) {
            for (auto& p : points_ori) {
                p.z += z_shift_;
            }
        }

        last_ox_ = 0.0f;
        last_oy_ = 0.0f;
        if (xy_center_ && !points_ori.empty()) {
            std::vector<float> xs;
            std::vector<float> ys;
            xs.reserve(points_ori.size());
            ys.reserve(points_ori.size());
            for (const auto& p : points_ori) {
                xs.push_back(p.x);
                ys.push_back(p.y);
            }
            const size_t mid = xs.size() / 2;
            std::nth_element(xs.begin(), xs.begin() + mid, xs.end());
            std::nth_element(ys.begin(), ys.begin() + mid, ys.end());
            const float med_x = xs[mid];
            const float med_y = ys[mid];
            const float bev_cx = 0.5f * (coors_range_[0] + coors_range_[3]);
            const float bev_cy = 0.5f * (coors_range_[1] + coors_range_[4]);
            last_ox_ = bev_cx - med_x;
            last_oy_ = bev_cy - med_y;
            for (auto& p : points_ori) {
                p.x += last_ox_;
                p.y += last_oy_;
            }
        }

        // Filter points to valid range
        std::vector<Point> points;
        pointCloudFilter(points_ori, points, coors_range_);

        if (points.empty()) {
            // Publish empty detection array
            nuport_perception_msgs::ObjectDetectionArray det_array;
            det_array.header = msg->header;
            det_pub_.publish(det_array);
            return;
        }

        // Voxelization on GPU
        int* d_num_points_per_voxel = nullptr;
        cudaMalloc((void**)&d_num_points_per_voxel, max_voxels_ * sizeof(int));
        cudaMemset(d_num_points_per_voxel, 0, max_voxels_ * sizeof(int));
        float* d_voxels = nullptr;
        cudaMalloc((void**)&d_voxels, max_voxels_ * max_points_ * sizeof(Point));
        cudaMemset(d_voxels, 0, max_voxels_ * max_points_ * sizeof(Point));
        int* d_coors = nullptr;
        cudaMalloc((void**)&d_coors, max_voxels_ * NDim_ * sizeof(int));
        cudaMemset(d_coors, 0, max_voxels_ * NDim_ * sizeof(int));

        int voxel_num = voxelizeGpu(points, voxel_size_, coors_range_, max_points_, max_voxels_,
                                    d_voxels, d_coors, d_num_points_per_voxel, NDim_);

        int* d_coors_padded = nullptr;
        cudaMalloc((void**)&d_coors_padded, voxel_num * (NDim_ + 1) * sizeof(int));
        cudaMemset(d_coors_padded, 0, voxel_num * (NDim_ + 1) * sizeof(int));
        padCoorsGPU(d_coors, d_coors_padded, voxel_num);
        cudaFree(d_coors);

        // TRT split inference
        std::vector<float> cls_output;
        std::vector<float> box_output;
        std::vector<float> dir_output;
        nvinfer1::Dims cls_dims{};
        nvinfer1::Dims box_dims{};
        nvinfer1::Dims dir_dims{};
        if (!trtInferSplit(d_voxels, d_coors_padded, d_num_points_per_voxel, voxel_num,
                           max_points_, cls_output, box_output, dir_output, cls_dims, box_dims, dir_dims)) {
            NODELET_ERROR_THROTTLE(1.0, "Split TRT inference failed for current frame");
            nuport_perception_msgs::ObjectDetectionArray det_array;
            det_array.header = msg->header;
            det_pub_.publish(det_array);
            return;
        }

        // Post processing (raw-head decode integration follows in the next implementation step).
        std::vector<Box3dfull> bboxes;
        postProcessingRawHeads(cls_output, box_output, dir_output, cls_dims, box_dims, dir_dims, bboxes);

        if (z_shift_ != 0.0f || xy_center_) {
            for (auto& box : bboxes) {
                if (z_shift_ != 0.0f) {
                    box.z -= z_shift_;
                }
                if (xy_center_) {
                    box.x -= last_ox_;
                    box.y -= last_oy_;
                }
            }
        }

        // Convert to ROS message
        nuport_perception_msgs::ObjectDetectionArray det_array;
        det_array.header = msg->header;

        for (const auto& box : bboxes) {
            nuport_perception_msgs::ObjectDetection det;
            det.likelihood = box.score;

            // Classification
            nuport_perception_msgs::ObjectClassification cls;
            cls.label = labelToClassId(box.label);
            cls.confidence = box.score;
            det.classification.push_back(cls);

            // Kinematics (pose)
            det.kinematics.pose.pose.position.x = box.x;
            det.kinematics.pose.pose.position.y = box.y;
            det.kinematics.pose.pose.position.z = box.z;
            det.kinematics.pose.pose.orientation.x = 0.0;
            det.kinematics.pose.pose.orientation.y = 0.0;
            det.kinematics.pose.pose.orientation.z = std::sin(box.theta * 0.5);
            det.kinematics.pose.pose.orientation.w = std::cos(box.theta * 0.5);

            // Shape
            det.shape.type = nuport_perception_msgs::Shape::BOUNDING_BOX;
            det.shape.dimensions.x = box.l;
            det.shape.dimensions.y = box.w;
            det.shape.dimensions.z = box.h;

            det_array.detections.push_back(det);
        }

        det_pub_.publish(det_array);
    }

    bool trtInferSplit(float* d_voxels, int* d_coors, int* d_num_points_per_voxel,
                       const int pillar_num, int& max_points,
                       std::vector<float>& cls_output,
                       std::vector<float>& box_output,
                       std::vector<float>& dir_output,
                       nvinfer1::Dims& cls_dims,
                       nvinfer1::Dims& box_dims,
                       nvinfer1::Dims& dir_dims)
    {
        void* inputPillarsDevice;
        void* inputCoorsBatchDevice;
        void* inputNpointsPerPillarDevice;
        CHECK_CUDA(cudaMalloc(&inputPillarsDevice, pillar_num * max_points * sizeof(Point)));
        CHECK_CUDA(cudaMalloc(&inputCoorsBatchDevice, pillar_num * 4 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&inputNpointsPerPillarDevice, pillar_num * sizeof(int)));

        CHECK_CUDA(cudaMemcpy(inputPillarsDevice, d_voxels, pillar_num * max_points * sizeof(Point), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(inputCoorsBatchDevice, d_coors, pillar_num * 4 * sizeof(int), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(inputNpointsPerPillarDevice, d_num_points_per_voxel, pillar_num * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaFree(d_voxels);
        cudaFree(d_coors);
        cudaFree(d_num_points_per_voxel);

        nvinfer1::Dims inputPillarDims, inputCoorsDims, inputNpointsPerPillarDims;
        inputPillarDims.nbDims = 3;
        inputPillarDims.d[0] = pillar_num;
        inputPillarDims.d[1] = max_points;
        inputPillarDims.d[2] = sizeof(Point) / sizeof(float);

        inputCoorsDims.nbDims = 2;
        inputCoorsDims.d[0] = pillar_num;
        inputCoorsDims.d[1] = 4;

        inputNpointsPerPillarDims.nbDims = 1;
        inputNpointsPerPillarDims.d[0] = pillar_num;

        if (!encoder_context_->setInputShape(tensor_name_encoder_pillars_.c_str(), inputPillarDims) ||
            !encoder_context_->setInputShape(tensor_name_encoder_coords_.c_str(), inputCoorsDims) ||
            !encoder_context_->setInputShape(tensor_name_encoder_num_points_.c_str(), inputNpointsPerPillarDims)) {
            cudaFree(inputPillarsDevice);
            cudaFree(inputCoorsBatchDevice);
            cudaFree(inputNpointsPerPillarDevice);
            return false;
        }

        const nvinfer1::Dims encoder_output_dims = encoder_context_->getTensorShape(tensor_name_encoder_output_.c_str());
        const size_t encoder_output_size = numElements(encoder_output_dims);
        if (encoder_output_size == 0) {
            cudaFree(inputPillarsDevice);
            cudaFree(inputCoorsBatchDevice);
            cudaFree(inputNpointsPerPillarDevice);
            return false;
        }

        void* encoderOutputDevice = nullptr;
        CHECK_CUDA(cudaMalloc(&encoderOutputDevice, encoder_output_size * sizeof(float)));

        encoder_context_->setTensorAddress(tensor_name_encoder_pillars_.c_str(), inputPillarsDevice);
        encoder_context_->setTensorAddress(tensor_name_encoder_coords_.c_str(), inputCoorsBatchDevice);
        encoder_context_->setTensorAddress(tensor_name_encoder_num_points_.c_str(), inputNpointsPerPillarDevice);
        encoder_context_->setTensorAddress(tensor_name_encoder_output_.c_str(), encoderOutputDevice);
        if (!encoder_context_->enqueueV3(0)) {
            cudaFree(inputPillarsDevice);
            cudaFree(inputCoorsBatchDevice);
            cudaFree(inputNpointsPerPillarDevice);
            cudaFree(encoderOutputDevice);
            return false;
        }

        if (!backbone_context_->setInputShape(tensor_name_backbone_input_.c_str(), encoder_output_dims)) {
            cudaFree(inputPillarsDevice);
            cudaFree(inputCoorsBatchDevice);
            cudaFree(inputNpointsPerPillarDevice);
            cudaFree(encoderOutputDevice);
            return false;
        }

        cls_dims = backbone_context_->getTensorShape(tensor_name_backbone_cls_output_.c_str());
        box_dims = backbone_context_->getTensorShape(tensor_name_backbone_box_output_.c_str());
        dir_dims = backbone_context_->getTensorShape(tensor_name_backbone_dir_output_.c_str());
        const size_t cls_size = numElements(cls_dims);
        const size_t box_size = numElements(box_dims);
        const size_t dir_size = numElements(dir_dims);
        if (cls_size == 0 || box_size == 0 || dir_size == 0) {
            cudaFree(inputPillarsDevice);
            cudaFree(inputCoorsBatchDevice);
            cudaFree(inputNpointsPerPillarDevice);
            cudaFree(encoderOutputDevice);
            return false;
        }

        void* clsDevice = nullptr;
        void* boxDevice = nullptr;
        void* dirDevice = nullptr;
        CHECK_CUDA(cudaMalloc(&clsDevice, cls_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&boxDevice, box_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dirDevice, dir_size * sizeof(float)));

        backbone_context_->setTensorAddress(tensor_name_backbone_input_.c_str(), encoderOutputDevice);
        backbone_context_->setTensorAddress(tensor_name_backbone_cls_output_.c_str(), clsDevice);
        backbone_context_->setTensorAddress(tensor_name_backbone_box_output_.c_str(), boxDevice);
        backbone_context_->setTensorAddress(tensor_name_backbone_dir_output_.c_str(), dirDevice);
        if (!backbone_context_->enqueueV3(0)) {
            cudaFree(inputPillarsDevice);
            cudaFree(inputCoorsBatchDevice);
            cudaFree(inputNpointsPerPillarDevice);
            cudaFree(encoderOutputDevice);
            cudaFree(clsDevice);
            cudaFree(boxDevice);
            cudaFree(dirDevice);
            return false;
        }

        cls_output.resize(cls_size);
        box_output.resize(box_size);
        dir_output.resize(dir_size);
        CHECK_CUDA(cudaMemcpy(cls_output.data(), clsDevice, cls_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(box_output.data(), boxDevice, box_size * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(dir_output.data(), dirDevice, dir_size * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(inputPillarsDevice);
        cudaFree(inputCoorsBatchDevice);
        cudaFree(inputNpointsPerPillarDevice);
        cudaFree(encoderOutputDevice);
        cudaFree(clsDevice);
        cudaFree(boxDevice);
        cudaFree(dirDevice);
        return true;
    }

    void postProcessingRawHeads(const std::vector<float>& cls_output,
                                const std::vector<float>& box_output,
                                const std::vector<float>& dir_output,
                                const nvinfer1::Dims& cls_dims,
                                const nvinfer1::Dims& box_dims,
                                const nvinfer1::Dims& dir_dims,
                                std::vector<Box3dfull>& bboxes_full)
    {
        RawHeadPostprocessConfig cfg;
        cfg.score_threshold = score_thr_;
        cfg.nms_threshold = nms_thr_;
        cfg.max_detections = max_num_;
        cfg.class_id_offset = class_id_offset_;
        cfg.grid_step = grid_step_;
        cfg.anchor_prototypes = anchor_prototypes_;
        std::string decode_error;
        const bool ok = decodeRawHeadsCUDA(cls_output, box_output, dir_output,
                                           cls_dims, box_dims, dir_dims, cfg, bboxes_full, &decode_error);
        if (!ok) {
            bboxes_full.clear();
            NODELET_WARN_THROTTLE(1.0,
            "Raw-head decode failed: %s", decode_error.c_str());
        }
    }

    void postProcessing(std::vector<float>& output, int& num_class, float& nms_thr,
                        float& score_thr, int& max_num, std::vector<Box3dfull>& bboxes_full)
    {
        std::vector<Box2d> valid_boxes_2d;
        std::vector<Box3d> valid_boxes_3d;
        std::vector<float> valid_scores;
        std::vector<int> valid_classes;
        std::vector<float> valid_directions;

        int n_type = 11;
        int num_boxes = output.size() / n_type;

        for (int i = 0; i < num_boxes; i++) {
            int offset = i * n_type;
            float score = output[offset + 7];
            if (score <= score_thr) {
                continue;
            }

            float x = output[offset + 0];
            float y = output[offset + 1];
            float z = output[offset + 2];
            float dx = output[offset + 3];
            float dy = output[offset + 4];
            float dz = output[offset + 5];
            float theta = output[offset + 6];
            int class_id = static_cast<int>(output[offset + 8]);
            float direction = output[offset + 10];

            Box2d box2d;
            box2d.x1 = x - dx / 2.0f;
            box2d.y1 = y - dy / 2.0f;
            box2d.x2 = x + dx / 2.0f;
            box2d.y2 = y + dy / 2.0f;
            box2d.theta = theta;

            Box3d box3d;
            box3d.x = x;
            box3d.y = y;
            box3d.z = z;
            box3d.w = dy; // width perpendicular to local heading
            box3d.l = dx; // length along local heading
            box3d.h = dz;
            box3d.theta = theta;

            valid_boxes_2d.push_back(box2d);
            valid_boxes_3d.push_back(box3d);
            valid_scores.push_back(score);
            valid_classes.push_back(class_id);
            valid_directions.push_back(direction);
        }

        if (valid_boxes_2d.empty()) {
            bboxes_full.clear();
            return;
        }

        std::vector<int> nms_filter_inds;
        nms(valid_boxes_2d, valid_scores, nms_thr, nms_filter_inds);

        std::vector<Box3dfull> bboxes_3d_nms;
        for (const auto ind : nms_filter_inds) {
            Box3dfull box3d_full;
            box3d_full.x = valid_boxes_3d[ind].x;
            box3d_full.y = valid_boxes_3d[ind].y;
            box3d_full.z = valid_boxes_3d[ind].z;
            box3d_full.w = valid_boxes_3d[ind].w;
            box3d_full.l = valid_boxes_3d[ind].l;
            box3d_full.h = valid_boxes_3d[ind].h;
            float limited_theta = limitPeriod(valid_boxes_3d[ind].theta);
            box3d_full.theta = (1.f - valid_directions[ind]) * M_PI + limited_theta;
            box3d_full.score = valid_scores[ind];
            box3d_full.label = valid_classes[ind];
            bboxes_3d_nms.push_back(box3d_full);
        }

        getTopkBoxes(bboxes_3d_nms, max_num, bboxes_full);
    }

    uint16_t labelToClassId(int label)
    {
        const int idx = label - class_id_offset_;
        if (idx < 0 || idx >= static_cast<int>(class_names_.size())) {
            return nuport_perception_msgs::ObjectClassification::UNKNOWN;
        }
        return lookupClassIdByName(class_names_[idx]);
    }

    void initClassMap()
    {
        class_name_to_label_.clear();
        class_name_to_label_["Car"] = nuport_perception_msgs::ObjectClassification::CAR;
        class_name_to_label_["Truck"] = nuport_perception_msgs::ObjectClassification::TRUCK;
        class_name_to_label_["Bus"] = nuport_perception_msgs::ObjectClassification::BUS;
        class_name_to_label_["Construction"] = nuport_perception_msgs::ObjectClassification::HEAVY_DUTY_VEHICLE;
        class_name_to_label_["Motorcycle"] = nuport_perception_msgs::ObjectClassification::MOTORCYCLE;
        class_name_to_label_["Bicycle"] = nuport_perception_msgs::ObjectClassification::BICYCLE;
        class_name_to_label_["Trailer"] = nuport_perception_msgs::ObjectClassification::TRAILER;
        class_name_to_label_["EmergencyVehicle"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Train"] = nuport_perception_msgs::ObjectClassification::HEAVY_DUTY_VEHICLE;
        class_name_to_label_["OtherVehicle"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["EgoTrailer"] = nuport_perception_msgs::ObjectClassification::TRAILER;
        class_name_to_label_["Pedestrian"] = nuport_perception_msgs::ObjectClassification::PEDESTRIAN;
        class_name_to_label_["Animal"] = nuport_perception_msgs::ObjectClassification::ANIMAL;
        class_name_to_label_["TrafficCone"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Barrier"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["PushablePullable"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Debris"] = nuport_perception_msgs::ObjectClassification::DEBRIS;
        class_name_to_label_["BicycleRack"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["TrafficSign"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;

        class_name_to_label_["Traffic_sign"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Ego_trailer"] = nuport_perception_msgs::ObjectClassification::TRAILER;
        class_name_to_label_["Trafficcone"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Other"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Bicycle_rack"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Emergency"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["Pushable_pullable"] = nuport_perception_msgs::ObjectClassification::UNKNOWN;
        class_name_to_label_["animal"] = nuport_perception_msgs::ObjectClassification::ANIMAL;

        NODELET_INFO("Loaded %zu class names with class_id_offset=%d", class_names_.size(), class_id_offset_);
        NODELET_INFO("Encoder tensor bindings: pillars=%s coords=%s num_points=%s output=%s",
                 tensor_name_encoder_pillars_.c_str(), tensor_name_encoder_coords_.c_str(),
                 tensor_name_encoder_num_points_.c_str(), tensor_name_encoder_output_.c_str());
        NODELET_INFO("Backbone tensor bindings: input=%s cls=%s box=%s dir=%s",
                 tensor_name_backbone_input_.c_str(), tensor_name_backbone_cls_output_.c_str(),
                 tensor_name_backbone_box_output_.c_str(), tensor_name_backbone_dir_output_.c_str());
        NODELET_INFO("Preprocess params: z_shift=%.3f xy_center=%s",
                 z_shift_, xy_center_ ? "true" : "false");
    }

    std::string sanitizeClassName(const std::string& name) const
    {
        std::string out = name;
        out.erase(std::remove(out.begin(), out.end(), ' '), out.end());
        return out;
    }

    uint16_t lookupClassIdByName(const std::string& raw_name) const
    {
        auto it = class_name_to_label_.find(raw_name);
        if (it != class_name_to_label_.end()) {
            return it->second;
        }
        const std::string normalized = sanitizeClassName(raw_name);
        it = class_name_to_label_.find(normalized);
        if (it != class_name_to_label_.end()) {
            return it->second;
        }
        return nuport_perception_msgs::ObjectClassification::UNKNOWN;
    }

    // Members
    ros::Subscriber pc_sub_;
    ros::Publisher det_pub_;

    TRTLogger trt_logger_;
    nvinfer1::IRuntime* encoder_runtime_;
    nvinfer1::ICudaEngine* encoder_engine_;
    nvinfer1::IExecutionContext* encoder_context_;
    nvinfer1::IRuntime* backbone_runtime_;
    nvinfer1::ICudaEngine* backbone_engine_;
    nvinfer1::IExecutionContext* backbone_context_;

    std::string input_topic_;
    std::string output_topic_;
    float score_thr_;
    float nms_thr_;
    int max_num_;
    int num_class_;
    int num_box_;
    int max_points_;
    int max_voxels_;
    int NDim_;
    int point_feature_dim_;
    std::vector<float> voxel_size_;
    std::vector<float> coors_range_;
    std::string anchor_prototypes_file_;
    std::string tensor_name_encoder_pillars_;
    std::string tensor_name_encoder_coords_;
    std::string tensor_name_encoder_num_points_;
    std::string tensor_name_encoder_output_;
    std::string tensor_name_backbone_input_;
    std::string tensor_name_backbone_cls_output_;
    std::string tensor_name_backbone_box_output_;
    std::string tensor_name_backbone_dir_output_;
    int class_id_offset_;
    float z_shift_;
    bool xy_center_ = false;
    float grid_step_ = 0.4f;
    float last_ox_ = 0.0f;
    float last_oy_ = 0.0f;
    std::vector<float> anchor_prototypes_;
    std::vector<std::string> class_names_;
    std::unordered_map<std::string, uint16_t> class_name_to_label_;
};

PLUGINLIB_EXPORT_CLASS(PointPillarsNodelet, nodelet::Nodelet)

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointpillars_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    PointPillarsNodelet node;
    node.init("pointpillars", {}, {}, nullptr, nullptr);
    ros::spin();
    return 0;
}
