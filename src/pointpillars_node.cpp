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
#include <unordered_map>

#include "common_trt.h"
#include "io_trt.hpp"
#include "voxelization_trt.h"
#include "utils.h"
#include "nms_trt.h"

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
    PointPillarsNodelet() : engine_(nullptr), context_(nullptr) {}

    ~PointPillarsNodelet()
    {
        delete context_;
        delete engine_;
    }

    void onInit() override
    {
        ros::NodeHandle& nh  = getNodeHandle();
        ros::NodeHandle& pnh = getPrivateNodeHandle();

        // Parameters
        std::string engine_path;
        pnh.param<std::string>("engine_path", engine_path, "");
        pnh.param<std::string>("input_topic", input_topic_, "/points");
        pnh.param<std::string>("output_topic", output_topic_, "/detections");
        pnh.param<float>("score_threshold", score_thr_, 0.1f);
        pnh.param<float>("nms_threshold", nms_thr_, 0.01f);
        pnh.param<int>("max_detections", max_num_, 50);
        pnh.param<int>("num_classes", num_class_, 19);
        pnh.param<int>("num_boxes", num_box_, 1000);
        pnh.param<int>("max_points_per_voxel", max_points_, 32);
        pnh.param<int>("max_voxels", max_voxels_, 40000);
        pnh.param<std::string>("tensor_name_pillars", tensor_name_pillars_, "input_pillars");
        pnh.param<std::string>("tensor_name_coords", tensor_name_coords_, "input_coors_batch");
        pnh.param<std::string>("tensor_name_num_points", tensor_name_num_points_, "input_npoints_per_pillar");
        pnh.param<int>("class_id_offset", class_id_offset_, 0);
        pnh.param<float>("z_shift", z_shift_, 0.0f);
        pnh.param<bool>("xy_center", xy_center_, false);

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

        if (engine_path.empty()) {
            NODELET_FATAL("Parameter 'engine_path' is required");
            return;
        }

        // Initialize TensorRT
        if (!initTRT(engine_path)) {
            NODELET_FATAL("Failed to initialize TensorRT engine");
            return;
        }

        // Publishers and subscribers
        det_pub_ = nh.advertise<nuport_perception_msgs::ObjectDetectionArray>(output_topic_, 1);
        pc_sub_ = nh.subscribe(input_topic_, 1, &PointPillarsNodelet::pointCloudCallback, this);

        NODELET_INFO("PointPillars nodelet initialized. Subscribing to: %s, Publishing to: %s",
                     input_topic_.c_str(), output_topic_.c_str());
    }

private:
    bool initTRT(const std::string& engine_path)
    {
        bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
        if (!didInitPlugins) {
            NODELET_WARN("Failed to initialize TensorRT plugins");
        }

        std::vector<char> engineData;
        try {
            engineData = readEngineFile(engine_path);
        } catch (const std::runtime_error& e) {
            NODELET_ERROR("Failed to read engine file: %s", e.what());
            return false;
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(trt_logger_);
        engine_ = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine_) {
            NODELET_ERROR("Failed to deserialize CUDA engine");
            return false;
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            NODELET_ERROR("Failed to create execution context");
            return false;
        }

        return true;
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

        // TRT inference
        std::vector<float> output(num_box_ * 11);
        trtInfer(d_voxels, d_coors_padded, d_num_points_per_voxel, voxel_num,
                 max_points_, engine_, context_, output);

        // Post processing
        std::vector<Box3dfull> bboxes;
        postProcessing(output, num_class_, nms_thr_, score_thr_, max_num_, bboxes);

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

    void trtInfer(float* d_voxels, int* d_coors, int* d_num_points_per_voxel,
                  const int pillar_num, int& max_points,
                  nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context,
                  std::vector<float>& output)
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

        void* outputDevice;
        CHECK_CUDA(cudaMalloc(&outputDevice, output.size() * sizeof(float)));

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

        context->setInputShape(tensor_name_pillars_.c_str(), inputPillarDims);
        context->setInputShape(tensor_name_coords_.c_str(), inputCoorsDims);
        context->setInputShape(tensor_name_num_points_.c_str(), inputNpointsPerPillarDims);

        context->setTensorAddress(tensor_name_pillars_.c_str(), inputPillarsDevice);
        context->setTensorAddress(tensor_name_coords_.c_str(), inputCoorsBatchDevice);
        context->setTensorAddress(tensor_name_num_points_.c_str(), inputNpointsPerPillarDevice);
        context->setTensorAddress("output_x", outputDevice);

        context->enqueueV3(0);

        cudaMemcpy(output.data(), outputDevice, output.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(inputPillarsDevice);
        cudaFree(inputCoorsBatchDevice);
        cudaFree(inputNpointsPerPillarDevice);
        cudaFree(outputDevice);
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
        NODELET_INFO("Tensor bindings: pillars=%s coords=%s num_points=%s",
                     tensor_name_pillars_.c_str(), tensor_name_coords_.c_str(), tensor_name_num_points_.c_str());
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
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

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
    std::vector<float> voxel_size_;
    std::vector<float> coors_range_;
    std::string tensor_name_pillars_;
    std::string tensor_name_coords_;
    std::string tensor_name_num_points_;
    int class_id_offset_;
    float z_shift_;
    bool xy_center_ = false;
    float last_ox_ = 0.0f;
    float last_oy_ = 0.0f;
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
