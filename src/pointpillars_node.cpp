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
#include <tf/transform_datatypes.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include "NvInferPlugin.h"
#include <cuda_runtime.h>

#include "common_trt.h"
#include "io_trt.hpp"
#include "voxelization_trt.h"
#include "utils.h"
#include "nms_trt.h"

class Logger : public nvinfer1::ILogger
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
} gLogger;

#define CHECK_CUDA(status) \
    if (status != 0) \
    { \
        ROS_FATAL("CUDA failure: %d", status); \
        ros::shutdown(); \
    }

class PointPillarsNode
{
public:
    PointPillarsNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    {
        // Parameters
        std::string engine_path;
        pnh.param<std::string>("engine_path", engine_path, "");
        pnh.param<std::string>("input_topic", input_topic_, "/points");
        pnh.param<std::string>("output_topic", output_topic_, "/detections");
        pnh.param<float>("score_threshold", score_thr_, 0.1f);
        pnh.param<float>("nms_threshold", nms_thr_, 0.01f);
        pnh.param<int>("max_detections", max_num_, 50);
        pnh.param<int>("num_classes", num_class_, 3);
        pnh.param<int>("num_boxes", num_box_, 100);
        pnh.param<int>("max_points_per_voxel", max_points_, 32);
        pnh.param<int>("max_voxels", max_voxels_, 40000);

        voxel_size_ = {0.16f, 0.16f, 4.0f};
        coors_range_ = {0.0f, -39.68f, -3.0f, 69.12f, 39.68f, 1.0f};
        NDim_ = 3;

        if (engine_path.empty()) {
            ROS_FATAL("Parameter 'engine_path' is required");
            ros::shutdown();
            return;
        }

        // Initialize TensorRT
        if (!initTRT(engine_path)) {
            ROS_FATAL("Failed to initialize TensorRT engine");
            ros::shutdown();
            return;
        }

        // Publishers and subscribers
        det_pub_ = nh.advertise<nuport_perception_msgs::ObjectDetectionArray>(output_topic_, 1);
        pc_sub_ = nh.subscribe(input_topic_, 1, &PointPillarsNode::pointCloudCallback, this);

        ROS_INFO("PointPillars node initialized. Subscribing to: %s, Publishing to: %s",
                 input_topic_.c_str(), output_topic_.c_str());
    }

    ~PointPillarsNode()
    {
        delete context_;
        delete engine_;
    }

private:
    bool initTRT(const std::string& engine_path)
    {
        bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
        if (!didInitPlugins) {
            ROS_WARN("Failed to initialize TensorRT plugins");
        }

        std::vector<char> engineData;
        try {
            engineData = readEngineFile(engine_path);
        } catch (const std::runtime_error& e) {
            ROS_ERROR("Failed to read engine file: %s", e.what());
            return false;
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        engine_ = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine_) {
            ROS_ERROR("Failed to deserialize CUDA engine");
            return false;
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            ROS_ERROR("Failed to create execution context");
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

        // Filter points to valid range
        std::vector<Point> points;
        pointCloudFiler(points_ori, points);

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
        std::vector<float> output(num_box_ * (7 + num_class_ + 1));
        trtInfer(d_voxels, d_coors_padded, d_num_points_per_voxel, voxel_num,
                 max_points_, engine_, context_, output);

        // Post processing
        std::vector<Box3dfull> bboxes;
        postProcessing(output, num_class_, nms_thr_, score_thr_, max_num_, bboxes);

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
            tf::Quaternion q;
            q.setRPY(0, 0, box.theta);
            det.kinematics.pose.pose.orientation.x = q.x();
            det.kinematics.pose.pose.orientation.y = q.y();
            det.kinematics.pose.pose.orientation.z = q.z();
            det.kinematics.pose.pose.orientation.w = q.w();

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

        context->setInputShape("input_pillars", inputPillarDims);
        context->setInputShape("input_coors_batch", inputCoorsDims);
        context->setInputShape("input_npoints_per_pillar", inputNpointsPerPillarDims);

        context->setTensorAddress("input_pillars", inputPillarsDevice);
        context->setTensorAddress("input_coors_batch", inputCoorsBatchDevice);
        context->setTensorAddress("input_npoints_per_pillar", inputNpointsPerPillarDevice);
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
        std::vector<Box2d> bboxes_2d;
        std::vector<Box3d> bboxes_3d;
        std::vector<std::vector<float>> scores_list;
        std::vector<float> direction_list;
        decodeDetResults(output, num_class, bboxes_2d, bboxes_3d, scores_list, direction_list);

        std::vector<Box3dfull> bboxes_3d_nms;
        for (int i = 0; i < num_class; i++) {
            std::vector<int> score_filter_inds;
            std::vector<float> scores;
            filterByScores(i, scores_list, score_thr, score_filter_inds, scores);
            std::vector<Box2d> bboxes_2d_filtered;
            std::vector<Box3d> bboxes_3d_filtered;
            std::vector<float> direction_filtered;
            obtainBoxByInds(score_filter_inds, bboxes_2d, bboxes_2d_filtered, bboxes_3d,
                           bboxes_3d_filtered, direction_list, direction_filtered);

            std::vector<int> nms_filter_inds;
            nms(bboxes_2d_filtered, scores, nms_thr, nms_filter_inds);

            for (const auto ind : nms_filter_inds) {
                Box3dfull box3d_full;
                box3d_full.x = bboxes_3d_filtered[ind].x;
                box3d_full.y = bboxes_3d_filtered[ind].y;
                box3d_full.z = bboxes_3d_filtered[ind].z;
                box3d_full.w = bboxes_3d_filtered[ind].w;
                box3d_full.l = bboxes_3d_filtered[ind].l;
                box3d_full.h = bboxes_3d_filtered[ind].h;
                float limited_theta = limitPeriod(bboxes_3d_filtered[ind].theta);
                box3d_full.theta = (1.f - direction_filtered[ind]) * M_PI + limited_theta;
                box3d_full.score = scores[ind];
                box3d_full.label = i;
                bboxes_3d_nms.push_back(box3d_full);
            }
        }
        getTopkBoxes(bboxes_3d_nms, max_num, bboxes_full);
    }

    uint16_t labelToClassId(int label)
    {
        // Map model class indices to nuport_perception_msgs classification labels
        // Model classes: Car, Truck, Bus, Construction, Motorcycle, Bicycle, Trailer,
        //   EmergencyVehicle, Train, OtherVehicle, EgoTrailer, Pedestrian, Animal,
        //   TrafficCone, Barrier, PushablePullable, Debris, BicycleRack, TrafficSign
        switch (label) {
            case 0:  return nuport_perception_msgs::ObjectClassification::CAR;
            case 1:  return nuport_perception_msgs::ObjectClassification::TRUCK;
            case 2:  return nuport_perception_msgs::ObjectClassification::BUS;
            case 3:  return nuport_perception_msgs::ObjectClassification::HEAVY_DUTY_VEHICLE; // Construction
            case 4:  return nuport_perception_msgs::ObjectClassification::MOTORCYCLE;
            case 5:  return nuport_perception_msgs::ObjectClassification::BICYCLE;
            case 6:  return nuport_perception_msgs::ObjectClassification::TRAILER;
            case 7:  return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // EmergencyVehicle
            case 8:  return nuport_perception_msgs::ObjectClassification::HEAVY_DUTY_VEHICLE; // Train
            case 9:  return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // OtherVehicle
            case 10: return nuport_perception_msgs::ObjectClassification::TRAILER;            // EgoTrailer
            case 11: return nuport_perception_msgs::ObjectClassification::PEDESTRIAN;
            case 12: return nuport_perception_msgs::ObjectClassification::ANIMAL;
            case 13: return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // TrafficCone
            case 14: return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // Barrier
            case 15: return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // PushablePullable
            case 16: return nuport_perception_msgs::ObjectClassification::DEBRIS;
            case 17: return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // BicycleRack
            case 18: return nuport_perception_msgs::ObjectClassification::UNKNOWN;            // TrafficSign
            default: return nuport_perception_msgs::ObjectClassification::UNKNOWN;
        }
    }

    // Members
    ros::Subscriber pc_sub_;
    ros::Publisher det_pub_;

    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

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
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointpillars_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    PointPillarsNode node(nh, pnh);
    ros::spin();
    return 0;
}
