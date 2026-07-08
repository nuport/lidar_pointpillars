

# lidar_pointpillars

ROS package for real-time 3D object detection from LiDAR point clouds using the [PointPillars](https://arxiv.org/abs/1812.05784) architecture, with TensorRT-accelerated inference.

## Overview

The node subscribes to a `sensor_msgs/PointCloud2` topic, voxelizes the incoming point cloud into pillars on the GPU, runs TensorRT inference, applies NMS post-processing, and publishes detected objects as `nuport_perception_msgs/ObjectDetectionArray`.

## Dependencies

- ROS (tested with ROS Noetic)
- CUDA
- TensorRT (default path: `/usr/local/TensorRT`, override with `TENSORRT_ROOT` env var)
- `nuport_perception_msgs`

## Building

```bash
cd ~/catkin_ws
catkin_make --pkg lidar_pointpillars
```

## Rename ONNX IO Tensors

```bash
roscd lidar_pointpillars/models
python /home/nuport/catkin_ws/src/autoware_lidar_transfusion/scripts/rename_onnx_outputs.py --map-inputs '{ "voxels" : "input_pillars", "coords" : "input_coors_batch", "voxel_num_points" : "input_npoints_per_pillar" }' pointpillar_mantruck_topk.onnx pointpillar_mantruck_topk_rename.onnx 
```

## Compiling the TensorRT engine

Convert the ONNX model to a TensorRT engine before running the node. The command below builds an engine with dynamic shapes optimised for ~5000 pillars:

```bash
/usr/src/tensorrt/bin/trtexec
  --onnx=./model.onnx
  --saveEngine=./model.trt
  --minShapes=input_pillars:200x32x4,input_coors_batch:200x4,input_npoints_per_pillar:200
  --maxShapes=input_pillars:40000x32x4,input_coors_batch:40000x4,input_npoints_per_pillar:40000
  --optShapes=input_pillars:5000x32x4,input_coors_batch:5000x4,input_npoints_per_pillar:5000

/usr/src/tensorrt/bin/trtexec
  --onnx=onnx/pointpillar_combined_v2_ep20.onnx
  --saveEngine=engines/pointpillar_combined_v2.trt
  --minShapes=voxels:200x20x4,coords:200x4,voxel_num_points:200
  --optShapes=voxels:5000x20x4,coords:5000x4,voxel_num_points:5000
  --maxShapes=voxels:40000x20x4,coords:40000x4,voxel_num_points:40000
  --fp16

/usr/src/tensorrt/bin/trtexec --onnx=onnx/backbone_rawhead_int8_qdq.onnx \
        --saveEngine=engines/backbone_rawhead_int8.trt \
        --fp16 --noDataTransfers --useCudaGraph --int8

/usr/src/tensorrt/bin/trtexec --onnx=onnx/encoder_fp32.onnx \
        --saveEngine=engines/encoder_fp16.trt \
        --fp16 --minShapes=voxels:200x20x4,coords:200x4,voxel_num_points:200 \
               --optShapes=voxels:5000x20x4,coords:5000x4,voxel_num_points:5000 \
               --maxShapes=voxels:40000x20x4,coords:40000x4,voxel_num_points:40000 \
        --noDataTransfers --useCudaGraph
```

The compiled engine is hardware-specific — it must be regenerated on every new machine or after a TensorRT version upgrade.

## Running

```bash
roslaunch lidar_pointpillars pointpillars.launch
```

## Anchor Prototypes For Raw-Head Decode

When using split encoder/backbone runtime with raw-head outputs, the ROS decoder
needs anchor prototypes (`n_anchors x 7`). If they are missing, you will see:

`Raw-head decode failed: anchor_prototypes size mismatch...`

Generate anchors from a decoded-head ONNX and write them to the default file:

```bash
python scripts/extract_anchor_prototypes.py \
  --decoded_onnx /path/to/backbone_decoded_fp32.onnx \
  --output config/anchor_prototypes_38x7.txt
```

The launch file already passes this path as `anchor_prototypes_file` by default.
You can override it:

```bash
roslaunch lidar_pointpillars pointpillars.launch \
  anchor_file:=/absolute/path/to/anchors.txt
```

By default the launch file replays a rosbag. For live sensor use, override the topic argument:

```bash
roslaunch lidar_pointpillars pointpillars.launch cloud_topic:=/your/lidar/topic
```

## ROS Interface

### Subscribed topics

| Topic | Type | Description |
|---|---|---|
| `/points` (default) | `sensor_msgs/PointCloud2` | Input point cloud (XYZI, intensity optional) |

### Published topics

| Topic | Type | Description |
|---|---|---|
| `/detections` (default) | `nuport_perception_msgs/ObjectDetectionArray` | 3D bounding box detections with class and score |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `engine_path` | *(required)* | Path to the compiled `.trt` engine file |
| `input_topic` | `/points` | Point cloud input topic |
| `output_topic` | `/detections` | Detection output topic |
| `score_threshold` | `0.1` | Minimum detection confidence score |
| `nms_threshold` | `0.01` | IoU threshold for Non-Maximum Suppression |
| `max_detections` | `50` | Maximum number of output detections per frame |
| `num_classes` | `3` | Number of object classes the model predicts |
| `num_boxes` | `100` | Number of raw anchor boxes from the model head |
| `max_points_per_voxel` | `32` | Maximum points sampled per pillar |
| `max_voxels` | `40000` | Maximum number of non-empty pillars per frame |

## Voxelization settings

The voxel grid is currently hard-coded in the node:

| Setting | Value |
|---|---|
| Voxel size (x, y, z) | 0.16 m, 0.16 m, 4.0 m |
| Detection range (x) | 0.0 – 69.12 m |
| Detection range (y) | −39.68 – 39.68 m |
| Detection range (z) | −3.0 – 1.0 m |