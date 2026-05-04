

# Compiling model

/usr/src/tensorrt/bin/trtexec --onnx=./model.onnx --saveEngine=./model.trt --minShapes=input_pillars:200x32x4,input_coors_batch:200x4,input_npoints_per_pillar:200 --maxShapes=input_pillars:40000x32x4,input_coors_batch:40000x4,input_npoints_per_pillar:40000 --optShapes=input_pillars:5000x32x4,input_coors_batch:5000x4,input_npoints_per_pillar:5000