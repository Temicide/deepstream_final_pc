#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"
#include <cmath>
#include <algorithm>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

extern "C" bool NvDsInferParseYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "Could not find output layer" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];
    
    // YOLOv8 output shape is [84, 8400] where 84 = 4 (bbox) + 80 (classes)
    int num_classes = 80;
    int num_anchors = 8400; // YOLOv8n with 640x640 input typically has 8400 anchors
    int dimensions = 4 + num_classes; // 84

    if (output.inferDims.numElements != dimensions * num_anchors) {
        // Fallback or warning if shapes don't match perfectly, but we will calculate based on elements
        num_anchors = output.inferDims.numElements / dimensions;
    }

    float* outputData = (float*)output.buffer;

    for (int i = 0; i < num_anchors; ++i) {
        float max_prob = 0.0f;
        int max_index = -1;

        // Find class with max probability
        for (int c = 0; c < num_classes; ++c) {
            float prob = outputData[(4 + c) * num_anchors + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_index = c;
            }
        }

        // Apply threshold
        if (max_prob > detectionParams.perClassPreclusterThreshold[max_index]) {
            float xc = outputData[0 * num_anchors + i];
            float yc = outputData[1 * num_anchors + i];
            float w = outputData[2 * num_anchors + i];
            float h = outputData[3 * num_anchors + i];

            float x1 = xc - w / 2.0f;
            float y1 = yc - h / 2.0f;

            NvDsInferParseObjectInfo obj;
            obj.classId = max_index;
            obj.detectionConfidence = max_prob;
            obj.left = CLIP(x1, 0, networkInfo.width - 1);
            obj.top = CLIP(y1, 0, networkInfo.height - 1);
            obj.width = CLIP(w, 0, networkInfo.width - 1);
            obj.height = CLIP(h, 0, networkInfo.height - 1);

            objectList.push_back(obj);
        }
    }

    return true;
}

extern "C" bool NvDsInferParseCustomYolo(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {
    return NvDsInferParseYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}
