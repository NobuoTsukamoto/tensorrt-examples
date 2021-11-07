/**
 * Copyright (c) 2021 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#ifndef OBJECT_DETECTOR_H_
#define OBJECT_DETECTOR_H_

#include <chrono>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include <NvInfer.h>

#include "buffers.h"

class BoundingBox
{
public:
    int class_id = 0;
    float scores = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float center_x = 0.0f;
    float center_y = 0.0f;
};

class ObjectDetector
{
public:
    ObjectDetector(const int input_width, const int input_heigth, const float score_threshold);

    bool LoadEngine(const std::string& model_path);

    std::unique_ptr<std::vector<BoundingBox>> RunInference(
        const cv::Mat& input_data,
        std::chrono::duration<double, std::milli>& time_span);

    const int Width() const;
    const int Height() const;
    const int Channels() const;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<samplesCommon::BufferManager> buffers;


    int32_t batch_size_ = 1;
    int32_t input_width_ = 512;
    int32_t input_height_ = 512;
    int32_t input_channels_ = 3;
    float score_threshold_ = 0.5f;

};

#endif /* OBJECT_DETECTOR_H_ */
