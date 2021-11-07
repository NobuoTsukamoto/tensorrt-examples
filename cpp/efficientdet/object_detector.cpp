/**
 * Copyright (c) 2020 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <iterator>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>

#include "common.h"
#include "logger.h"

#include "object_detector.h"

ObjectDetector::ObjectDetector(const int input_width, const int input_heigth, const float score_threshold)
    : input_width_(input_width)
    , input_height_(input_heigth)
    , score_threshold_(score_threshold)
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
}

bool ObjectDetector::LoadEngine(const std::string &model_path)
{

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (runtime_ == nullptr)
    {
        return false;
    }

    std::vector<unsigned char> buffer;
    std::ifstream stream(model_path, std::ios::binary);
    if (!stream)
    {
        return false;
    }
    stream >> std::noskipws;
    std::copy(std::istream_iterator<unsigned char>(stream),
              std::istream_iterator<unsigned char>(),
              back_inserter(buffer));
    stream.close();


    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (engine_ == nullptr)
    {
        return false;
    }
    std::cout << "engine->hasImplicitBatchDimension(): " << engine_->hasImplicitBatchDimension()<< std::endl;

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (context_ != nullptr)
    {
        return false;
    }
    return true;
}

std::unique_ptr<std::vector<BoundingBox>> ObjectDetector::RunInference(
    const cv::Mat &input_data,
    std::chrono::duration<double, std::milli> &time_span)
{
    const auto &start_time = std::chrono::steady_clock::now();

    auto results = std::make_unique<std::vector<BoundingBox>>();

    // Create RAII buffer manager object
    if (buffers == nullptr)
    {
        std::cout << "New buffers" << std::endl;
        buffers = std::make_unique<samplesCommon::BufferManager>(engine_, batch_size_);
    }

    // resize and normalize.
    cv::Mat resize_im, rgb_im, normalize_im;

    cv::resize(input_data, resize_im, cv::Size(input_width_, input_height_));
    cv::cvtColor(resize_im, rgb_im, cv::COLOR_BGR2RGB);
    rgb_im.convertTo(normalize_im, CV_32FC3);

    // Fill data buffer
    float *host_data_buffer = static_cast<float *>(buffers->getHostBuffer("images:0"));

    // Host memory for input buffer
    memcpy(host_data_buffer, normalize_im.data, normalize_im.elemSize() * normalize_im.total());

    // Memcpy from host input buffers to device input buffers
    buffers->copyInputToDevice();

    auto status = context_->execute(batch_size_, buffers->getDeviceBindings().data());
    if (status)
    {
        // Memcpy from device output buffers to host output buffers
        buffers->copyOutputToHost();

        const auto num_detections = static_cast<const int32_t*>(buffers->getHostBuffer("num_detections"));
        const auto detection_boxes = static_cast<const float*>(buffers->getHostBuffer("detection_boxes"));
        const auto detection_scores = static_cast<const float*>(buffers->getHostBuffer("detection_scores"));
        const auto detection_classes = static_cast<const int32_t*>(buffers->getHostBuffer("detection_classes"));

        for (auto i = 0; i < *num_detections; i++)
        {
            if (detection_scores[i] >= score_threshold_)
            {
                auto bounding_box = std::make_unique<BoundingBox>();
                auto y0 = detection_boxes[4 * i + 0];
                auto x0 = detection_boxes[4 * i + 1];
                auto y1 = detection_boxes[4 * i + 2];
                auto x1 = detection_boxes[4 * i + 3];

                bounding_box->class_id = (int)detection_classes[i];
                bounding_box->scores = detection_scores[i];
                bounding_box->x = x0;
                bounding_box->y = y0;
                bounding_box->width = x1 - x0;
                bounding_box->height = y1 - y0;
                bounding_box->center_x = bounding_box->x + (bounding_box->width / 2.0f);
                bounding_box->center_y = bounding_box->y + (bounding_box->height / 2.0f);

#if 0
                std::cout << "index     : " << i << std::endl;
                std::cout << "  class_id: " << bounding_box->class_id << std::endl;
                std::cout << "  scores  : " << bounding_box->scores << std::endl;
                std::cout << "  x       : " << bounding_box->x << std::endl;
                std::cout << "  y       : " << bounding_box->y << std::endl;
                std::cout << "  width   : " << bounding_box->width << std::endl;
                std::cout << "  height  : " << bounding_box->height << std::endl;
                std::cout << "  center  : " << bounding_box->center_x << ", " << bounding_box->center_y << std::endl;
                std::cout << "  y       : " << bounding_box->y << std::endl;
#endif
                results->emplace_back(std::move(*bounding_box));
            }
        }
    }
    else
    {
        std::cout << "context_->execute return" << status << std::endl;
    }

    time_span =
        std::chrono::steady_clock::now() - start_time;

    return results;
}

const int ObjectDetector::Width() const
{
    return input_width_;
}

const int ObjectDetector::Height() const
{
    return input_height_;
}

const int ObjectDetector::Channels() const
{
    return input_channels_;
}

