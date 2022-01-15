/**
 * Copyright (c) 2021 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <fstream>
#include <iostream>
#include <map>

#include <opencv2/opencv.hpp>

#include <boost/algorithm/string.hpp>

#include "object_detector.h"

const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{s score        |0.5 | score threshold.}"
    "{w width        |512 | input model width.}"
    "{H height       |512 | input model height.}"
    "{l label        |.   | path to label file.}"
    "{@input         |    | path to trt engine file.}"
    "{@image         |    | path to image engine file.}"
    ;

const cv::Scalar kWhiteColor = cv::Scalar(246, 250, 250);
const cv::Scalar kBuleColor = cv::Scalar(255, 209, 0);

std::unique_ptr<std::map<long, std::string>> ReadLabelFile(const std::string& label_path)
{
    auto labels = std::make_unique<std::map<long, std::string>>();

    std::ifstream ifs(label_path);
    if (ifs.is_open())
    {
        std::string label = "";
        while (std::getline(ifs, label))
        {
            std::vector<std::string> result;

            boost::algorithm::split(result, label, boost::is_any_of(" ")); // Split by space.
            if (result.size() < 2)
            {
                std::cout << "Expect 2-D input label (" << result.size() << ")." << std::endl;
                continue;
            }
             
            auto label_string = result[2];
            for (size_t i = 3; i < result.size(); i++)
            {
                label_string += " " + result[i];
            }
            auto id = std::stol(result[0]);
            labels->insert(std::make_pair(id, label_string));
        }
    }
    else
    {
        std::cout << "Label file not found. : " << label_path << std::endl;
    }
    return labels;
}

void DrawCaption(
    cv::Mat& im,
    const cv::Point& point,
    const std::string& caption)
{
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

int main(int argc, char* argv[]) try
{
    // Argument parsing
    cv::String model_path;
    cv::String image_path;
    cv::CommandLineParser parser(argc, argv, kKeys);
    if (parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }
    auto input_width = parser.get<int>("width");
    auto input_height = parser.get<int>("height");
    auto score_threshold = parser.get<float>("score");
    auto label_path = parser.get<cv::String>("label");
    if (parser.has("@input"))
    {
        model_path = parser.get<cv::String>("@input");
    }
    else
    {
        std::cout << "No model file path." << std::endl;
        return 0;
    }
    if (parser.has("@image"))
    {
        image_path = parser.get<cv::String>("@image");
    }
    else
    {
        std::cout << "No image file path." << std::endl;
        return 0;
    }
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }
    std::cout << "model path      : " << model_path << std::endl;
    std::cout << "image path      : " << image_path << std::endl;
    std::cout << "label path      : " << label_path << std::endl;
    std::cout << "input width     : " << input_width << std::endl;
    std::cout << "input height    : " << input_height << std::endl;
    std::cout << "score threshold : " << score_threshold << std::endl;

    // Create Object detector
    auto detector = std::make_unique<ObjectDetector>(input_width, input_height, score_threshold);

    detector->LoadEngine(model_path);
    auto width = detector->Width();
    auto height = detector->Height();

    // Load label file
    auto labels = ReadLabelFile(label_path);

    // Load image
    cv::Mat frame, input_im;
    frame = cv::imread(image_path);
    int cap_width = frame.cols;
    int cap_height = frame.rows;
    auto scale_width = (double)cap_width / input_width;
    auto scale_height = (double)cap_height / input_height;

    std::cout << "scale width  : " << scale_width << std::endl;
    std::cout << "scale height : " << scale_height << std::endl;

    // Run inference.
    std::chrono::duration<double, std::milli> inference_time_span;

    const auto& result = detector->RunInference(frame, inference_time_span);

    cv::resize(frame, frame, cv::Size(input_width, input_height));
    for (const auto& object : *result)
    {
        /*
        auto x = int(object.x * cap_width / input_width);
        auto y = int(object.y * cap_height / input_heigth);
        auto w = int(object.width * cap_width / input_width);
        auto h = int(object.height * cap_height / input_heigth);
        */
        auto x = int(object.x * scale_width);
        auto y = int(object.y * scale_height);
        auto w = int(object.width * scale_width);
        auto h = int(object.height * scale_height);

        // Draw bounding box
        cv::rectangle(frame, cv::Rect(x, y, w, h), kBuleColor, 2);

        // Draw Caption
        std::ostringstream caption;

        auto it = labels->find(object.class_id);
        if (it != std::end(*labels))
        {
            caption << it->second;
        }
        else
        {
            caption << "ID: " << std::to_string(object.class_id);
        }
        caption << "(" << std::fixed << std::setprecision(2) << object.scores << ")";
        DrawCaption(frame, cv::Point(x, y), caption.str());
    }

    // Calc fps and draw fps and inference time.
    std::ostringstream time_caption;

    time_caption << std::fixed << std::setprecision(2) << inference_time_span.count() << " ms";
    DrawCaption(frame, cv::Point(10, 60), time_caption.str());

    // Output Image
    cv::imwrite("output.png", frame);

    return EXIT_SUCCESS;

}
catch (const cv::Exception& e)
{
    std::cerr << "OpenCV error calling :\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
