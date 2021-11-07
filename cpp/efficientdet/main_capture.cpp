/**
 * Copyright (c) 2021 Nobuo Tsukamoto
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 */

#include <fstream>
#include <iostream>
#include <map>
#include <list>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

#include "object_detector.h"

const cv::String kKeys =
    "{help h usage ? |    | show help command.}"
    "{f file         |    | path to video file.}"
    "{s score        |0.5 | score threshold.}"
    "{w width        |512 | input model width.}"
    "{H height       |512 | input model height.}"
    "{l label        |.   | path to label file.}"
    "{o output       |    | output video file path.}"
    "{@input         |    | path to trt engine file.}"
    ;

const cv::String kWindowName = "TensorRT Efficentdet example.";
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
    cv::String video_file = "";
    if (parser.has("file"))
    {
        video_file = parser.get<cv::String>("file");
    }
    cv::String output_file = "";
    if (parser.has("output"))
    {
        output_file = parser.get<cv::String>("output");
    }
    if (!parser.check()) {
        parser.printErrors();
        return 1;
    }

    std::cout << "model path             : " << model_path << std::endl;
    std::cout << "input video file path  : " << video_file << std::endl;
    std::cout << "output video file path : " << output_file << std::endl;
    std::cout << "label path             : " << label_path << std::endl;
    std::cout << "input width            : " << input_width << std::endl;
    std::cout << "input height           : " << input_height << std::endl;
    std::cout << "score threshold        : " << score_threshold << std::endl;

    // Create Object detector
    auto detector = std::make_unique<ObjectDetector>(input_width, input_height, score_threshold);

    detector->LoadEngine(model_path);
    auto width = detector->Width();
    auto height = detector->Height();

    // Load label file
    auto labels = ReadLabelFile(label_path);

    // Window setting
    cv::namedWindow(kWindowName,
        cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
    cv::moveWindow(kWindowName, 100, 100);

    // Videocapture setting.
    cv::VideoCapture cap(video_file);
    if (video_file.empty())
    {
      cap.open(0);
    }
    else
    {
      // cap.open(video_file);
      std::cout << "openfile : " << video_file << std::endl;
    }
    auto cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    auto scale_width = (double)cap_width / input_width;
    auto scale_height = (double)cap_height / input_height;
    auto fps = (double)cap.get(cv::CAP_PROP_FPS);

    std::cout << "Start capture." << " isOpened: " << std::boolalpha << cap.isOpened() << std::endl;
    std::cout << "scale width  : " << scale_width << std::endl;
    std::cout << "scale height : " << scale_height << std::endl;

    // Output Videocaputre setting.
    cv::VideoWriter writer;
    if (!output_file.empty())   
    {
        auto codec = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
        writer.open(output_file, codec, fps, cv::Size(cap_width, cap_height));
        if (!writer.isOpened())
        {
            std::cout << "Could not open the output video file for write." << std::endl;
        }
    }

    std::list<double> elapsed_list;

    while(cap.isOpened())
    {
        const auto& start_time = std::chrono::steady_clock::now();
        cv::Mat frame, input_im;

        cap >> frame;

        // Run inference.
        std::chrono::duration<double, std::milli> inference_time_span;

        const auto& result = detector->RunInference(frame, inference_time_span);
        std::chrono::duration<double, std::milli> time_span = std::chrono::steady_clock::now() - start_time;

        for (const auto& object : *result)
        {
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

        time_caption << "Inference: " << std::fixed << std::setprecision(2) << inference_time_span.count() << " ms";
        elapsed_list.emplace_back(inference_time_span.count());
        if (elapsed_list.size() > 100)
        {
            elapsed_list.pop_front();

            auto ave = std::accumulate(std::begin(elapsed_list), std::end(elapsed_list), 0.0) / elapsed_list.size();
            time_caption << ", AVG: " << std::fixed << std::setprecision(2) << ave << " ms";
        }
        DrawCaption(frame, cv::Point(10, 30), time_caption.str());

        // Output video file.
        if (writer.isOpened())
        {
            writer.write(frame);
        }

        cv::imshow(kWindowName, frame);
        // Handle the keyboard before moving to the next frame
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
        {
            break;  // Escape
        }

    }

    cap.release();
    writer.release();

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
