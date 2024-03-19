/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FeatureDetector.h"

#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/core/util/table.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vision/features/ORBExtractor.h"

#ifdef SNAKE_CUDA
#    include "saiga/cuda/imageProcessing/ORBExtractorGPU.h"
#endif

#include "Preprocess/Input.h"
#include "Map/Frame.h"
#include "opencv2/imgproc.hpp"

namespace Snake
{
FeatureDetector::FeatureDetector() : Module(ModuleType::FEATURE_DETECTOR)
{
    if (settings.fd_gpu)
    {
#ifdef SNAKE_CUDA
        extractorGPU2 =
            std::make_unique<ORBExtractorGPU>(settings.fd_features, settings.fd_scale_factor, settings.fd_levels,
                                              settings.fd_iniThFAST, settings.fd_minThFAST);
#else
        SAIGA_EXIT_ERROR("CUDA not found. GPU Feature Detector disabled.");
#endif
    }
    // 创建特征提取器
    else {
        extractor = std::make_unique<ORBExtractor>(settings.fd_features, settings.fd_scale_factor, settings.fd_levels,
                                                   settings.fd_iniThFAST, settings.fd_minThFAST, settings.fd_threads);
    }

    start_frame = settings.datasetParameters.startFrame;
    tmpDir      = settings.datasetParameters.dir + "/features/";

    if (settings.fd_bufferToFile) {
        std::filesystem::create_directory(tmpDir);
    }

    scalePyramid = ScalePyramid(settings.fd_levels, settings.fd_scale_factor);

    if (settings.fd_bufferToFile) {
        std::filesystem::create_directory(tmpDir);
    }

    featureDetectionThread = ScopedThread([this]() {
        Saiga::Random::setSeed(settings.randomSeed);
        Saiga::setThreadName("FeatureDetection");

        while (true) {
            float time;
            auto frame = input->GetFrame();
            if (!frame) {
                break;
            }

            {
                Saiga::ScopedTimer timer(time);
                // 检测特征点
                Detect(*frame);
            }

            (*output_table) << frame->id << frame->N << time;

            output_buffer.set(frame);
        }
        output_buffer.set(nullptr);
    });
    CreateTable({7, 10, 10}, {"Frame", "Features", "Time (ms)"});
}

FeatureDetector::~FeatureDetector() {}

/**
 *
 * @param frame
 */
void FeatureDetector::Detect(Frame& frame)
{
    auto timer = ModuleTimer();

    bool compute_left  = true;
    bool compute_right = settings.inputType == InputType::Stereo;

    std::string feature_file       = tmpDir + "/" + to_string(frame.id + start_frame) + ".features";
    std::string feature_file_right = tmpDir + "/" + to_string(frame.id + start_frame) + "_right.features";

    // 特征缓存文件存在，则直接读取，不再计算
    if (settings.fd_bufferToFile) {
        // 左目特征文件存在，则读取
        if (compute_left && std::filesystem::exists(feature_file)) {
            BinaryFile bf(feature_file, std::ios_base::in);
            bf >> frame.keypoints >> frame.descriptors;
            compute_left = false;
        }
        // 右目特征文件存在，则读取
        if (compute_right && std::filesystem::exists(feature_file_right)) {
            BinaryFile bf_right(feature_file_right, std::ios_base::in);
            bf_right >> frame.keypoints_right >> frame.descriptors_right;
            compute_right = false;
        }
    }

    // 提取左目
    if (compute_left) {
        std::vector<Saiga::KeyPoint<float> > mvKeys;
        if (settings.fd_gpu) {
#ifdef SNAKE_CUDA
            (*extractorGPU2).Detect(frame.image.getImageView(), mvKeys, frame.descriptors);
#endif
        }
        // 提取特征点和描述子：特征点缓存到mvKeys
        else {
            extractor->Detect(frame.image.getImageView(), mvKeys, frame.descriptors);
        }

        // 特征点类型从 Saiga::KeyPoint<float> 转换为 KeyPoint<double>，存储到 frame.keypoints 中
        frame.keypoints.reserve(mvKeys.size());
        for (auto& kp : mvKeys) {
            frame.keypoints.emplace_back(kp.cast<double>());
        }

        // 设置了缓存文件路径，则保存特征点和描述子
        if (settings.fd_bufferToFile)
        {
            BinaryFile bf(feature_file, std::ios_base::out);
            bf << frame.keypoints << frame.descriptors;
            std::cout << "Saving features to file: " << feature_file << std::endl;
        }
    }

    // 提取右目
    if (compute_right) {
        std::vector<Saiga::KeyPoint<float> > mvKeysRight;

        if (settings.fd_gpu) {
#ifdef SNAKE_CUDA
            (*extractorGPU2).Detect(frame.right_image.getImageView(), mvKeysRight, frame.descriptors_right);
#endif
        }
        else {
            extractor->Detect(frame.right_image.getImageView(), mvKeysRight, frame.descriptors_right);
        }

        frame.keypoints_right.reserve(mvKeysRight.size());
        for (auto& kp : mvKeysRight) {
            frame.keypoints_right.emplace_back(kp.cast<double>());
        }

        if (settings.fd_bufferToFile) {
            BinaryFile bf(feature_file_right, std::ios_base::out);
            bf << frame.keypoints_right << frame.descriptors_right;
            std::cout << "Saving features to file: " << feature_file_right << std::endl;
        }
    }
    // = 左目特征点个数
    frame.N = frame.keypoints.size();
}

}  // namespace Snake
