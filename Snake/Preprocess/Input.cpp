/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Input.h"

#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/vision/camera/all.h"
#include "saiga/vision/opencv/opencv.h"

#include "Preprocess/StereoTransforms.h"
#include "opencv2/imgproc.hpp"

namespace Snake {
/**
 * Input的构造函数
 */
Input::Input() : Module(ModuleType::INPUT) {
    if (settings.inputType == InputType::Mono) {
        settings.datasetParameters.force_monocular = true;
    } else {
        settings.datasetParameters.force_monocular = false;
    }

    CreateCamera();

    switch (settings.inputType) {
    case InputType::Mono:
        SAIGA_ASSERT(mono_intrinsics.model.K.fx != 1);
        stereo_cam = mono_intrinsics.dummyStereoCamera();
        break;
    case InputType::Stereo:
        SAIGA_ASSERT(stereo_intrinsics.model.K.fx != 1);
        mono_intrinsics = stereo_intrinsics;

        stereo_cam = stereo_intrinsics.stereoCamera();
        th_depth = stereo_intrinsics.maxDepth;
        break;
    case InputType::RGBD:
        SAIGA_ASSERT(rgbd_intrinsics.model.K.fx != 1);
        mono_intrinsics = rgbd_intrinsics;
        stereo_cam = rgbd_intrinsics.stereoCamera();
        th_depth = rgbd_intrinsics.maxDepth;
        break;
    }

    if (camera_mono)
        globalCamera = camera_mono.get();
    if (camera_stereo)
        globalCamera = camera_stereo.get();
    if (camera_rgbd)
        globalCamera = camera_rgbd.get();

    // 双目
    if (settings.inputType == InputType::Stereo) {
        // 若畸变系数的平方范数 != 0，需要极线矫正
        bool need_rectify = stereo_intrinsics.model.dis.Coeffs().squaredNorm() != 0;

        if (need_rectify) {
            std::cout << "需进行极线矫正" << std::endl;

            // 极线校正，计算矫正后 左右目的 新内参矩阵、未校正相机坐标系到矫正相机坐标系的旋转矩阵、矫正后的bfx
            Rectify();

            // 更新相机采参数 和 双目相机对象
            stereo_intrinsics.bf = rect_right.bf; // 校正后的bfx
            stereo_cam.bf = rect_right.bf;
            mono_intrinsics.model.K = rect_left.K_dst; // 校正后的内参
            mono_intrinsics.model.dis = Distortion();

            stereo_cam = StereoCamera4(rect_left.K_dst, stereo_intrinsics.bf); // 用校正后的内参、bfx 构建双目相机
        } else {
            // 左右目相机的投影矩阵设为单位阵
            rect_left.Identity(stereo_intrinsics.model.K, stereo_intrinsics.bf);
            rect_right.Identity(stereo_intrinsics.rightModel.K, stereo_intrinsics.bf);
            // 重新构造双目相机对象
            stereo_cam = stereo_intrinsics.stereoCamera();
        }

        SAIGA_ASSERT(stereo_intrinsics.bf != 0);
        SAIGA_ASSERT(stereo_cam.bf != 0);
    }
    // 单目
    else {
        rect_left.K_dst = K;
        rect_left.K_src = K;
        rect_left.D_src = mono_intrinsics.model.dis;
    }

    std::cout << mono_intrinsics << std::endl;

    // 根据单目相机的图像尺寸和畸变参数，计算特征点边界框的大小和网格的行列数
    featureGridBounds.computeFromIntrinsicsDist(mono_intrinsics.imageSize.w, mono_intrinsics.imageSize.h, rect_left.K_dst, rect_left.D_src);

    CreateTable({7, 10, 10}, {"Frame", "Time (s)", "Load (ms)"});
    SAIGA_ASSERT(output_table);
}

void Input::CreateCamera() {
    switch (settings.sensorType) {
    case SensorType::PRIMESENSE: {
#ifdef SAIGA_USE_OPENNI2
        VLOG(0) << "Input set to Asus Primesense.";

        SAIGA_ASSERT(settings.inputType == InputType::RGBD);
        rgbd_intrinsics.fromConfigFile(settings.config_file);
        auto c = std::make_unique<Saiga::RGBDCameraOpenni>(rgbd_intrinsics);
        rgbd_intrinsics = c->intrinsics();
        camera_rgbd = std::move(c);
        break;
#else
        SAIGA_EXIT_ERROR("Openni not found.");
        break;
#endif
    }
    case SensorType::SAIGA_RAW: {
        SAIGA_ASSERT(settings.inputType == InputType::Mono || settings.inputType == InputType::RGBD);
        auto c = std::make_unique<Saiga::SaigaDataset>(settings.datasetParameters, true);
        if (settings.inputType == InputType::RGBD)
            rgbd_intrinsics = c->intrinsics();
        else
            mono_intrinsics = c->intrinsics();
        has_imu = true;
        imu = c->getIMU().value();
        is_dataset = true;
        camera_rgbd = std::move(c);
        break;
    }
    case SensorType::TUM_RGBD: {
        SAIGA_ASSERT(settings.inputType == InputType::Mono || settings.inputType == InputType::RGBD);
        auto c = std::make_unique<Saiga::TumRGBDDataset>(settings.datasetParameters);
        if (settings.inputType == InputType::RGBD)
            rgbd_intrinsics = c->intrinsics();
        else
            mono_intrinsics = c->intrinsics();
        camera_rgbd = std::move(c);
        is_dataset = true;
        break;
    }
    case SensorType::SCANNET: {
        VLOG(0) << "Input set to Scannet dataset.";

        SAIGA_ASSERT(settings.inputType == InputType::Mono || settings.inputType == InputType::RGBD);
        auto c = std::make_unique<Saiga::ScannetDataset>(settings.datasetParameters);
        if (settings.inputType == InputType::RGBD)
            rgbd_intrinsics = c->intrinsics();
        else
            mono_intrinsics = c->intrinsics();
        camera_rgbd = std::move(c);
        is_dataset = true;
        break;
    }
    case SensorType::ZJU: {
        SAIGA_ASSERT(settings.inputType == InputType::Mono);
        auto c = std::make_unique<Saiga::ZJUDataset>(settings.datasetParameters);
        c->saveGroundTruthTrajectory(settings.evalDir + settings.out_file_prefix + "_gt.tum");
        c->saveGroundTruthTrajectory(settings.evalDir + "/frames/" + settings.out_file_prefix + "_gt.tum");
        mono_intrinsics = c->intrinsics;
        has_imu = true;
        imu = c->getIMU().value();
        camera_mono = std::move(c);
        is_dataset = true;
        break;
    }
    case SensorType::EUROC: {
        SAIGA_ASSERT(settings.inputType == InputType::Mono || settings.inputType == InputType::Stereo);
        auto c = std::make_unique<Saiga::EuRoCDataset>(settings.datasetParameters);
        c->saveGroundTruthTrajectory(settings.evalDir + settings.out_file_prefix + "_gt.tum");
        c->saveGroundTruthTrajectory(settings.evalDir + "/frames/" + settings.out_file_prefix + "_gt.tum");

        if (settings.inputType == InputType::Stereo)
            stereo_intrinsics = c->intrinsics;
        else
            mono_intrinsics = c->intrinsics;
        SAIGA_ASSERT(c->getIMU().has_value());
        has_imu = true;
        imu = c->getIMU().value();
        is_dataset = true;
        camera_stereo = std::move(c);
        break;
    }
    case SensorType::KITTI: {
        SAIGA_ASSERT(settings.inputType == InputType::Mono || settings.inputType == InputType::Stereo);
        auto c = std::make_unique<Saiga::KittiDataset>(settings.datasetParameters);
        c->saveGroundTruthTrajectory(settings.evalDir + settings.out_file_prefix + "_gt.tum");
        c->saveGroundTruthTrajectory(settings.evalDir + "/frames/" + settings.out_file_prefix + "_gt.tum");
        if (settings.inputType == InputType::Stereo)
            stereo_intrinsics = c->intrinsics;
        else
            mono_intrinsics = c->intrinsics;
        camera_stereo = std::move(c);
        is_dataset = true;
        break;
    }
#ifdef SAIGA_USE_K4A
    case SensorType::KINECT_AZURE: {
        SAIGA_ASSERT(settings.inputType == InputType::Mono || settings.inputType == InputType::RGBD);
        KinectCamera::KinectParams k_params;
        k_params.color = false;
        auto c = std::make_unique<Saiga::KinectCamera>(k_params);
        if (settings.inputType == InputType::RGBD)
            rgbd_intrinsics = c->intrinsics();
        else
            mono_intrinsics = c->intrinsics();
        has_imu = true;
        imu = c->getIMU().value();
        camera_rgbd = std::move(c);
        is_dataset = false;
        break;
    }
#endif
    default:
        SAIGA_EXIT_ERROR("Invalid Sensor ID");
    }

    if (has_imu) {
        SAIGA_ASSERT(imu.frequency > 0);
        SAIGA_ASSERT(imu.frequency_sqrt > 0);
        std::cout << imu << std::endl;
    }
    //    exit(0);
}

void Input::run() {
    running = true;
    // 创建 CameraInput 线程，读取图像帧
    camera_thread = ScopedThread([this]() {
        Saiga::Random::setSeed(settings.randomSeed);
        Saiga::setThreadName("CameraInput");

        std::shared_ptr<Frame> last_frame;
        auto dataset_rgbd = dynamic_cast<DatasetCameraBase *>(camera_rgbd.get());
        auto dataset_mono = dynamic_cast<DatasetCameraBase *>(camera_mono.get());
        auto dataset_stereo = dynamic_cast<DatasetCameraBase *>(camera_stereo.get());

        if (dataset_rgbd)
            dataset_rgbd->ResetTime();
        if (dataset_mono)
            dataset_mono->ResetTime();
        if (dataset_stereo)
            dataset_stereo->ResetTime();

        // 主循环
        while (running) {
            // 读取下一帧图像 frame
            std::shared_ptr<Frame> frame;
            {
                auto timer = ModuleTimer();
                frame = ReadNextFrame();
            }

            if (!frame || stop_camera) {
                std::cout << "Camera Disconnected." << std::endl;
                running = false;
                break;
            }

#if 0
            // 随机丢弃一些帧，默认该段代码被注释
            if ((frame->id > 500 && Random::sampleBool(0.1)) || frame->id == 200)
            {
                frame->image.makeZero();
                frame->image_rgb.makeZero();
            }
#endif

            // 建立当前帧和上一帧间的关系
            if (store_previous_frame) {
                frame->previousFrame = last_frame;
            }
            last_frame = frame;

            if (first_image) {
                start_timestamp = frame->timeStamp;
                first_image = false;
            }
            // 帧ID、当前帧相对于初始帧的时间戳、上次处理的时间 记录到 output_table
            (*output_table) << frame->id << (frame->timeStamp - start_timestamp) << LastTime();

            // 将当前帧 frame 发送到 camera_slot 中
            camera_slot.set(frame);

            // 如果 pause 为真,则等待一毫秒后再继续
            while (pause && running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        camera_slot.set(nullptr);
    });

    // 创建 Grayscale 线程，将帧转为灰度图，并将其添加到 output_buffer 中
    process_image_thread = ScopedThread([this]() {
        Saiga::Random::setSeed(settings.randomSeed);
        Saiga::setThreadName("Grayscale");
        // 主循环
        while (true) {
            // 从 camera_slot 中获取帧
            auto frame = camera_slot.get();

            if (!frame) {
                break;
            }

            float gray_time;

            {
                Saiga::ScopedTimer timer(gray_time);
                computeGrayscaleImage(*frame); // 转为灰度图
            }

            // 处理后的灰度图 帧放到 output_buffer 中
            output_buffer.add(frame);
        }
        output_buffer.add(nullptr);
    });
}

/**
 * 读取下一帧
 * @return
 */
FramePtr Input::ReadNextFrame() {
    FramePtr frame = Saiga::make_aligned_shared<Frame>(frameId++);
    if (camera_rgbd) {
        FrameData img;
        auto gotImage = camera_rgbd->getImageSync(img);
        if (!gotImage) {
            return nullptr;
        }

        frame->imu_data = img.imu_data;

        frame->timeStamp = img.timeStamp;
        frame->groundTruth = img.groundTruth;
        frame->id = img.id;
        frame->image_rgb = std::move(img.image_rgb);
        frame->image = std::move(img.image);
        frame->depth_image = std::move(img.depth_image);
    } else if (camera_mono) {
        FrameData img;
        auto gotImage = camera_mono->getImageSync(img);
        if (!gotImage) {
            return nullptr;
        }
        frame->imu_data = img.imu_data;

        frame->timeStamp = img.timeStamp;
        frame->groundTruth = img.groundTruth;
        frame->id = img.id;
        frame->image_rgb = std::move(img.image_rgb);
        frame->image = std::move(img.image);
    }
    // 双目模式
    else if (camera_stereo) {
        FrameData img;

        auto gotImage = camera_stereo->getImageSync(img);
        if (!gotImage) {
            return nullptr;
        }
        frame->imu_data = img.imu_data;

        frame->timeStamp = img.timeStamp;
        frame->groundTruth = img.groundTruth;
        frame->id = img.id;
        frame->image_rgb = std::move(img.image_rgb);
        frame->image = std::move(img.image);

        frame->right_image = std::move(img.right_image);
        frame->right_image_rgb = std::move(img.right_image_rgb);
    }

    //    std::cout << std::setprecision(20) << frame->timeStamp << " " << frame->imu_frame.time_end << " "
    //              << frame->imu_frame.data.back().timestamp << std::endl;

    //    SAIGA_ASSERT(frame->timeStamp == frame->imu_frame.time_end);
    //    SAIGA_ASSERT(frame->timeStamp == frame->imu_frame.data.back().timestamp);
    return frame;
}

/**
 * RGB转为灰度度
 * @param frame
 */
void Input::computeGrayscaleImage(Frame &frame) {
    // 帧的没有彩色图像 或 灰度图像已存在，则不进行处理，直接返回
    if (frame.image_rgb.rows == 0 || frame.image.rows > 0) {
        SAIGA_ASSERT(frame.image.valid());
        return;
    }

    // 创建灰度图像
    frame.image.create(frame.image_rgb.rows, frame.image_rgb.cols);
#if 1
    // 使用OpenCV库进行灰度转换
    cv::setNumThreads(0);
    cv::Mat4b cv_src = Saiga::ImageViewToMat(frame.image_rgb.getImageView());
    cv::Mat1b cv_dst = Saiga::ImageViewToMat(frame.image.getImageView());
    cv::cvtColor(cv_src, cv_dst, cv::COLOR_RGBA2GRAY);
    std::string filename = "/home/liuzhi/Project/Optimization_Results/snake_img/" + std::to_string(frame.id) + ".png";
    cv::imwrite(filename, cv_dst);
#else
    ImageTransformation::RGBAToGray8(frame.image_rgb, frame.image);
#endif
}

} // namespace Snake
