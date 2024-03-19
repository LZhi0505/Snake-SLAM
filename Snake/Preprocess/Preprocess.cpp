/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Preprocess.h"

#include "FeatureDetector.h"
#include "Input.h"
#include "opencv2/imgproc.hpp"
namespace Snake
{
Preprocess::Preprocess() : Module(ModuleType::PREPROCESS)
{
    thread = Thread([this]() {
        Saiga::Random::setSeed(settings.randomSeed);
        Saiga::setThreadName("Preprocess");
        while (true) {
            //! 获取下一个帧的数据
            auto frame = featureDetector->GetFrame();

            if (!frame) {
                break;
            }
            // 处理帧
            Process(*frame);
            // 向输出缓冲区中添加一个空指针，以表示处理结束
            output_buffer.set(frame);
        }
        // 向输出缓冲区中添加一个空指针，以表示处理结束
        output_buffer.set(nullptr);
    });
    // 创建一个表格，列名分别为 "Frame", "Stereo P.", "Time (ms)"
    CreateTable({7, 12, 10}, {"Frame", "Stereo P.", "Time (ms)"});
}

void Preprocess::Process(Frame& frame)
{
    int stereo_matches = 0;
    {
        auto timer = ModuleTimer();
        frame.allocateTmp();
        // 关键点去畸变
        undistortKeypoints(frame);
        // 计算特征点网格，并重新排列特征点顺序
        computeFeatureGrid(frame);

        if (settings.inputType == InputType::RGBD) {
            stereo_matches = ComputeStereoFromRGBD(frame);
        }
        // 双目立体匹配
        else if (settings.inputType == InputType::Stereo) {
            stereo_matches = StereoMatching(frame);
        }
    }
    (*output_table) << frame.id << stereo_matches << LastTime();
}

/**
 * 计算当前帧关键帧点的 去畸变后 的归一化平面x,y坐标 和 像素坐标
 * @param frame
 */
void Preprocess::undistortKeypoints(Frame& frame)
{
    // 确保畸变校正特征点数组为空
    SAIGA_ASSERT(frame.undistorted_keypoints.size() == 0);

    // 遍历每个关键点
    for (int i = 0; i < frame.N; i++) {
        // 将关键点添加到 去畸变关键点数组中
        frame.undistorted_keypoints.emplace_back(frame.keypoints[i]);

        // 该关键点坐标
        Vec2 p = frame.keypoints[i].point;
        SAIGA_ASSERT(frame.image.inImage(p.y(), p.x()));    // 检查是否在图像范围内

        // 对该关键点进行去畸变
        // 有畸变的像素坐标 根据有畸变的内参 转为 有畸变的归一化平面x，y坐标
        p = rect_left.K_src.unproject2(p);
        //                Vec2 p_normalized = undistortNormalizedPoint(p, rect_left.D_src);
        // 去畸变：使用高斯牛顿方法来优化去畸变的效果
        Vec2 p_normalized = undistortPointGN(p, p, rect_left.D_src);

        // 通过左目的旋转矩阵 变换到 相机坐标x,y,1
        Vec3 p_r = rect_left.R * Vec3(p_normalized(0), p_normalized(1), 1);
        // 重新计算到归一化平面x,y坐标
        p                                    = Vec2(p_r(0) / p_r(2), p_r(1) / p_r(2));
        frame.normalized_points[i]           = p;
        // 通过左目的内参矩阵 转换到 像素坐标系下
        p                                    = rect_left.K_dst.normalizedToImage(p);
        frame.undistorted_keypoints[i].point = p;
    }
}

int Preprocess::ComputeStereoFromRGBD(Frame& frame)
{
    SAIGA_ASSERT(settings.inputType == InputType::RGBD);
    SAIGA_ASSERT(frame.depth_image.valid());

    auto N = frame.N;


    int num_matches = 0;


    for (int i = 0; i < N; i++)
    {
        auto& kpun = frame.undistorted_keypoints[i];

        Vec2 normalized_point = K.unproject2(kpun.point);
        Vec2 distorted_point  = distortNormalizedPoint(normalized_point, rgbd_intrinsics.depthModel.dis);
        Vec2 reprojected      = rgbd_intrinsics.depthModel.K.normalizedToImage(distorted_point);

        int x = reprojected.x() + 0.5;
        int y = reprojected.y() + 0.5;

        SAIGA_ASSERT(frame.depth_image.inImage(y, x));
        auto depth = frame.depth_image(y, x);

        SAIGA_ASSERT(depth >= 0);
        SAIGA_ASSERT(depth < 20);
        if (depth > 0)
        {
            frame.depth[i]        = depth;
            auto disparity        = rgbd_intrinsics.bf / depth;
            frame.right_points[i] = kpun.point(0) - disparity;
            num_matches++;
        }
        else
        {
            frame.depth[i]        = -1;
            frame.right_points[i] = -1;
        }
    }
    return num_matches;
}

/**
 *
 * @param frame
 * @return
 */
int Preprocess::StereoMatching(Frame& frame)
{
    SAIGA_ASSERT(settings.inputType == InputType::Stereo);
    SAIGA_ASSERT(frame.keypoints.size() > 0);
    SAIGA_ASSERT(frame.keypoints_right.size() > 0);

    // Transform points to rectified space
    std::vector<KeyPoint> left_rectified  = frame.keypoints;
    std::vector<KeyPoint> right_rectified = frame.keypoints_right;

    int min_y = 1023123;
    int max_y = -19284;


    // Set limits for search
    const float min_disp = 0;
    const float max_disp = rect_left.bf * 0.5;

    for (int i = 0; i < left_rectified.size(); ++i)
    {
        left_rectified[i].point = rect_left.Forward(frame.keypoints[i].point);
    }

    for (int i = 0; i < right_rectified.size(); ++i)
    {
        right_rectified[i].point = rect_right.Forward(frame.keypoints_right[i].point);
        min_y                    = std::min(min_y, Saiga::iRound(right_rectified[i].point.y()));
        max_y                    = std::max(max_y, Saiga::iRound(right_rectified[i].point.y()));
    }

    std::vector<std::vector<int>> row_map(max_y - min_y + 1);
    for (int i = 0; i < right_rectified.size(); ++i)
    {
        int row = Saiga::iRound(right_rectified[i].point.y()) - min_y;
        SAIGA_ASSERT(row >= 0 && row < row_map.size());
        row_map[row].push_back(i);
    }

    int num_matches = 0;
    for (int i = 0; i < left_rectified.size(); ++i)
    {
        auto& kp = left_rectified[i];

        int y        = Saiga::iRound(left_rectified[i].point.y());
        int y_center = y - min_y;

        // go over +-r rows
        float r = ceil(2.0f * scalePyramid.Scale(kp.octave));

        int best_id          = -1;
        int best_dist        = 250;
        int second_best_dist = 250;

        for (int row = y_center - r; row <= y_center + r; ++row)
        {
            if (row < 0 || row >= row_map.size()) continue;

            for (auto other_id : row_map[row])
            {
                double disparity = left_rectified[i].point.x() - right_rectified[other_id].point.x();
                if (disparity < min_disp || disparity > max_disp)
                {
                    continue;
                }

                if (std::abs(left_rectified[i].octave - right_rectified[other_id].octave) > 1)
                {
                    continue;
                }

                auto dist = distance(frame.descriptors[i], frame.descriptors_right[other_id]);
                if (dist < best_dist)
                {
                    second_best_dist = best_dist;
                    best_dist        = dist;
                    best_id          = other_id;
                }
                else if (dist < second_best_dist)
                {
                    second_best_dist = dist;
                }
            }
        }


        if (best_dist > (settings.fd_relaxed_stereo ? 75 : 40)) continue;

        if (best_dist > (settings.fd_relaxed_stereo ? 0.9 : 0.7) * second_best_dist)
        {
            continue;
        }

        auto angle1 = left_rectified[i].angle;
        auto angle2 = right_rectified[best_id].angle;
        float rot   = std::min(std::abs(angle1 - angle2),
                             std::min(std::abs((angle1 + 365) - angle2), std::abs(angle1 - (angle2 + 365))));

        if (rot > (settings.fd_relaxed_stereo ? 25 : 5))
        {
            continue;
        }


        auto right_point = right_rectified[best_id].point.x();

        double disparity = left_rectified[i].point.x() - right_point;

        if (disparity <= 0.001)
        {
            disparity   = 0.001;
            right_point = left_rectified[i].point.x() - disparity;
        }

        frame.right_points[i] = right_point;
        frame.depth[i]        = rect_left.bf / disparity;
        num_matches++;

        SAIGA_ASSERT(frame.depth[i] > 0);
    }
    return num_matches;
}

/**
 * 计算特征点的网格索引
 * @param frame
 */
void Preprocess::computeFeatureGrid(Frame& frame)
{
    // 计算特征点的网格索引
    auto permutation = frame.grid.create(featureGridBounds, frame.undistorted_keypoints);

    int N = permutation.size();
    std::vector<KeyPoint> mvKeys2(N);
    std::vector<Saiga::DescriptorORB> descriptors2(N);
    std::vector<KeyPoint> mvKeysUn2(N);
    AlignedVector<Vec2> norm2(N);

    // 遍历当前帧每个特征点
    for (int i = 0; i < N; ++i)
    {
        // 跟据索引值 permutation[i]，将原始特征点、描述子、去畸变特征点和归一化坐标点按照新的顺序存储到相应容器中
        mvKeys2[permutation[i]]      = frame.keypoints[i];
        descriptors2[permutation[i]] = frame.descriptors[i];
        mvKeysUn2[permutation[i]]    = frame.undistorted_keypoints[i];
        norm2[permutation[i]]        = frame.normalized_points[i];
    }

    frame.keypoints.swap(mvKeys2);
    frame.descriptors.swap(descriptors2);
    frame.undistorted_keypoints.swap(mvKeysUn2);
    frame.normalized_points.swap(norm2);
}

}  // namespace Snake
