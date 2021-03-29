/*
 * ICPOdometry.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_

#include "../Cuda/internal.h"

#include <sophus/se3.hpp>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

class CameraIntrinsics {
    float fx, cx, fy, cy;
public:
    CameraIntrinsics(float fx, float cx, float fy, float cy);

    float getFx() const;
    float getCx() const;
    float getFy() const;
    float getCy() const;
};
class ICPOdometry {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ICPOdometry(int width, int height,
              float distThresh = 0.10f,
              float angleThresh = sinf(20.f * 3.14159254f / 180.f));

  virtual ~ICPOdometry();

  void setIntrFromCamera(const CameraIntrinsics &cameraIntrinsics, bool isICPModel);

  void initICP(unsigned short *depth, const CameraIntrinsics &cameraIntrinsics, const float depthCutoff = 20.0f);

  void initICPModel(unsigned short *depth, const CameraIntrinsics &cameraIntrinsics, const float depthCutoff = 20.0f);

  void getIncrementalTransformation(Sophus::SE3d &T_prev_curr,
                                    bool useICPModelIntr,
                                    int threads, int blocks,
                                    int iterations0 = 10,
                                    int iterations1 = 5,
                                    int iterations2 = 4);

  float lastError;
  float lastInliers;

private:
  std::vector<DeviceArray2D<unsigned short>> depth_tmp;

  std::vector<DeviceArray2D<float>> vmaps_prev;
  std::vector<DeviceArray2D<float>> nmaps_prev;

  std::vector<DeviceArray2D<float>> vmaps_curr;
  std::vector<DeviceArray2D<float>> nmaps_curr;

  Intr intrICP;
  Intr intrICPModel;

  DeviceArray<Eigen::Matrix<float, 29, 1, Eigen::DontAlign>> sumData;
  DeviceArray<Eigen::Matrix<float, 29, 1, Eigen::DontAlign>> outData;

  static const int NUM_PYRS = 3;

  std::vector<int> iterations;

  float dist_thresh;
  float angle_thresh;

  const int width;
  const int height;
};

#endif /* ICPODOMETRY_H_ */
