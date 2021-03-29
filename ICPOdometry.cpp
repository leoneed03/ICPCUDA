/*
 * ICPOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include "ICPOdometry.h"

ICPOdometry::ICPOdometry(int width, int height,
                         float distThresh, float angleThresh)
        : lastError(0), lastInliers(width * height), dist_thresh(distThresh),
          angle_thresh(angleThresh), width(width), height(height) {
    sumData.create(MAX_THREADS);
    outData.create(1);

//    intr.cx = cx;
//    intr.cy = cy;
//    intr.fx = fx;
//    intr.fy = fy;

    iterations.reserve(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_prev.resize(NUM_PYRS);
    nmaps_prev.resize(NUM_PYRS);

    vmaps_curr.resize(NUM_PYRS);
    nmaps_curr.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i) {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create(pyr_rows, pyr_cols);

        vmaps_prev[i].create(pyr_rows * 3, pyr_cols);
        nmaps_prev[i].create(pyr_rows * 3, pyr_cols);

        vmaps_curr[i].create(pyr_rows * 3, pyr_cols);
        nmaps_curr[i].create(pyr_rows * 3, pyr_cols);
    }
}

ICPOdometry::~ICPOdometry() {}

void ICPOdometry::setIntrFromCamera(const CameraIntrinsics &cameraIntrinsics, bool isICPModel) {

    if (isICPModel) {
        intrICPModel.fx = cameraIntrinsics.getFx();
        intrICPModel.cx = cameraIntrinsics.getCx();
        intrICPModel.fy = cameraIntrinsics.getFy();
        intrICPModel.cy = cameraIntrinsics.getCy();
    } else {
        intrICP.fx = cameraIntrinsics.getFx();
        intrICP.cx = cameraIntrinsics.getCx();
        intrICP.fy = cameraIntrinsics.getFy();
        intrICP.cy = cameraIntrinsics.getCy();
    }
}
void ICPOdometry::initICP(unsigned short *depth, const CameraIntrinsics &cameraIntrinsics, const float depthCutoff) {
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

    setIntrFromCamera(cameraIntrinsics, false);

    for (int i = 1; i < NUM_PYRS; ++i) {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for (int i = 0; i < NUM_PYRS; ++i) {
        createVMap(intrICP(i), depth_tmp[i], vmaps_curr[i], depthCutoff);
        createNMap(vmaps_curr[i], nmaps_curr[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::initICPModel(unsigned short *depth, const CameraIntrinsics &cameraIntrinsics, const float depthCutoff) {
    depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

    setIntrFromCamera(cameraIntrinsics, true);

    for (int i = 1; i < NUM_PYRS; ++i) {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for (int i = 0; i < NUM_PYRS; ++i) {
        createVMap(intrICPModel(i), depth_tmp[i], vmaps_prev[i], depthCutoff);
        createNMap(vmaps_prev[i], nmaps_prev[i]);
    }

    cudaDeviceSynchronize();
}

void ICPOdometry::getIncrementalTransformation(Sophus::SE3d &T_prev_curr,
                                               bool useICPModelIntr,
                                               int threads, int blocks,
                                               int iterations0,
                                               int iterations1,
                                               int iterations2) {
    iterations[0] = iterations0;
    iterations[1] = iterations1;
    iterations[2] = iterations2;

    for (int i = NUM_PYRS - 1; i >= 0; i--) {
        for (int j = 0; j < iterations[i]; j++) {
            float residual_inliers[2];
            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            const Intr &intr = (useICPModelIntr) ? (intrICPModel) : (intrICP);
            estimateStep(T_prev_curr.rotationMatrix().cast<float>().eval(),
                         T_prev_curr.translation().cast<float>().eval(),
                         vmaps_curr[i], nmaps_curr[i], intr(i), vmaps_prev[i],
                         nmaps_prev[i], dist_thresh, angle_thresh, sumData, outData,
                         A_icp.data(), b_icp.data(), &residual_inliers[0], threads,
                         blocks);

            lastError = sqrt(residual_inliers[0]) / residual_inliers[1];
            lastInliers = residual_inliers[1];

            const Eigen::Matrix<double, 6, 1> update =
                    A_icp.cast<double>().ldlt().solve(b_icp.cast<double>());

            T_prev_curr = Sophus::SE3d::exp(update) * T_prev_curr;
        }
    }
}

CameraIntrinsics::CameraIntrinsics(float fxToSet, float cxToSet, float fyToSet, float cyToSet) :
        fx(fxToSet),
        cx(cxToSet),
        fy(fyToSet),
        cy(cyToSet) {}

float CameraIntrinsics::getFx() const {
    return fx;
}

float CameraIntrinsics::getCx() const {
    return cx;
}

float CameraIntrinsics::getFy() const {
    return fy;
}

float CameraIntrinsics::getCy() const {
    return cy;
}
