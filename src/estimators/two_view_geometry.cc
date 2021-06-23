// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "estimators/two_view_geometry.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <unordered_set>

#include "base/camera.h"
#include "base/essential_matrix.h"
#include "base/homography_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/essential_matrix.h"
#include "estimators/fundamental_matrix.h"
#include "estimators/homography_matrix.h"
#include "estimators/translation_transform.h"
#include "optim/loransac.h"
#include "optim/ransac.h"
#include "util/math.h"
#include "util/random.h"

namespace colmap {
namespace {

FeatureMatches ExtractInlierMatches(const FeatureMatches& matches,
                                    const size_t num_inliers,
                                    const std::vector<char>& inlier_mask) {
  FeatureMatches inlier_matches(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_matches[j] = matches[i];
      j += 1;
    }
  }
  return inlier_matches;
}

FeatureMatches ExtractOutlierMatches(const FeatureMatches& matches,
                                     const FeatureMatches& inlier_matches) {
  CHECK_GE(matches.size(), inlier_matches.size());

  std::unordered_set<std::pair<point2D_t, point2D_t>> inlier_matches_set;
  inlier_matches_set.reserve(inlier_matches.size());
  for (const auto& match : inlier_matches) {
    inlier_matches_set.emplace(match.point2D_idx1, match.point2D_idx2);
  }

  FeatureMatches outlier_matches;
  outlier_matches.reserve(matches.size() - inlier_matches.size());

  for (const auto& match : matches) {
    if (inlier_matches_set.count(
            std::make_pair(match.point2D_idx1, match.point2D_idx2)) == 0) {
      outlier_matches.push_back(match);
    }
  }

  return outlier_matches;
}

inline bool IsImagePointInBoundingBox(const Eigen::Vector2d& point,
                                      const double minx, const double maxx,
                                      const double miny, const double maxy) {
  return point.x() >= minx && point.x() <= maxx && point.y() >= miny &&
         point.y() <= maxy;
}

}  // namespace

void TwoViewGeometry::Invert() {
  F.transposeInPlace();
  E.transposeInPlace();
  H = H.inverse().eval();

  const Eigen::Vector4d orig_qvec = qvec;
  const Eigen::Vector3d orig_tvec = tvec;
  InvertPose(orig_qvec, orig_tvec, &qvec, &tvec);

  for (auto& match : inlier_matches) {
    std::swap(match.point2D_idx1, match.point2D_idx2);
  }
}

void TwoViewGeometry::EstimateCalibrated(const Camera& camera1,
                                         const FeatureKeypoints& keypoints1,
                                         const Camera& camera2,
                                         const FeatureKeypoints& keypoints2,
                                         const FeatureMatches& matches,
                                         const Options& options) {
  if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
    EstimateCalibratedGeometry(camera1, keypoints1, camera2, keypoints2,
                               matches, options);
  } else {
    std::cout << "The camera calibration is needed for both images."
              << std::endl;
    std::cout << "Please introduce the focal length and the principal point "
              << "and use the same camera for all images." << std::endl;
    exit(1);
  }
}

void TwoViewGeometry::EstimateInitCalibrated(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
    EstimateInitCalibratedGeometry(camera1, points1, camera2, points2, matches,
                                   options);
  } else {
    EstimateUncalibrated(camera1, points1, camera2, points2, matches, options);
  }
}

void TwoViewGeometry::EstimateMultiple(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  FeatureMatches remaining_matches = matches;
  std::vector<TwoViewGeometry> two_view_geometries;

  while (true) {
    TwoViewGeometry two_view_geometry;
    two_view_geometry.EstimateInitCalibrated(camera1, points1, camera2, points2,
                                             remaining_matches, options);
    if (two_view_geometry.config == ConfigurationType::DEGENERATE) {
      break;
    }

    if (options.multiple_ignore_watermark) {
      if (two_view_geometry.config != ConfigurationType::WATERMARK) {
        two_view_geometries.push_back(two_view_geometry);
      }
    } else {
      two_view_geometries.push_back(two_view_geometry);
    }

    remaining_matches = ExtractOutlierMatches(remaining_matches,
                                              two_view_geometry.inlier_matches);
  }

  if (two_view_geometries.empty()) {
    config = ConfigurationType::DEGENERATE;
  } else if (two_view_geometries.size() == 1) {
    *this = two_view_geometries[0];
  } else {
    config = ConfigurationType::MULTIPLE;

    for (const auto& two_view_geometry : two_view_geometries) {
      inlier_matches.insert(inlier_matches.end(),
                            two_view_geometry.inlier_matches.begin(),
                            two_view_geometry.inlier_matches.end());
    }
  }
}

bool TwoViewGeometry::EstimateRelativePose(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2) {
  // We need a valid epopolar geometry to estimate the relative pose.
  if (config != CALIBRATED && config != UNCALIBRATED && config != PLANAR &&
      config != PANORAMIC && config != PLANAR_OR_PANORAMIC) {
    return false;
  }

  // Extract normalized inlier points.
  std::vector<Eigen::Vector2d> inlier_points1_normalized;
  inlier_points1_normalized.reserve(inlier_matches.size());
  std::vector<Eigen::Vector2d> inlier_points2_normalized;
  inlier_points2_normalized.reserve(inlier_matches.size());
  for (const auto& match : inlier_matches) {
    const point2D_t idx1 = match.point2D_idx1;
    const point2D_t idx2 = match.point2D_idx2;
    inlier_points1_normalized.push_back(camera1.ImageToWorld(points1[idx1]));
    inlier_points2_normalized.push_back(camera2.ImageToWorld(points2[idx2]));
  }

  Eigen::Matrix3d R;
  std::vector<Eigen::Vector3d> points3D;

  if (config == CALIBRATED || config == UNCALIBRATED) {
    // Try to recover relative pose for calibrated and uncalibrated
    // configurations. In the uncalibrated case, this most likely leads to a
    // ill-defined reconstruction, but sometimes it succeeds anyways after e.g.
    // subsequent bundle-adjustment etc.
    PoseFromEssentialMatrix(E, inlier_points1_normalized,
                            inlier_points2_normalized, &R, &tvec, &points3D);
  } else if (config == PLANAR || config == PANORAMIC ||
             config == PLANAR_OR_PANORAMIC) {
    Eigen::Vector3d n;
    PoseFromHomographyMatrix(
        H, camera1.CalibrationMatrix(), camera2.CalibrationMatrix(),
        inlier_points1_normalized, inlier_points2_normalized, &R, &tvec, &n,
        &points3D);
  } else {
    return false;
  }

  qvec = RotationMatrixToQuaternion(R);

  if (points3D.empty()) {
    tri_angle = 0;
  } else {
    tri_angle = Median(CalculateTriangulationAngles(
        Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D));
  }

  if (config == PLANAR_OR_PANORAMIC) {
    if (tvec.norm() == 0) {
      config = PANORAMIC;
      tri_angle = 0;
    } else {
      config = PLANAR;
    }
  }

  return true;
}

void TwoViewGeometry::EstimateCalibratedGeometry(
    const Camera& camera1, const FeatureKeypoints& keypoints1,
    const Camera& camera2, const FeatureKeypoints& keypoints2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Save the key points in the two-view geometry.
  keypoints_1 = keypoints1;
  keypoints_2 = keypoints2;

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1_normalized(matches.size());
  std::vector<Eigen::Vector2d> matched_points2_normalized(matches.size());

  for (size_t i = 0; i < matches.size(); ++i) {
    const point2D_t idx1 = matches[i].point2D_idx1;
    const point2D_t idx2 = matches[i].point2D_idx2;

    Eigen::Vector2d pt_1(keypoints1[idx1].x, keypoints1[idx1].y);
    Eigen::Vector2d pt_2(keypoints2[idx2].x, keypoints2[idx2].y);

    matched_points1_normalized[i] = camera1.ImageToWorld(pt_1);
    matched_points2_normalized[i] = camera2.ImageToWorld(pt_2);
  }

  // Estimate calibrates model.

  auto E_ransac_options = options.ransac_options;
  E_ransac_options.max_error =
      (camera1.ImageToWorldThreshold(options.ransac_options.max_error) +
       camera2.ImageToWorldThreshold(options.ransac_options.max_error)) /
      2;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      E_ransac(E_ransac_options);
  const auto E_report =
      E_ransac.Estimate(matched_points1_normalized, matched_points2_normalized);
  E = E_report.model;

  CHECK_EQ(camera1.FocalLengthX(), camera2.FocalLengthX());
  CHECK_EQ(camera1.FocalLengthY(), camera2.FocalLengthY());

  // Get coordenates of focal length
  double f_x = camera1.FocalLengthX();
  double f_y = camera1.FocalLengthY();

  CHECK_EQ(camera1.PrincipalPointX(), camera2.PrincipalPointX());
  CHECK_EQ(camera1.PrincipalPointY(), camera2.PrincipalPointY());

  // Get coordenates of principal point
  double c_x = camera1.PrincipalPointX();
  double c_y = camera1.PrincipalPointY();

  // Calculate the matrix K
  Eigen::MatrixXd K(3, 3);
  K << f_x, 0.0f, c_x, 0.0f, f_y, c_y, 0.0f, 0.0f, 1.0f;

  // Calculate a undistortionated F matrix.
  F = K.transpose().inverse() * E * K.inverse();

  // Undistortionate key points in both images
  for (FeatureKeypoint& keypoint : keypoints_1) {
    Eigen::Vector2d pt_1(keypoint.x, keypoint.y);
    // Undistortionated key poits coordinates in first image.
    Eigen::Vector2d und_1 = camera1.ImageToWorld(pt_1);
    keypoint.x = und_1[0] * f_x + c_x;
    keypoint.y = und_1[1] * f_y + c_y;
  }

  for (FeatureKeypoint& keypoint : keypoints_2) {
    Eigen::Vector2d pt_2(keypoint.x, keypoint.y);
    // Undistortionated key poits coordinates in second image.
    Eigen::Vector2d und_2 = camera2.ImageToWorld(pt_2);
    keypoint.x = und_2[0] * f_x + c_x;
    keypoint.y = und_2[1] * f_y + c_y;
  }

  if ((!E_report.success) ||
      (E_report.support.num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  const std::vector<char>* best_inlier_mask = nullptr;
  size_t num_inliers = 0;

  if (E_report.success &&
      E_report.support.num_inliers >= options.min_num_inliers) {
    num_inliers = E_report.support.num_inliers;
    best_inlier_mask = &E_report.inlier_mask;
    config = ConfigurationType::CALIBRATED;

  } else {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  if (best_inlier_mask != nullptr) {
    inlier_matches =
        ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
  }
}

void TwoViewGeometry::EstimateInitCalibratedGeometry(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_points2(matches.size());
  std::vector<Eigen::Vector2d> matched_points1_normalized(matches.size());
  std::vector<Eigen::Vector2d> matched_points2_normalized(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    const point2D_t idx1 = matches[i].point2D_idx1;
    const point2D_t idx2 = matches[i].point2D_idx2;
    matched_points1[i] = points1[idx1];
    matched_points2[i] = points2[idx2];
    matched_points1_normalized[i] = camera1.ImageToWorld(points1[idx1]);
    matched_points2_normalized[i] = camera2.ImageToWorld(points2[idx2]);
  }

  // Estimate calibrates model.

  auto E_ransac_options = options.ransac_options;
  E_ransac_options.max_error =
      (camera1.ImageToWorldThreshold(options.ransac_options.max_error) +
       camera2.ImageToWorldThreshold(options.ransac_options.max_error)) /
      2;

  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      E_ransac(E_ransac_options);
  const auto E_report =
      E_ransac.Estimate(matched_points1_normalized, matched_points2_normalized);
  E = E_report.model;

  if ((!E_report.success) ||
      (E_report.support.num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  const std::vector<char>* best_inlier_mask = nullptr;
  size_t num_inliers = 0;

  if (E_report.success &&
      E_report.support.num_inliers >= options.min_num_inliers) {
    num_inliers = E_report.support.num_inliers;
    best_inlier_mask = &E_report.inlier_mask;
    config = ConfigurationType::CALIBRATED;

  } else {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  if (best_inlier_mask != nullptr) {
    inlier_matches =
        ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);
  }
}

void TwoViewGeometry::EstimateUncalibrated(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_points2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_points1[i] = points1[matches[i].point2D_idx1];
    matched_points2[i] = points2[matches[i].point2D_idx2];
  }

  // Estimate epipolar model.

  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
  F = F_report.model;

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
  H = H_report.model;

  if ((!F_report.success && !H_report.success) ||
      (F_report.support.num_inliers < options.min_num_inliers &&
       H_report.support.num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Determine inlier ratios of different models.

  const double H_F_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      F_report.support.num_inliers;

  if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
    config = ConfigurationType::PLANAR_OR_PANORAMIC;
  } else {
    config = ConfigurationType::UNCALIBRATED;
  }

  inlier_matches = ExtractInlierMatches(matches, F_report.support.num_inliers,
                                        F_report.inlier_mask);

  if (options.detect_watermark &&
      DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                      F_report.support.num_inliers, F_report.inlier_mask,
                      options)) {
    config = ConfigurationType::WATERMARK;
  }
}

bool TwoViewGeometry::DetectWatermark(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const size_t num_inliers, const std::vector<char>& inlier_mask,
    const Options& options) {
  options.Check();

  // Check if inlier points in border region and extract inlier matches.

  const double diagonal1 = std::sqrt(camera1.Width() * camera1.Width() +
                                     camera1.Height() * camera1.Height());
  const double diagonal2 = std::sqrt(camera2.Width() * camera2.Width() +
                                     camera2.Height() * camera2.Height());
  const double minx1 = options.watermark_border_size * diagonal1;
  const double miny1 = minx1;
  const double maxx1 = camera1.Width() - minx1;
  const double maxy1 = camera1.Height() - miny1;
  const double minx2 = options.watermark_border_size * diagonal2;
  const double miny2 = minx2;
  const double maxx2 = camera2.Width() - minx2;
  const double maxy2 = camera2.Height() - miny2;

  std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2(num_inliers);

  size_t num_matches_in_border = 0;

  size_t j = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const auto& point1 = points1[i];
      const auto& point2 = points2[i];

      inlier_points1[j] = point1;
      inlier_points2[j] = point2;
      j += 1;

      if (!IsImagePointInBoundingBox(point1, minx1, maxx1, miny1, maxy1) &&
          !IsImagePointInBoundingBox(point2, minx2, maxx2, miny2, maxy2)) {
        num_matches_in_border += 1;
      }
    }
  }

  const double matches_in_border_ratio =
      static_cast<double>(num_matches_in_border) / num_inliers;

  if (matches_in_border_ratio < options.watermark_min_inlier_ratio) {
    return false;
  }

  // Check if matches follow a translational model.

  RANSACOptions ransac_options = options.ransac_options;
  ransac_options.min_inlier_ratio = options.watermark_min_inlier_ratio;

  LORANSAC<TranslationTransformEstimator<2>, TranslationTransformEstimator<2>>
      ransac(ransac_options);
  const auto report = ransac.Estimate(inlier_points1, inlier_points2);

  const double inlier_ratio =
      static_cast<double>(report.support.num_inliers) / num_inliers;

  return inlier_ratio >= options.watermark_min_inlier_ratio;
}

}  // namespace colmap
