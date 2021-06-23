COLMAP
======

About
-----

This COLMAP version has been modified for work with colonoscopy images. Now COLMAP only estimate
calibrated geometries and discart those that don't have a minimun inliers number. This implicated
estimate the **Esential** matrix using **RANSAC** and the undistortionated key points coordinates. To make 
posible an implementation of **Guided Matching**, the **Fundamental** matrix is calculated using the intrinsic matrix **K**.

**Modified Files:**

* colmap/src/estimators/two_view_geometry.cc
* colmap/src/estimators/two_view_geometry.h
* colmap/src/feature/matching.cc
* colmap/src/feature/sif.cc
* colmap/sfm/incremental_mapper.cc