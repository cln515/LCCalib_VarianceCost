//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "fusionba/BasicPly.h"
#include "fusionba/utility.h"

using namespace Eigen;

#define SIDELENGTH 20000
#define THRESHOLD_LENGTH 0.01
#ifndef IMAGEMYUTILITY
#define IMAGEMYUTILITY
void getGraySubPixel(cv::Mat image,cv::Point2f p,double* ret);
void getGraySubPixel_uchar(cv::Mat image,cv::Point2f p,double* ret);
void getGraySubPixel_float(cv::Mat image,cv::Point2f p,double* ret);
void getSubPixel_float(cv::Mat image,cv::Point2f p,double* ret);
void getGraySubPixel(cv::Mat image,cv::Point2f p,double* ret,double* dret);
void getColorSubPixel(cv::Mat image, cv::Point2d p, double *ret);

void panoramaRectification(cv::Mat image1, cv::Mat image2, cv::Mat& dstimage1, cv::Mat& dstimage2, Vector3d epi, Matrix3d R, Matrix3d& R0, Matrix3d& R1);
void panoramaRectification(cv::Mat image1, cv::Mat& dstimage1, Vector3d epi, Matrix3d R, Matrix3d& R0);
void goodFeatureToTrack_onProjection(cv::Mat image, vector<cv::Point2f> proj_points, vector<int>& selectedIdx, double minDistance, int maxCorners, double cornerThres=0.0);
void fisheye2panorama(cv::Mat inputArray, cv::Mat& outputArray, cv::Mat cameraParameter, cv::Mat distortionCoeff);

void meshGenerater(Vector3d center, Matrix3d rot, float vertAngle, BasicPly& plyData, double threshold = THRESHOLD_LENGTH);
#endif
