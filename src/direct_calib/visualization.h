//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once
#include <opencv2/opencv.hpp>
#include <camera.h>
#include <utility.h>

void colorMapping(std::vector<Eigen::Vector4d> pc, std::vector<cv::Mat> imgs, std::vector<cv::Mat> masks, std::vector<int> usedImageIdx,
	std::vector<Eigen::Matrix4d> tf_cams, std::vector<uchar>& color, std::vector<double> timestamps, std::vector<double> imtimestamp, camera*cam, double thres, unsigned char* posibility);
void imageProjection(std::string imageFileName, cv::Mat projImage, std::vector<Eigen::Vector4d> points, std::vector<uchar> color, Eigen::Matrix4d camera_pose,
	int stPtIdx, int endPtIdx, camera* cam);
void getColorSubPixel_diff_mask(cv::Mat image, cv::Mat mask, cv::Point2d p, double *ret, double * dret, double& weight);
void getColorSubPixel_mask(cv::Mat image, cv::Mat mask, cv::Point2d p, double *ret, double& weight);
void getGraySubPixel(cv::Mat image, cv::Point2f p, double* ret);
void getGraySubPixel(cv::Mat image, cv::Mat mask, cv::Point2f p, double *ret, double& weight);
void edgeDetection(cv::Mat in, cv::Mat& out);