//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once

#include <dlib/optimization.h>
#include <iostream>
#include <fstream>
#include <camera_simulator.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <utility.h>
#include <camera.h>
#include <direct_calib/visualization.h>
#include <direct_calib/mesh_gen.h>
#define DMIN 1.25
#define DMAX 1.50

typedef dlib::matrix<double, 0, 1> column_vector;

class movement {
public:
	movement() {};
	void loadMovementMetaShape(std::string movement);

private:


};


class fusionscan {
public:
	fusionscan() {};
	void setscan(std::string scaninfo);
	void setInitialMotion(std::string motionFile);
	void visualize(const column_vector& m, std::string base, int outNum);
	void setImage(int usedImageNum, double c2f=1.0);
	void setPoints(int downsample);
	void setCamera(nlohmann::json baseConf, double c2f = 1.0);
	void setInitCalib(nlohmann::json baseConf);

	void setSpecular(cv::Mat img, cv::Mat& outimg);
	void setMask(nlohmann::json baseConf, double c2f = 1.0);
	void setEdge(nlohmann::json baseConf);

	Eigen::Matrix4d timeMotion(double ts);
	void visiblePosibilityCalc(std::string plypath);

	double costComputation(const column_vector& m);
	void outputCalib(const column_vector& m,std::string outjson);
	double varianceCalc(std::vector<Eigen::Vector4d> pointcloud);
	double edgenessCalc(std::vector<Eigen::Vector4d> pointcloud, std::vector<float> ts);
	double onePointsColorAndVarianceCalc(Eigen::Vector4d pt, long long ptidx, long long pc_size,double* color, double* variance);
	double onePointsEdgenessCalc(Eigen::Vector4d pt,float ts, long long ptidx, long long pc_size, double* color);

	void colorMapping(std::vector<Eigen::Vector4d> pc, std::vector<uchar>& color, std::vector<double> timestamps);

	void deformation(Eigen::Vector4d q, Eigen::Vector3d t, double scale);

	void optimization_test();
	void pointColorWeightComputation();
	void setTimeThreshold(double tThresh) {
		timeThresh = tThresh;
	}

	void distModeCheck() {
		int trans[] = { 9, 10, 11 };
		bool intrans = false;
		for (int j = 0; j < 3; j++) {
			if (inParam(trans[j])) {
				intrans = true;
			};
		}
		if (!intrans) {
			distmode = 1;
			std::cout << "distance mode: normal" << std::endl;
			return;
		}
		
		bool inrot = false;
		int rot[] = { 0,1,2,3,4,5,6,7,8 };
		for (int j = 0; j < 6; j++) {
			if (inParam(rot[j])) {
				inrot = true;
			};
		}
		if (!inrot) {
			distmode = -1;
			std::cout << "distance mode: trans" << std::endl;
			return;
		}
		distmode = 0;
	}
	void directModeSet(int mode) {
		distmode = mode;
	}

	std::vector<int>optvalue;
	column_vector m_;
private:
	long long findIdx(float qt);
	long long edgeFindIdx(float qt);
	bool inParam(int i) {
		for (auto itr : optvalue) {
			if (itr == i)return true;
		}
		return false;
	}
	
	//motion deformation
	void calcColorAndWeight();

	//threshold 0.1,0.1
	double depthRateCompute(double dm, double dp) {
		
		double dth = 0.1 + dm;
		if (dth > dp) {
			return 1;
		}
		else if (dth + 0.1 >dp) {
			return 1.0 - (dp-dth)/0.1;
		
		}
		return 0;
	}
	
	double lossFunction(double in) {
		//in: coldiff_2
		//small change: week
		//large change: strong

		double out = 0;
		double thresh = 2 / 255.0;
		if (in < thresh) {
			out = in/128.0;
		}
		else {
			out = (in - thresh * (127.0 / 128.0));
		}

		return out;
	}

	float* allpt;
	cv::Mat scans;
	long long ptnum;

	camera* cam;
	PanoramaRenderer* pr;
	unsigned char* posibility=NULL;

	double sync_offset;
	std::vector<std::string> filelist;
	std::vector<double> image_timestamp;
	int startFrame, endFrame;

	std::vector<double> cammot_timestamp;
	std::vector<Eigen::Matrix4d> cameramotion;
	std::vector<double> motionTranslationWeight, motionRotationWeight;
	std::vector<double> ptweight;
	std::vector<double> ptdist;

	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> mask;
	std::vector<int> usedIdx;
	std::vector<long long> usedPtIdx;
	std::vector<Eigen::Matrix4d> deformedMotion;

	bool bEdge = false;
	float* allpt_edge;
	long long ptnum_edge;
	std::vector<long long> usedEdgePtIdx;
	std::vector<cv::Mat> edgeMat;

	int distmode = 0;
	double timeThresh = 5.0;

	Eigen::Matrix3d extR;
	Eigen::Vector3d extt;
};
