//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <fusionba\utility.h>
#if USE_OPENCV
#include <opencv2/opencv.hpp>
#include <image_utility/image_utility.h>
#endif


class PointSequenceLoader {
public:
	void loadPointStream(std::string filePath,long long pnum);
	bool getNextPointData(float* dat);
	bool backPointData();
	void writePlyReflectance(string filePath, double* timeList, _6dof* motionData, Matrix4d& calib, int firstFrame, int lastFrame);
	void writePlyReflectance(string filePath, double* timeList, _6dof* motionData, Matrix4d& calib, int firstFrame, int lastFrame, int skip, double refThres=0.001);
	void writePlyReflectance(string filePath, string filePathPreview, double* timeList, _6dof* motionData, Matrix4d& calib, int firstFrame, int lastFrame, int skip);
	void getPCReflectance(double* timeList, _6dof* motionData, Matrix4d& calib, int firstFrame, int lastFrame, int skip, vector<float>& v, vector<float>& rf);
	void setTimeOffset(double ts_) { ts = ts_; };
#if USE_OPENCV
	void getPCColor(double* timeList, _6dof* motionData, Matrix4d& calib, int firstFrame, int lastFrame, vector<string>imgFileNameList, vector<float>& v, vector<uchar>& rgba, int skip = 0);
#endif
	__int64 seekByTime(double t);
	double getTimeOffset() { return ts; }

private:
	bool seekByPointIdx(__int64 idx);
	std::ifstream* ifs=NULL;
	__int64 pointNum;
	int fcnt;
	std::string baseFilePath;
	bool bStream;
	double ts=0;

};