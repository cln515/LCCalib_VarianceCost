//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once
#include "ceres/ceres.h"
#include "fusionba/SensorFusion.h"
#include "fusionba/BasicPly.h"
#include "fusionba/image_utility.h"
#define FN_VAL 40

class SensorFusionBA : public SensorFusion {
public:
	vector<_6dof> framePos;
	vector<float> vertices;
	vector<float> refs;
	vector<double> vTimestamps;
	vector<double> edgeness;
	vector<vector<int>> septAreas;
	vector<cv::Mat>baseImages;
	vector<cv::Mat>edgeImages;
	vector<int> plottedFrames;
	vector<Matrix3d> transMats;
	vector<Matrix4d> changedMats;

	int fitv = 5;
	double dcpara = 15.0;
	bool bfixTrans = false;

	bool LoadMotion(string motionFile);
	void WriteMotion(string motionFile);
	void WriteMotionAscii(string motionFile);
	void BatchBA(int baseFrame, int frameNum,int skipNum);
	void BatchBA_scale(int baseFrame, int frameNum, int skipNum);
	void BatchBA_TL(int baseFrame, int frameNum, int skipNum);
	void BatchBA_Global(int baseFrame, int frameNum, int skipNum);
	string BatchBA_Global_seq(int baseFrame, int frameNum, int skipNum, int overlap);
	string BatchBA_Global_seq(vector<int> sampledFrame, int overlap);
	string BatchBA_Global_seq2(vector<int> sampledFrame, int overlap);
	void BatchBA_R(int baseFrame, int frameNum, int skipNum);
};

void pointMatching(vector<cv::Mat> images);
vector<map<int, cv::Point2f>> pointTracking(vector<cv::Mat> images, double** mots,int skipNum=1,cv::Mat mask=cv::Mat(),double cPara=15.0,int fItv=2);


struct BaDepthUnknownCostFunction {
public:
	BaDepthUnknownCostFunction(Vector3d& xibase, Vector3d& bv1_)
	{

		xim1 = xibase;
		xi = bv1_;

	};
	bool operator()(const double* parameters, double* residual) const {
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];
		double x = parameters[3];
		double y = parameters[4];
		double z = parameters[5];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T;T << x, y, z;
		T = T.normalized();
		Vector3d vec = R.transpose() * xim1;
		T = R.transpose() * T;
		//double xbar = xi(0);
		//double ybar = xi(1);

		//vec << -ybar * T(2) + T(1), xbar * T(2) - T(0), -xbar * T(1) + ybar*T(0);

		Vector3d vert = vec.cross(T).normalized();

		residual[0] = vert.dot(xi);
		return true;
	}
private:
	Vector3d xi;
	Vector3d xim1;
};

struct BaFeatureProjCostFunction {
public:
	BaFeatureProjCostFunction(Vector3d& xibase, double* w_)
	{
		w = w_;

		xim1 = xibase;

	};
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];
		double x = parameters[3];
		double y = parameters[4];
		double z = parameters[5];

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T;T << x, y, z;
		Vector3d P;P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d xim1;
	double* w;
};



struct depthKnownCostFunc {
public:
	depthKnownCostFunc(Vector3d& p3d_, Vector3d& eyeDirec_)
	{
		p = p3d_;
		v = eyeDirec_;
	};
	bool operator()(const double* parameters, double* residual) const {
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];
		double x = parameters[3];
		double y = parameters[4];
		double z = parameters[5];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T;T << x, y, z;
	
		//double xbar = eyeDirec(0);
		//double ybar = eyeDirec(1);

		//double err1 = ((R.block(0, 0, 1, 3) - xbar * R.block(2, 0, 1, 3))*p3d)(0) + T(0) - xbar*T(2);
		//double err2 = ((R.block(1, 0, 1, 3) - ybar * R.block(2, 0, 1, 3))*p3d)(0) + T(1) - ybar * T(2);
		Vector3d plot = R.transpose() * (p - T);
		Vector3d err = v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d p;
	Vector3d v;
};

struct BADepthKnownCostFunc {
public:
	BADepthKnownCostFunc(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
	};
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		double rx = parameters[0];//local 2 world
		double ry = parameters[1];
		double rz = parameters[2];
		double x = parameters[3];
		double y = parameters[4];
		double z = parameters[5];

		double rx2 = parameters2[0];//world 2 local
		double ry2 = parameters2[1];
		double rz2 = parameters2[2];
		double x2 = parameters2[3];
		double y2 = parameters2[4];
		double z2 = parameters2[5];

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;

		//double xbar = eyeDirec(0);
		//double ybar = eyeDirec(1);

		//double err1 = ((R.block(0, 0, 1, 3) - xbar * R.block(2, 0, 1, 3))*p3d)(0) + T(0) - xbar*T(2);
		//double err2 = ((R.block(1, 0, 1, 3) - ybar * R.block(2, 0, 1, 3))*p3d)(0) + T(1) - ybar * T(2);
		
		Vector3d p_ = R * p + T;
		Vector3d plot = R2.transpose() * (p_ - T2);
		
		
		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d p;
	Vector3d v;
	double* w;
};

struct BADepthKnownCostFunc_anckor {
public:
	BADepthKnownCostFunc_anckor(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
	};
	bool operator()(const double* parameters2, double* residual) const {
		//double rx = parameters[0];//local 2 world
		//double ry = parameters[1];
		//double rz = parameters[2];
		//double x = parameters[3];
		//double y = parameters[4];
		//double z = parameters[5];//first frame

		double rx2 = parameters2[0];//world 2 local
		double ry2 = parameters2[1];
		double rz2 = parameters2[2];
		double x2 = parameters2[3];
		double y2 = parameters2[4];
		double z2 = parameters2[5];

		Matrix3d R = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x2, y2, z2;
		
		Vector3d plot = R.transpose() * (p - T);


		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d p;
	Vector3d v;
	double* w;
};

struct BADepthKnownCostFunc_anckor_rev {
public:
	BADepthKnownCostFunc_anckor_rev(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
	};
	bool operator()(const double* parameters2, double* residual) const {
		//double rx = parameters[0];//local 2 world
		//double ry = parameters[1];
		//double rz = parameters[2];
		//double x = parameters[3];
		//double y = parameters[4];
		//double z = parameters[5];//first frame

		double rx2 = parameters2[0];//world 2 local
		double ry2 = parameters2[1];
		double rz2 = parameters2[2];
		double x2 = parameters2[3];
		double y2 = parameters2[4];
		double z2 = parameters2[5];

		Matrix3d R = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x2, y2, z2;

		Vector3d plot = R * p + T;


		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d p;
	Vector3d v;
	double* w;
};








struct BADepthKnownCostFunc_direction {
public:
	BADepthKnownCostFunc_direction(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot1_, double* rot2_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot1 = rot1_;
		paramRot2 = rot2_;
	};
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		double rx = paramRot1[0];//local 2 world
		double ry = paramRot1[1];
		double rz = paramRot1[2];
		double x = parameters[0];
		double y = parameters[1];
		double z = parameters[2];

		double rx2 = paramRot2[0];//world 2 local
		double ry2 = paramRot2[1];
		double rz2 = paramRot2[2];
		double x2 = parameters2[0];
		double y2 = parameters2[1];
		double z2 = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;
		//double xbar = eyeDirec(0);
		//double ybar = eyeDirec(1);

		//double err1 = ((R.block(0, 0, 1, 3) - xbar * R.block(2, 0, 1, 3))*p3d)(0) + T(0) - xbar*T(2);
		//double err2 = ((R.block(1, 0, 1, 3) - ybar * R.block(2, 0, 1, 3))*p3d)(0) + T(1) - ybar * T(2);
		Vector3d vjd = R * v;
		Vector3d p1, p2, v1, v2;
		p1 = T;
		p2 = p_;
		v1 = (T2 - T).normalized();
		v2 = -vjd;

		double s, t, v1v2;
		v1v2 = v1.dot(v2);
		s = (p2 - p1).dot(v1 - v1v2 * v2) / (1 - v1v2 * v1v2);
		t = -(p2 - p1).dot(v2 - v1v2 * v1) / (1 - v1v2 * v1v2);
		residual[0] = (-s * v1 + (p2 - p1) + t * v2).norm();
		//residual[0] = residual[0];
		return true;
	}
private:
	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot1;
	double* paramRot2;
};




struct BADepthKnownCostFunc_PrjT {
public:
	BADepthKnownCostFunc_PrjT(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot1_, double* rot2_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot1 = rot1_;
		paramRot2 = rot2_;
	};


	void set(double* paramt1,double *paramt2) {
		trans1=paramt1;
		trans2 = paramt2;
	};

	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}
		double rx = paramRot1[0];//local 2 world
		double ry = paramRot1[1];
		double rz = paramRot1[2];
		double x = parameters[0];
		double y = parameters[1];
		double z = parameters[2];

		double rx2 = paramRot2[0];//world 2 local
		double ry2 = paramRot2[1];
		double rz2 = paramRot2[2];
		double x2 = parameters2[0];
		double y2 = parameters2[1];
		double z2 = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(trans1,trans2, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}

	double squaredCostEveluate() {
		double resid[3];
		operator()(trans1, trans2, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		return err;
	}
	bool bInlier() {
		return enable;
	}
	Vector3d p;
	Vector3d v;


	void debug() {
		double rx = paramRot1[0];//local 2 world
		double ry = paramRot1[1];
		double rz = paramRot1[2];
		double x = trans1[0];
		double y = trans1[1];
		double z = trans1[2];

		double rx2 = paramRot2[0];//world 2 local
		double ry2 = paramRot2[1];
		double rz2 = paramRot2[2];
		double x2 = trans2[0];
		double y2 = trans2[1];
		double z2 = trans2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T; T << x, y, z;
		Vector3d T2; T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;
		Vector3d plot = R2.transpose() * (p_ - T2);
		Vector3d err = w[0] * v.cross(plot) / plot.norm();

		std::cout << "plot,p,T,T2" << std::endl;
		std::cout << plot.transpose() << std::endl;
		std::cout << plot.normalized().transpose() << std::endl;
		std::cout << v.transpose() << std::endl;
		std::cout << p.transpose() << std::endl;
		std::cout << T.transpose() << std::endl;
		std::cout << T2.transpose() << std::endl;
		std::cout << err.norm() << std::endl;
	};

private:
	bool enable;
	double* trans1;
	double* trans2;

	double* w;
	double* paramRot1;
	double* paramRot2;
};


struct BADepthKnownCostFunc_direction_anckor {
public:
	BADepthKnownCostFunc_direction_anckor(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
	};
	bool operator()(const double* parameters2, double* residual) const {
		//double rx = parameters[0];//local 2 world
		//double ry = parameters[1];
		//double rz = parameters[2];
		//double x = parameters[3];
		//double y = parameters[4];
		//double z = parameters[5];//first frame

		double rx2 = paramRot[0];//world 2 local
		double ry2 = paramRot[1];
		double rz2 = paramRot[2];
		double x2 = parameters2[0];
		double y2 = parameters2[1];
		double z2 = parameters2[2];

		Matrix3d R = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x2, y2, z2;
		T = T.normalized();
		//Vector3d plot = R.transpose() * (p - T);


		Vector3d vjd = R * v;

		Vector3d c;c << 0, 0, 0;

		Vector3d p1, p2, v1, v2;
		p1=c;
		p2=p;
		v1=T;
		v2=-vjd;

		double s, t, v1v2;
		v1v2 = v1.dot(v2);
		s = (p2 - p1).dot(v1 - v1v2 * v2) / (1 - v1v2 * v1v2);
		t = -(p2 - p1).dot(v2 - v1v2 * v1)/ (1 - v1v2 * v1v2);
		residual[0] = (-s * v1 + (p2 - p1) + t * v2).norm();

		/*double Val = T.dot(vjd);
		double s = (p - c).dot(T - Val * vjd) / (1 - Val * Val);
		double alpha = (p - c).dot(vjd - Val * T) / (1 - Val * Val);*/

		//residual[0] = (((p - c) - alpha * vjd) - s * T).norm();

		return true;
	}
private:
	Vector3d p;
	Vector3d v;
	Vector3d c;
	double* w;	double* paramRot;
};


struct BADepthKnownCostFunc_PrjT_anckor {
public:
	BADepthKnownCostFunc_PrjT_anckor(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
		anchorm[0] = anchorm[1] = anchorm[2] = anchorm[3] = anchorm[4] = anchorm[5] = 0;
	};
	BADepthKnownCostFunc_PrjT_anckor(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_,double* anchormotion)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
		anchorm[0] = anchormotion[0];
		anchorm[1] = anchormotion[1];
		anchorm[2] = anchormotion[2];
		anchorm[3] = anchormotion[3];
		anchorm[4] = anchormotion[4];
		anchorm[5] = anchormotion[5];
	};

	void set(double* paramt) {
		params = paramt;
	};

	bool operator()(const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}
		double rx = anchorm[0];
		double ry = anchorm[1];
		double rz = anchorm[2];
		double x = anchorm[3];
		double y = anchorm[4];
		double z = anchorm[5];

		double rx2 = paramRot[0];//world 2 local
		double ry2 = paramRot[1];
		double rz2 = paramRot[2];
		double x2 = parameters2[0];
		double y2 = parameters2[1];
		double z2 = parameters2[2];


		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
		/*double Val = T.dot(vjd);
		double s = (p - c).dot(T - Val * vjd) / (1 - Val * Val);
		double alpha = (p - c).dot(vjd - Val * T) / (1 - Val * Val);*/

		//residual[0] = (((p - c) - alpha * vjd) - s * T).norm();

		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(params, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}

	double squaredCostEveluate() {
		double resid[3];
		operator()(params, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		return err;
	}
	bool bInlier() {
		return enable;
	}

private:
	bool enable = true;
	double* params;
	double anchorm[6];
	Vector3d p;
	Vector3d v;
	Vector3d c;
	double* w;	double* paramRot;
};







struct BADepthKnownCostFunc_direction_anckor_rev {
public:
	BADepthKnownCostFunc_direction_anckor_rev(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
	};
	bool operator()(const double* parameters2, double* residual) const {
		//double rx = parameters[0];//local 2 world
		//double ry = parameters[1];
		//double rz = parameters[2];
		//double x = parameters[3];
		//double y = parameters[4];
		//double z = parameters[5];//first frame

		double rx2 = paramRot[0];//world 2 local
		double ry2 = paramRot[1];
		double rz2 = paramRot[2];
		double x2 = parameters2[0];
		double y2 = parameters2[1];
		double z2 = parameters2[2];

		Matrix3d R = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x2, y2, z2;

		Vector3d p_ = R * p + T;

		Vector3d c;c << 0, 0, 0;

		//double Val = -T.dot(v);
		//double s = (p_ - c).dot(-T - Val * v) / (1 - Val * Val);
		//double alpha = (p_ - c).dot(v - Val * -T) / (1 - Val * Val);

		Vector3d p1, p2, v1, v2;
		p1 = T;
		p2 = p_;
		v1 =( c-T).normalized();
		v2 = -v;

		double s, t, v1v2;
		v1v2 = v1.dot(v2);
		s = (p2 - p1).dot(v1 - v1v2 * v2) / (1 - v1v2 * v1v2);
		t = -(p2 - p1).dot(v2 - v1v2 * v1) / (1 - v1v2 * v1v2);
		residual[0] = (-s * v1 + (p2 - p1) + t * v2).norm();


		return true;
	}
private:

	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot;
};

struct BADepthKnownCostFunc_PrjT_anckor_rev {
public:
	BADepthKnownCostFunc_PrjT_anckor_rev(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
		anchorm[0] = anchorm[1] = anchorm[2] = anchorm[3] = anchorm[4] = anchorm[5] = 0;
	};
	BADepthKnownCostFunc_PrjT_anckor_rev(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_,double* anchormotion)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
		anchorm[0] = anchormotion[0];
		anchorm[1] = anchormotion[1];
		anchorm[2] = anchormotion[2];
		anchorm[3] = anchormotion[3];
		anchorm[4] = anchormotion[4];
		anchorm[5] = anchormotion[5];
	};

	void set(double* paramt) {
		params = paramt;
	};

	bool operator()(const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2]=0;
			return true;
		}
		
		double rx = anchorm[0];
		double ry = anchorm[1];
		double rz = anchorm[2];
		double x = anchorm[3];
		double y = anchorm[4];
		double z = anchorm[5];

		double rx2 = paramRot[0];//world 2 local
		double ry2 = paramRot[1];
		double rz2 = paramRot[2];
		double x2 = parameters2[0];
		double y2 = parameters2[1];
		double z2 = parameters2[2];

		//R,xyz2: local 2 world
		//R2,xyz: world 2 local
		Matrix3d R = axisRot2R(rx2, ry2, rz2);
		Matrix3d R2 = axisRot2R(rx, ry, rz);

		Vector3d T;T << x2, y2, z2;
		Vector3d T2;T2 << x, y, z;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(params, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}

	double squaredCostEveluate() {
		double resid[3];
		operator()(params, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		return err;
	}
	bool bInlier() {
		return enable;
	}


private:
	bool enable = true;
	double* params;

	double anchorm[6];
	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot;
};

struct BADepthKnownCostFunc_scale{
public:
	BADepthKnownCostFunc_scale(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, int* idxs,double**paramRots_,double** paramTranses_, bool rev_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		idx1 = idxs[0];
		idx2 = idxs[1];
		rev = rev_;
		paramRots = paramRots_;
		paramTranses = paramTranses_;
	};
	bool operator()(const double* parameters2, double* residual) const {
		//double rx = parameters[0];//local 2 world
		//double ry = parameters[1];
		//double rz = parameters[2];
		//double x = parameters[3];
		//double y = parameters[4];
		//double z = parameters[5];//first frame

		double sc = parameters2[0];
		Matrix3d R =Matrix3d::Identity();
		Vector3d T;T << 0, 0, 0;
		
		
		_6dof src,base1,base2,targ;
		Matrix4d trf=Matrix4d::Identity();
		if (rev) {
			for (int i = idx1;i > idx2;i--) {
				Matrix4d m1, m2;
				if (i > 1)base1 = { paramRots[i-2][0], paramRots[i - 2][1], paramRots[i - 2][2] ,paramTranses[i-2][0], paramTranses[i-2][1], paramTranses[i-2][2] };
				else base1 = { 0,0,0,0,0,0 };
				src = { paramRots[i-1][0], paramRots[i - 1][1], paramRots[i - 1][2] ,paramTranses[i - 1][0], paramTranses[i - 1][1], paramTranses[i - 1][2] };
				m1 = _6dof2m(src);
				m2 = _6dof2m(base1);
				m1 = m2.inverse()*m1;m1.block(0, 3, 3, 1) *= sc;
				trf = m1*trf;
			}		
		}
		else {
			for (int i = idx1;i < idx2;i++) {
				Matrix4d m1, m2;
				if (i > 0)src = { paramRots[i - 1][0], paramRots[i - 1][1], paramRots[i - 1][2] ,paramTranses[i - 1][0], paramTranses[i - 1][1], paramTranses[i - 1][2] };
				else src = { 0,0,0,0,0,0 };
				base1 = { paramRots[i][0], paramRots[i][1], paramRots[i][2] ,paramTranses[i][0], paramTranses[i][1], paramTranses[i][2] };
				m1 = _6dof2m(src);
				m2 = _6dof2m(base1);
				m1 = m2.inverse()*m1;m1.block(0, 3, 3, 1) *= sc;
				trf = m1 * trf;
			}
		}

		Vector3d plot = (trf.block(0,0,3,3) * p + trf.block(0, 3, 3, 1));
		Vector3d err = v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);

		return true;
	}
private:
	Vector3d p;
	Vector3d v;

	int idx1, idx2;
	bool rev;
	_6dof m_base;
	double* w;
	double** paramRots;
	double** paramTranses;
};
//
//struct BADepthKnownCostFunc_direction_diff {
//public:
//	BADepthKnownCostFunc_direction_diff(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, int* idxs, double**paramRots_, double** paramTranses_, bool rev_)
//	{
//		w = w_;
//		p = p3d_;
//		v = eyeDirec_;
//		idx1 = idxs[0];
//		idx2 = idxs[1];
//		rev = rev_;
//		paramRots = paramRots_;
//		paramScales = paramScales_;
//	};
//	bool operator()(const double* parameters2, double* residual) const {
//		//double rx = parameters[0];//local 2 world
//		//double ry = parameters[1];
//		//double rz = parameters[2];
//		//double x = parameters[3];
//		//double y = parameters[4];
//		//double z = parameters[5];//first frame
//
//		//double sc = parameters2[0];
//		Matrix3d R = Matrix3d::Identity();
//		Vector3d T;T << 0, 0, 0;
//
//
//		_6dof src, base1, base2, targ;
//		Matrix4d trf = Matrix4d::Identity();
//		if (rev) {
//			for (int i = idx1;i > idx2;i--) {
//				Matrix4d m1, m2;
//				if (i > 1)base1 = { paramRots[i - 2][0], paramRots[i - 2][1], paramRots[i - 2][2] ,paramTranses[i - 2][0], paramTranses[i - 2][1], paramTranses[i - 2][2] };
//				else base1 = { 0,0,0,0,0,0 };
//				src = { paramRots[i - 1][0], paramRots[i - 1][1], paramRots[i - 1][2] ,paramTranses[i - 1][0], paramTranses[i - 1][1], paramTranses[i - 1][2] };
//				m1 = _6dof2m(src);
//				m2 = _6dof2m(base1);
//				m1 = m2.inverse()*m1;m1.block(0, 3, 3, 1) *= parameters2[i - 1];
//				trf = m1 * trf;
//			}
//		}
//		else {
//			for (int i = idx1;i < idx2;i++) {
//				Matrix4d m1, m2;
//				if (i > 0)src = { paramRots[i - 1][0], paramRots[i - 1][1], paramRots[i - 1][2] ,paramTranses[i - 1][0], paramTranses[i - 1][1], paramTranses[i - 1][2] };
//				else src = { 0,0,0,0,0,0 };
//				base1 = { paramRots[i][0], paramRots[i][1], paramRots[i][2] ,paramTranses[i][0], paramTranses[i][1], paramTranses[i][2] };
//				m1 = _6dof2m(src);
//				m2 = _6dof2m(base1);
//				m1 = m2.inverse()*m1;m1.block(0, 3, 3, 1) *= parameters2[i];
//				trf = m1 * trf;
//			}
//		}
//
//		Vector3d plot = (trf.block(0, 0, 3, 3) * p + trf.block(0, 3, 3, 1));
//		Vector3d err = v.cross(plot) / plot.norm();
//		residual[0] = err(0);
//		residual[1] = err(1);
//		residual[2] = err(2);
//
//		return true;
//	}
//private:
//	Vector3d p;
//	Vector3d v;
//
//	int idx1, idx2;
//	bool rev;
//	_6dof m_base;
//	double* w;
//	double** paramRots;
//	double* paramScales;
//};

struct BaFeatureProjCostFunction_ {
public:
	BaFeatureProjCostFunction_(Vector3d& xibase, double* w_)
	{
		w = w_;

		xim1 = xibase;

	};
	void set(double* paramRot, double* paramTrans, double* paramP) {
		rot = paramRot;
		trans = paramTrans;
		point = paramP;
		locw = sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
	}
	bool operator()(const double* paramRot, const double* paramTrans, const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}
		double rx = paramRot[0];
		double ry = paramRot[1];
		double rz = paramRot[2];
		double x = paramTrans[0];
		double y = paramTrans[1];
		double z = paramTrans[2];
		if (bFixTrans) {
			x = trans[0];
			y = trans[1];
			z = trans[2];
		}

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T; T << x, y, z;
		Vector3d P; P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = locw * w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(rot, trans, point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		err = err / (locw*locw);
		if (err != err) {
			std::cout << "nan:" << point[0] << std::endl;
			enable = false;
		}
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
	double squaredCostEveluate() {
		double resid[3];
		operator()(rot, trans, point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		err = err / (locw*locw);
		return err;
	}
	void dbg() {
		double resid[3];
		operator()(rot, trans, point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		std::cout << err << "," << point[0] << "," << point[1] << "," << point[2] << std::endl;
		err = err / (locw*locw);
	}
	bool bInlier() {
		return enable;}
	void fixTrans(bool bFixed) {
		bFixTrans= bFixed;
	}
private:
	bool enable = true;
	bool bFixTrans=false;

	Vector3d xim1;
	double* w;
	double locw = 0;//distance weight
	double* rot;
	double* trans;
	double* point;
};

struct BaFeatureProjCostFunction_fbend {
public:
	BaFeatureProjCostFunction_fbend(Vector3d& xibase, double* w_,_6dof mot,double w1_,double w2_)
	{
		w = w_;
		Matrix4d m = _6dof2m(mot);
		oR = m.block(0, 0, 3, 3);
		ot = m.block(0, 3, 3, 1);
		xim1 = xibase;
		w1 = w1_;
		w2 = w2_;
	};
	void set(double* parambase, double* paramend, double* paramP) {
		bparam1 = parambase;
		bparam2 = paramend;
		point = paramP;

	}
	bool operator()( const double* parambase, const double* paramend, const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}

		//double qx1 = parambase[0];
		//double qy1 = parambase[1];
		//double qz1 = parambase[2];
		//double qw1 = 1 - qx1 * qx1 - qy1 * qy1 - qz1 * qz1;qw1 = sqrt(qw1);
		Vector4d q1; q1<<parambase[0], parambase[1], parambase[2],1;
		q1 = q1.normalized();
		Vector4d q2; q2 << paramend[0], paramend[1], paramend[2], 1;
		q2 = q2.normalized();
		Vector4d qmix = w1 * q1 + w2 * q2;
		qmix = qmix.normalized();
		Matrix3d dR = q2dcm(qmix);

		Vector3d t1, t2;
		t1 << parambase[3], parambase[4], parambase[5];
		t2 << paramend[3], paramend[4], paramend[5];
		Vector3d dt = w1 * t1 + w2 * t2;

		//double rx = rot[0];
		//double ry = rot[1];
		//double rz = rot[2];
		//double x = paramTrans[0];
		//double y = paramTrans[1];
		//double z = paramTrans[2];

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = dR*oR;

		Vector3d T = dt+ot;
		Vector3d P;P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()( bparam1, bparam2,point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}

private:
	bool enable = true;

	Matrix3d oR;
	Vector3d ot;
	double w1, w2;

	Vector3d xim1;
	double* w;

	double* bparam1;
	double* bparam2;
	double* point;
};

struct BaFeatureProjCostFunction_fbend_anchor {
public:
	BaFeatureProjCostFunction_fbend_anchor(Vector3d& xibase, double* w_, _6dof mot, double w1_, double w2_)
	{
		w = w_;
		Matrix4d m = _6dof2m(mot);
		oR = m.block(0, 0, 3, 3);
		ot = m.block(0, 3, 3, 1);
		xim1 = xibase;
		w1 = w1_;
		w2 = w2_;
	};
	void set( double* paramend, double* paramP) {
		//bparam1 = parambase;
		bparam2 = paramend;
		point = paramP;

	}
	bool operator()( const double* paramend, const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}

		//double qx1 = parambase[0];
		//double qy1 = parambase[1];
		//double qz1 = parambase[2];
		//double qw1 = 1 - qx1 * qx1 - qy1 * qy1 - qz1 * qz1;qw1 = sqrt(qw1);
		Vector4d q1; q1 << 0, 0,0, 1;
		q1 = q1.normalized();
		Vector4d q2; q2 << paramend[0], paramend[1], paramend[2], 1;
		q2 = q2.normalized();
		Vector4d qmix = w1 * q1 + w2 * q2;
		qmix = qmix.normalized();
		Matrix3d dR = q2dcm(qmix);

		Vector3d t1, t2;
		t1 <<0,0,0;
		t2 << paramend[3], paramend[4], paramend[5];
		Vector3d dt = w1 * t1 + w2 * t2;

		//double rx = rot[0];
		//double ry = rot[1];
		//double rz = rot[2];
		//double x = paramTrans[0];
		//double y = paramTrans[1];
		//double z = paramTrans[2];

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = dR * oR;

		Vector3d T = dt + ot;
		Vector3d P;P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(bparam2, point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
private:
	bool enable = true;

	Matrix3d oR;
	Vector3d ot;
	double w1, w2;

	Vector3d xim1;
	double* w;

	double* bparam1;
	double* bparam2;
	double* point;
};


struct BaFeatureProjCostFunction_Ancker {
public:
	BaFeatureProjCostFunction_Ancker(Vector3d& xibase, double* w_)
	{
		w = w_;
		xim1 = xibase;
		anchorm[0] = anchorm[1] = anchorm[2] = anchorm[3] = anchorm[4] = anchorm[5] = 0;
	};
	BaFeatureProjCostFunction_Ancker(Vector3d& xibase, double* w_, double* anchormotion)
	{
		w = w_;
		xim1 = xibase;
		anchorm[0] = anchormotion[0];
		anchorm[1] = anchormotion[1];
		anchorm[2] = anchormotion[2];
		anchorm[3] = anchormotion[3];
		anchorm[4] = anchormotion[4];
		anchorm[5] = anchormotion[5];
	};
	void set(double* point_) {
		point = point_;
		locw = sqrt(point[0] * point[0]+ point[1] * point[1] + point[2] * point[2]);
	}


	bool operator()(const double* parameters2, double* residual) const {
		double rx = anchorm[0];
		double ry = anchorm[1];
		double rz = anchorm[2];
		double x = anchorm[3];
		double y = anchorm[4];
		double z = anchorm[5];

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T;T << x, y, z;
		Vector3d P;P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = locw * w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		err = err / (locw*locw);
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
	double squaredCostEveluate() {
		double resid[3];
		operator()(point, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		err = err / (locw*locw);
		return err;
	}
	bool bInlier() {
		return enable;
	}
private:
	bool enable = true;
	double* point;
	double locw = 1.0;
	double anchorm[6];
	Vector3d xim1;
	double* w;
};

//
//struct BaFeatureProjCostFunction_Ancker {
//public:
//	BaFeatureProjCostFunction_Ancker(Vector3d& xibase, double* w_)
//	{
//		w = w_;
//		xim1 = xibase;
//		anchorm[0] = anchorm[1] = anchorm[2] = anchorm[3] = anchorm[4] = anchorm[5] = 0;
//	};
//	BaFeatureProjCostFunction_Ancker(Vector3d& xibase, double* w_, double* anchormotion)
//	{
//		w = w_;
//		xim1 = xibase;
//		anchorm[0] = anchormotion[0];
//		anchorm[1] = anchormotion[1];
//		anchorm[2] = anchormotion[2];
//		anchorm[3] = anchormotion[3];
//		anchorm[4] = anchormotion[4];
//		anchorm[5] = anchormotion[5];
//	};
//	void set(double* point_) {
//		point = point_;
//	}
//
//
//	bool operator()(const double* parameters2, double* residual) const {
//		double rx = anchorm[0];
//		double ry = anchorm[1];
//		double rz = anchorm[2];
//		double x = anchorm[3];
//		double y = anchorm[4];
//		double z = anchorm[5];
//
//		double px = parameters2[0];
//		double py = parameters2[1];
//		double pz = parameters2[2];
//
//		Matrix3d R = axisRot2R(rx, ry, rz);
//
//		Vector3d T;T << x, y, z;
//		Vector3d P;P << px, py, pz;
//		P = R.transpose()*(P - T);
//		P = P.normalized();
//		Vector3d err = w[0] * xim1.cross(P);
//		residual[0] = err(0);
//		residual[1] = err(1);
//		residual[2] = err(2);
//		return true;
//	}
//	bool outlierRejection(double threshold) {
//		enable = true;
//		double resid[3];
//		operator()(point, resid);
//		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
//		if (err > threshold) enable = false;
//		else enable = true;
//		return enable;
//	}
//private:
//	bool enable = true;
//	double* point;
//
//	double anchorm[6];
//	Vector3d xim1;
//	double* w;
//};


struct scaleAnckor {
public:
	scaleAnckor(double initScale_) {
		initScale = initScale_;
	}
	bool operator()(const double* paramTrans, double* residual) const {

		double scale = paramTrans[0] * paramTrans[0] + paramTrans[1] * paramTrans[1] + paramTrans[2] * paramTrans[2];
		residual[0] = initScale - scale;
		return true;
	}
private:
	double initScale;
};

struct scaleAnckor_b {
public:
	scaleAnckor_b(double* originT_) {
		initScale = originT_[0]* originT_[0]+ originT_[1]*originT_[1]* originT_[2]* originT_[2];
		originT = originT_;

	}
	bool operator()(const double* paramTrans, double* residual) const {

		Vector3d movedT;movedT << originT[0]+paramTrans[3], originT[1] + paramTrans[4], originT[2] + paramTrans[5];

		double scale = movedT.squaredNorm();
		residual[0] = initScale - scale;
		return true;
	}
private:
	double* originT;
	double initScale;
};


struct BaFeatureProjCostFunctionT {
public:
	BaFeatureProjCostFunctionT(Vector3d& xibase, double* w_, double* rot_)
	{
		w = w_;

		xim1 = xibase;
		rot = rot_;
	};
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		double rx = rot[0];
		double ry = rot[1];
		double rz = rot[2];
		double x = parameters[0];
		double y = parameters[1];
		double z = parameters[2];

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T;T << x, y, z;
		Vector3d P;P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d xim1;
	double* w;
	double* rot;
};

struct BaFeatureProjCostFunctionR {
public:
	BaFeatureProjCostFunctionR(Vector3d& xibase, double* w_, double* trans_)
	{
		w = w_;

		xim1 = xibase;
		trans = trans_;
	};
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		double x = trans[0];
		double y = trans[1];
		double z = trans[2];
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];

		double px = parameters2[0];
		double py = parameters2[1];
		double pz = parameters2[2];

		Matrix3d R = axisRot2R(rx, ry, rz);

		Vector3d T;T << x, y, z;
		Vector3d P;P << px, py, pz;
		P = R.transpose()*(P - T);
		P = P.normalized();
		Vector3d err = w[0] * xim1.cross(P);
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
private:
	Vector3d xim1;
	double* w;
	double* trans;
};







struct BADepthKnownCostFunc_PrjT_anckor2 {
public:
	BADepthKnownCostFunc_PrjT_anckor2(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_, double* t_,double w1,double w2)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
		paramTrans = t_;
		anchorm[0] = anchorm[1] = anchorm[2] = anchorm[3] = anchorm[4] = anchorm[5] = 0;
		wa1 = w1;wa2 = w2;
	};
	BADepthKnownCostFunc_PrjT_anckor2(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_, double* t_, double* anchormotion, double w1, double w2)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot = rot_;
		paramTrans = t_;
		anchorm[0] = anchormotion[0];
		anchorm[1] = anchormotion[1];
		anchorm[2] = anchormotion[2];
		anchorm[3] = anchormotion[3];
		anchorm[4] = anchormotion[4];
		anchorm[5] = anchormotion[5];
		wa1 = w1;wa2 = w2;
	};

	void set(double* paramt,double* paramt2) {
		params = paramt;
		params2 = paramt2;
	};

	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}


		Vector3d t1, t2;
		if (parameters != NULL) {
			t1 << parameters[0], parameters[1], parameters[2];
		}
		else {
			t1 << 0, 0, 0;
		}
		t2 << parameters2[0], parameters2[1], parameters2[2];
		Vector3d dt1 = wa1 * t1 + wa2 * t2;
		

		double rx = anchorm[0];
		double ry = anchorm[1];
		double rz = anchorm[2];
		double x = anchorm[3];
		double y = anchorm[4];
		double z = anchorm[5];

		double rx2 = paramRot[0];//world 2 local
		double ry2 = paramRot[1];
		double rz2 = paramRot[2];
		double x2 = paramTrans[0] + dt1(0);
		double y2 = paramTrans[1] + dt1(1);
		double z2 = paramTrans[2] + dt1(2);


		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
		/*double Val = T.dot(vjd);
		double s = (p - c).dot(T - Val * vjd) / (1 - Val * Val);
		double alpha = (p - c).dot(vjd - Val * T) / (1 - Val * Val);*/

		//residual[0] = (((p - c) - alpha * vjd) - s * T).norm();

		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(params, params2, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
private:
	bool enable = true;
	double* params,*params2;
	double anchorm[6];
	Vector3d p;
	Vector3d v;
	Vector3d c;
	double* w;	double* paramRot; double* paramTrans;
	double wa1, wa2;
};


struct BADepthKnownCostFunc_PrjT_anchor_wrap {
public:
	BADepthKnownCostFunc_PrjT_anchor_wrap(BADepthKnownCostFunc_PrjT_anckor2* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		return f_->operator()(parameters, parameters2, residual);
	}
private:
	BADepthKnownCostFunc_PrjT_anckor2* f_;
};
struct BADepthKnownCostFunc_PrjT_anchor_wrap_anc {
public:
	BADepthKnownCostFunc_PrjT_anchor_wrap_anc(BADepthKnownCostFunc_PrjT_anckor2* f) { f_ = f; }
	bool operator()(const double* parameters2, double* residual) const {
		return f_->operator()(NULL, parameters2, residual);
	}
private:
	BADepthKnownCostFunc_PrjT_anckor2* f_;
};


struct BADepthKnownCostFunc_PrjT_anckor_rev2 {
public:
	BADepthKnownCostFunc_PrjT_anckor_rev2(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_, double* t_, double w1, double w2)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramTrans = t_;
		paramRot = rot_;
		anchorm[0] = anchorm[1] = anchorm[2] = anchorm[3] = anchorm[4] = anchorm[5] = 0;		wa1 = w1;wa2 = w2;
	};
	BADepthKnownCostFunc_PrjT_anckor_rev2(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot_, double* t_, double* anchormotion, double w1, double w2)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramTrans = t_;
		paramRot = rot_;
		anchorm[0] = anchormotion[0];
		anchorm[1] = anchormotion[1];
		anchorm[2] = anchormotion[2];
		anchorm[3] = anchormotion[3];
		anchorm[4] = anchormotion[4];
		anchorm[5] = anchormotion[5];		wa1 = w1;wa2 = w2;
	};

	void set(double* paramt, double* paramt2) {
		params = paramt;
		params2 = paramt2;
	};
	bool operator()(const double* parameters,const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}


		Vector3d t1, t2;
		if (parameters != NULL) {
			t1 << parameters[0], parameters[1], parameters[2];
		}
		else {
			t1 << 0, 0, 0;
		}
		t2 << parameters2[0], parameters2[1], parameters2[2];
		Vector3d dt1 = wa1 * t1 + wa2 * t2;

		double rx = anchorm[0];
		double ry = anchorm[1];
		double rz = anchorm[2];
		double x = anchorm[3];
		double y = anchorm[4];
		double z = anchorm[5];

		double rx2 = paramRot[0];//world 2 local
		double ry2 = paramRot[1];
		double rz2 = paramRot[2];
		double x2 = paramTrans[0] + dt1(0);
		double y2 = paramTrans[1] + dt1(1);
		double z2 = paramTrans[2] + dt1(2);

		//R,xyz2: local 2 world
		//R2,xyz: world 2 local
		Matrix3d R = axisRot2R(rx2, ry2, rz2);
		Matrix3d R2 = axisRot2R(rx, ry, rz);

		Vector3d T;T << x2, y2, z2;
		Vector3d T2;T2 << x, y, z;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(params,params2, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
private:
	bool enable = true;
	double* params, *params2;

	double anchorm[6];
	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot;
	double* paramTrans;
	double wa1, wa2;
};


struct BADepthKnownCostFunc_PrjT_anchor_rev_wrap {
public:
	BADepthKnownCostFunc_PrjT_anchor_rev_wrap(BADepthKnownCostFunc_PrjT_anckor_rev2* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		return f_->operator()(parameters, parameters2, residual);
	}
private:
	BADepthKnownCostFunc_PrjT_anckor_rev2* f_;
};

struct BADepthKnownCostFunc_PrjT_anchor_rev_wrap_anc {
public:
	BADepthKnownCostFunc_PrjT_anchor_rev_wrap_anc(BADepthKnownCostFunc_PrjT_anckor_rev2* f) { f_ = f; }
	bool operator()(const double* parameters2, double* residual) const {
		return f_->operator()(NULL, parameters2, residual);
	}
private:
	BADepthKnownCostFunc_PrjT_anckor_rev2* f_;
};


// case 1 in same section
struct BADepthKnownCostFunc_PrjT2 {
public:
	BADepthKnownCostFunc_PrjT2(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot1_, double* rot2_, double* trans1_, double* trans2_,double w1,double w2, double w3, double w4)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot1 = rot1_;
		paramRot2 = rot2_;
		paramTrans1 = trans1_;
		paramTrans2 = trans2_;
		wa1=w1, wa2 = w2, wb1 = w3, wb2 = w4;



	};


	void set(double* paramt1, double *paramt2) {
		trans1 = paramt1;
		trans2 = paramt2;
	};

	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}


		Vector3d t1, t2;
		t1 << parameters[0], parameters[1], parameters[2];
		t2 << parameters2[0], parameters2[1], parameters2[2];
		Vector3d dt1 = wa1 * t1 + wa2 * t2;
		Vector3d dt2 = wb1 * t1 + wb2 * t2;

		double rx = paramRot1[0];//local 2 world
		double ry = paramRot1[1];
		double rz = paramRot1[2];
		double x = paramTrans1[0] + + dt1(0);
		double y = paramTrans1[1] + dt1(1);
		double z = paramTrans1[2] + dt1(2);

		double rx2 = paramRot2[0];//world 2 local
		double ry2 = paramRot2[1];
		double rz2 = paramRot2[2];
		double x2 = paramTrans2[0] + dt2(0);
		double y2 = paramTrans2[1] + dt2(1);
		double z2 = paramTrans2[2] + dt2(2);

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(trans1, trans2, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
private:
	bool enable;
	double* trans1;
	double* trans2;

	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot1; double* paramTrans1; 
	double wa1, wa2, wb1, wb2;
	double* paramRot2; double* paramTrans2;
};


// case 1 in adjacent section
struct BADepthKnownCostFunc_PrjT3 {
public:
	BADepthKnownCostFunc_PrjT3(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot1_, double* rot2_, double* trans1_, double* trans2_)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot1 = rot1_;
		paramRot2 = rot2_;
		paramTrans1 = trans1_;
		paramTrans2 = trans2_;
	};


	void set(double* paramt1, double *paramt2, double *paramt3) {
		trans1 = paramt1;
		trans2 = paramt2;
		trans3 = paramt3;
	};

	bool operator()(const double* parameters, const double* parameters2, const double* parameters3, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}


		Vector3d t1, t2, t3;
		t1 << parameters[0], parameters[1], parameters[2];
		t2 << parameters2[0], parameters2[1], parameters2[2];
		t3 << parameters3[0], parameters3[1], parameters3[2];
		Vector3d dt1 = wa1 * t1 + wa2 * t2;
		Vector3d dt2 = wb1 * t2 + wb2 * t3;

		double rx = paramRot1[0];//local 2 world
		double ry = paramRot1[1];
		double rz = paramRot1[2];
		double x = paramTrans1[0] + +dt1(0);
		double y = paramTrans1[1] + dt1(1);
		double z = paramTrans1[2] + dt1(2);

		double rx2 = paramRot2[0];//world 2 local
		double ry2 = paramRot2[1];
		double rz2 = paramRot2[2];
		double x2 = paramTrans2[0] + dt2(0);
		double y2 = paramTrans2[1] + dt2(1);
		double z2 = paramTrans2[2] + dt2(2);

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(trans1, trans2,trans3, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
private:
	bool enable;
	double* trans1;
	double* trans2;
	double* trans3;

	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot1; double* paramTrans1;
	double wa1, wa2, wb1, wb2;
	double* paramRot2; double* paramTrans2;
};


// case 1 in hanareta section
struct BADepthKnownCostFunc_PrjT4 {
public:
	BADepthKnownCostFunc_PrjT4(Vector3d& p3d_, Vector3d& eyeDirec_, double* w_, double* rot1_, double* rot2_, double* trans1_, double* trans2_, double w1, double w2, double w3, double w4)
	{
		w = w_;
		p = p3d_;
		v = eyeDirec_;
		paramRot1 = rot1_;
		paramRot2 = rot2_;
		paramTrans1 = trans1_;
		paramTrans2 = trans2_;
		wa1 = w1, wa2 = w2, wb1 = w3, wb2 = w4;
	};


	void set(double* paramt1, double *paramt2, double *paramt3,double* paramt4) {
		trans1 = paramt1;
		trans2 = paramt2;
		trans3 = paramt3;
		trans4 = paramt4;
	};

	bool operator()(const double* parameters, const double* parameters2, const double* parameters3, const double* parameters4, double* residual) const {
		if (!enable) {
			residual[0] = residual[1] = residual[2] = 0;
			return true;
		}


		Vector3d t1, t2, t3, t4;
		if (parameters != NULL) {
			t1 << parameters[0], parameters[1], parameters[2];
		}
		else {
			t1 << 0, 0, 0;
		}
		t2 << parameters2[0], parameters2[1], parameters2[2];
		if (parameters3 != NULL) {
			t3 << parameters3[0], parameters3[1], parameters3[2];
		}
		else {
			t3 << 0, 0, 0;
		}
//		t3 << parameters3[0], parameters3[1], parameters3[2];
		t4 << parameters4[0], parameters4[1], parameters4[2];
		Vector3d dt1 = wa1 * t1 + wa2 * t2;
		Vector3d dt2 = wb1 * t3 + wb2 * t4;

		double rx = paramRot1[0];//local 2 world
		double ry = paramRot1[1];
		double rz = paramRot1[2];
		double x = paramTrans1[0] + +dt1(0);
		double y = paramTrans1[1] + dt1(1);
		double z = paramTrans1[2] + dt1(2);

		double rx2 = paramRot2[0];//world 2 local
		double ry2 = paramRot2[1];
		double rz2 = paramRot2[2];
		double x2 = paramTrans2[0] + dt2(0);
		double y2 = paramTrans2[1] + dt2(1);
		double z2 = paramTrans2[2] + dt2(2);

		Matrix3d R = axisRot2R(rx, ry, rz);
		Matrix3d R2 = axisRot2R(rx2, ry2, rz2);

		Vector3d T;T << x, y, z;
		Vector3d T2;T2 << x2, y2, z2;
		Vector3d p_ = R * p + T;

		Vector3d plot = R2.transpose() * (p_ - T2);

		Vector3d err = w[0] * v.cross(plot) / plot.norm();
		residual[0] = err(0);
		residual[1] = err(1);
		residual[2] = err(2);
		return true;
	}
	bool outlierRejection(double threshold) {
		enable = true;
		double resid[3];
		operator()(trans1, trans2, trans3, trans4, resid);
		double err = resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2];
		if (err > threshold) enable = false;
		else enable = true;
		return enable;
	}
private:
	bool enable;
	double* trans1;
	double* trans2;
	double* trans3;
	double* trans4;

	Vector3d p;
	Vector3d v;
	double* w;
	double* paramRot1; double* paramTrans1;
	double wa1, wa2, wb1, wb2;
	double* paramRot2; double* paramTrans2;
};

//case 1_ anc
struct BADepthKnownCostFunc_PrjT_wrap_1a {
public:
	BADepthKnownCostFunc_PrjT_wrap_1a(BADepthKnownCostFunc_PrjT4* f) { f_ = f; }
	bool operator()( const double* parameters2, double* residual) const {
		return f_->operator()(NULL, parameters2, NULL, parameters2, residual);
	}
private:
	BADepthKnownCostFunc_PrjT4* f_;
};
//case 1
struct BADepthKnownCostFunc_PrjT_wrap_1 {
public:
	BADepthKnownCostFunc_PrjT_wrap_1(BADepthKnownCostFunc_PrjT4* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		return f_->operator()(parameters, parameters2, parameters, parameters2, residual);
	}
private:
	BADepthKnownCostFunc_PrjT4* f_;
};

//case2 anc
struct BADepthKnownCostFunc_PrjT_wrap_2a {
public:
	BADepthKnownCostFunc_PrjT_wrap_2a(BADepthKnownCostFunc_PrjT4* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, double* residual) const {
		if (backward) {
			return f_->operator()( parameters, parameters2, NULL, parameters, residual);
		}
		else {
			return f_->operator()(NULL, parameters, parameters, parameters2, residual);
		}
		
	}
	bool backward = false;
private:
	BADepthKnownCostFunc_PrjT4* f_;
};
struct BADepthKnownCostFunc_PrjT_wrap_2 {
public:
	BADepthKnownCostFunc_PrjT_wrap_2(BADepthKnownCostFunc_PrjT4* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, const double* parameters3, double* residual) const {
		if (backward) {
			return f_->operator()(parameters, parameters2, parameters2, parameters3, residual);
		}
		else {
			return f_->operator()(parameters2 , parameters3, parameters, parameters2, residual);
		}

	}
	bool backward = false;
private:
	BADepthKnownCostFunc_PrjT4* f_;
};

//case3 anc
struct BADepthKnownCostFunc_PrjT_wrap_3a {
public:
	BADepthKnownCostFunc_PrjT_wrap_3a(BADepthKnownCostFunc_PrjT4* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, const double* parameters3, double* residual) const {
		if (backward) {
			return f_->operator()(parameters2, parameters3, NULL, parameters, residual);
		}
		else {
			return f_->operator()(NULL, parameters, parameters2, parameters3, residual);
		}

	}
	bool backward = false;
private:
	BADepthKnownCostFunc_PrjT4* f_;
};
struct BADepthKnownCostFunc_PrjT_wrap_3 {
public:
	BADepthKnownCostFunc_PrjT_wrap_3(BADepthKnownCostFunc_PrjT4* f) { f_ = f; }
	bool operator()(const double* parameters, const double* parameters2, const double* parameters3, const double* parameters4, double* residual) const {
		if (backward) {
			return f_->operator()(parameters3, parameters4, parameters, parameters2, residual);
		}
		else {
			return f_->operator()(parameters, parameters2, parameters3, parameters4, residual);
		}

	}
	bool backward = false;
private:
	BADepthKnownCostFunc_PrjT4* f_;
};


class AdjustableHuberLoss : public ceres::LossFunction {
public:
	explicit AdjustableHuberLoss(double a) : a_(a), b_(a * a) { }
	virtual void Evaluate(double, double*) const;
	void resetScale(double a) {
		a_ = a;
		b_ = a*a;
	}
private:
	double a_;
	// b = a^2.
	double b_;
};


