//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once

#include <camera.h>
#include <opencv2/opencv.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <opengv\relative_pose\CentralRelativeAdapter.hpp>
#include <opengv\relative_pose\methods.hpp>
#include <opengv\sac\Ransac.hpp>
#include <opengv\sac_problems\relative_pose\CentralRelativePoseSacProblem.hpp>
#include <ceres\ceres.h>
#include <utility.h>
#include <ply_object.h>
#include <pc_proc.h>

void cameraPoseEst_lin(std::vector<Eigen::Vector3d> bvs1, std::vector<Eigen::Vector3d> bvs2, cvl_toolkit::_6dof& mot, double threshold = (1.0 - cos(atan(sqrt(2.0)*0.5 / 400.0))), int max_itr = 1000);
void cameraPoseRelEst_non_lin(std::vector<Eigen::Vector3d> bvs1, std::vector<Eigen::Vector3d> bvs2, cvl_toolkit::_6dof& mot, double loss_scale = 1.0e-3);

std::vector<int>  nonLinearAbsTransSolving(std::vector<Eigen::Vector3d> points3d, std::vector<Eigen::Vector3d> bearingVect, cvl_toolkit::_6dof& motion, double tolerance);
std::vector<int> nonLinearAbsSolving(std::vector<Eigen::Vector3d> points3d, std::vector<Eigen::Vector3d> bearingVect, cvl_toolkit::_6dof& motion, double tolerance);

std::vector<Eigen::Matrix4d> initCameraMotionComputation(std::vector<cv::Mat> imgs, std::vector<int> motionList, std::vector<cv::Mat>imgsg, camera* pc,cv::Mat mask);
Eigen::Matrix4d calibrationFromMotion(std::vector<Eigen::Matrix4d>& motA, std::vector<Eigen::Matrix4d> motB);

std::vector<Eigen::Matrix4d> cameraMotionComputationWithDepth(std::vector<cv::Mat> imgs_gray, std::vector<Eigen::Matrix4d>& initCamMotion, std::vector<cvl_toolkit::plyObject> prs
	, std::vector<Eigen::Matrix4d> lidarPos, std::vector<int> motionList, Eigen::Matrix4d extParam, double thres, camera* pc, cv::Mat mask);


Eigen::Matrix4d calibrationFromMotion2(std::vector<Eigen::Matrix4d> motA, std::vector<Eigen::Matrix4d> motB);




//cost function

struct depthUnknownDirectionCostFunc {
public:
	depthUnknownDirectionCostFunc(Eigen::Vector3d& xprev, Eigen::Vector3d& xcur)
	{
		xim1 = xprev;
		xi = xcur;
	};
	bool operator()(const double* parameters, double* residual) const {
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];
		double phi = parameters[3];
		double theta = parameters[4];

		double x = sin(phi)*cos(theta);
		double y = sin(phi)*sin(theta);
		double z = cos(phi);

		Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(rx, ry, rz);
		Eigen::Vector3d T; T << x, y, z;

		T = T.normalized();
		Eigen::Vector3d vec = R * xi;

		Eigen::Vector3d vert = vec.cross(T).normalized();

		residual[0] = vert.dot(xim1);
		return true;
	}
private:
	Eigen::Vector3d xi;
	double x2d, y2d;
	Eigen::Vector3d xim1;
};


struct axisAlignCostFunc {
public:
	axisAlignCostFunc(Eigen::Matrix3d& RA_, Eigen::Matrix3d& RB_)
	{
		double anglea, angleb;
		Eigen::Vector3d axis;
		Eigen::Matrix3d rot = RA_;
		cvl_toolkit::conversion::mat2axis_angle(rot, p3d, anglea);
		cvl_toolkit::conversion::mat2axis_angle(RB_, eyeDirec, angleb);

		weight = sqrt(1 - cos(anglea));


	};
	bool operator()(const double* parameters, double* residual) const {
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];

		Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(rx, ry, rz);
		Eigen::Vector3d plot = (R.transpose()*p3d).normalized();

		residual[0] = eyeDirec.cross(plot).norm();//eyeDirec.cross(plot).norm();
		return true;
	}
private:
	Eigen::Vector3d p3d;
	Eigen::Vector3d eyeDirec;
	double weight;
};



struct projectionCostFunc {
public:
	projectionCostFunc(Eigen::Vector3d& p3d_, Eigen::Vector3d& eyeDirec_)
	{
		p3d = p3d_;
		eyeDirec = eyeDirec_;
		sqrt(p3d.norm());
	};
	bool operator()(const double* parameters, double* residual) const {
		double rx = parameters[0];
		double ry = parameters[1];
		double rz = parameters[2];
		double x = parameters[3];
		double y = parameters[4];
		double z = parameters[5];

		Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(rx, ry, rz);
		Eigen::Vector3d T; T << x, y, z;
		Eigen::Vector3d plot = R.transpose()*(p3d - T).normalized();

		residual[0] = eyeDirec.cross(plot).norm();
		return true;
	}

private:
	Eigen::Vector3d p3d;
	double x2d, y2d;
	Eigen::Vector3d eyeDirec;
	int w, h;

	double weight;
};


struct projectionCostFunc_T {
public:
	projectionCostFunc_T(Eigen::Vector3d& p3d_, Eigen::Vector3d& eyeDirec_, double* rot_)
	{
		p3d = p3d_;
		eyeDirec = eyeDirec_;
		rot = rot_;
	};
	bool operator()(const double* parameters, double* residual) const {
		double x = parameters[0];
		double y = parameters[1];
		double z = parameters[2];
		double rx = rot[0];
		double ry = rot[1];
		double rz = rot[2];

		Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(rx, ry, rz);
		Eigen::Vector3d T; T << x, y, z;
		Eigen::Vector3d plot = R.transpose()*(p3d - T).normalized();

		residual[0] = eyeDirec.cross(plot).norm();
		return true;
	}
private:
	Eigen::Vector3d p3d;
	double x2d, y2d;
	Eigen::Vector3d eyeDirec;
	double* rot;
	int w, h;
	double weight;
};
