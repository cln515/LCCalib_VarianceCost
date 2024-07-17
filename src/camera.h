//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma once
#define _USE_MATH_DEFINES

#include <Eigen/Core>
#include <Eigen/Dense>


class camera {
public:
	virtual void projection(Eigen::Vector3d v, double& x, double& y) = 0;
	virtual Eigen::Vector3d reverse_projection(Eigen::Vector3d pix) = 0;
};


class omnidirectionalCamera :public camera {
private:
	double width, height;
public:
	omnidirectionalCamera(double width_, double height_) {
		width = width_;
		height = height_;
	}
	void projection(Eigen::Vector3d v, double& x, double& y);
	Eigen::Vector3d reverse_projection(Eigen::Vector3d pix);
};


class perspectiveCamera :public camera {
private:
	double cx, cy;
	double fx, fy;
public:

	perspectiveCamera(double cx_, double cy_, double fx_, double fy_) {
		cx = cx_;
		cy = cy_;
		fx = fx_;
		fy = fy_;
	}

	void projection(Eigen::Vector3d v, double& x, double& y);
	Eigen::Vector3d reverse_projection(Eigen::Vector3d pix);

};

