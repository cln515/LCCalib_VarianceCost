//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <camera.h>

void omnidirectionalCamera::projection(Eigen::Vector3d v, double& x, double& y) {
	double px = v(0), py = v(1), pz = v(2);

	double r = sqrt(px*px + py * py + pz * pz);
	double theta = atan2(py, px);
	double phi = acos(pz / r);
	x = (-theta + M_PI)*width / (M_PI * 2);
	y = phi / M_PI * height;

};

Eigen::Vector3d omnidirectionalCamera::reverse_projection(Eigen::Vector3d v) {

	Eigen::Vector3d ret;
	double theta = -(v(0)*(M_PI * 2) / width - M_PI);
	double phi = v(1) / height * M_PI;
	ret << sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi);

	return ret;
};



void perspectiveCamera::projection(Eigen::Vector3d v, double& x, double& y) {
	Eigen::Matrix3d intrinsic;
	intrinsic << fx, 0, cx,
		0, fy, cy,
		0, 0, 1;
	Eigen::Vector3d vd = intrinsic * v;
	x = vd(0) / vd(2);
	y = vd(1) / vd(2);
}

Eigen::Vector3d perspectiveCamera::reverse_projection(Eigen::Vector3d v) {
	Eigen::Matrix3d intrinsic, inv_intrinsic;
	intrinsic << fx, 0, cx,
		0, fy, cy,
		0, 0, 1;
	inv_intrinsic = intrinsic.inverse();
	Eigen::Vector3d vd = inv_intrinsic * v;
	return vd.normalized();
}