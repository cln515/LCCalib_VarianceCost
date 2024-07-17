//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#define NOMINMAX
#include <mb_calibration.h>
#include <iostream>
#include <fstream>
#include <nlohmann\json.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pc_proc.h>
#include <camera.h>

long long findIdx(float* scan,float qt,long long maxIdx ) {
	__int64 startIdx = 0, endIdx = maxIdx - 1, centerIdx = startIdx;
	while (endIdx - startIdx > 1) {
		centerIdx = (endIdx + startIdx) / 2;
		
		double tc = scan[centerIdx*5+4];
		if (tc >= qt) {
			endIdx = centerIdx;
		};
		if (tc < qt) {
			startIdx = centerIdx;
		};
	}
	centerIdx = (endIdx + startIdx) / 2 + 1;
	return centerIdx;
}

void omniTrans(double x, double y, double z, double& phi, double& theta) {
	double r = sqrt(x*x + y * y + z * z);
	theta = atan2(y, x);
	phi = acos(z / r);
};

bool makeFolder(std::string folderPath) {
	std::cout << "create " + folderPath << std::endl;
#if defined(WIN32) || defined(WIN64)
	if (MakeSureDirectoryPathExists(folderPath.c_str())) {
		std::cout << "succeeded." << std::endl;
		return true;
	}
	else {
		std::cout << "failed" << std::endl;
		return false;
	}
#elif defined(__unix__)

	std::string folderPath_make = folderPath.substr(0, folderPath.find_last_of('/'));
	//	std::cout<<folderPath_make;
	std::string cmd = "mkdir -p " + folderPath_make;
	system(cmd.c_str());
#endif
}

void goodFeatureToTrack_onProjection__(cv::Mat image, std::vector<cv::Point2f> proj_points, std::vector<int>& selectedIdx, double minDistance, int maxCorners, double cornerThres) {
	cv::Mat eig, tmp;
	cv::cornerHarris(image, eig, 3, 5, 0.04);//corner calculation
	cv::Mat tmpMask = cv::Mat::zeros(image.rows, image.cols, CV_8U);
	std::map<unsigned int, int> point_idx;
	std::map<unsigned int, double> point_dist;
	for (int j = 0; j < proj_points.size(); j++) {
		cv::Point2f p = proj_points.at(j);
		unsigned int pint = ((int)p.x) + ((int)p.y) * image.cols;;
		if (p.x >= image.cols || p.y >= image.rows)continue;
		tmpMask.at<uchar>(p) = 255;//remove mask on projected point
		double cendx = p.x - (int)p.x - 0.5;
		double cendy = p.y - (int)p.y - 0.5;
		double distcenter = cendx * cendx + cendy * cendy;
		auto itr = point_dist.find(pint);
		if (itr == point_dist.end() || itr->second > distcenter) {
			point_dist.insert(std::map<unsigned int, double>::value_type(pint, distcenter));
			point_idx.insert(std::map<unsigned int, int>::value_type(pint, j));
		}
	}

	cv::dilate(tmpMask, tmpMask, cv::Mat());//expansion non-masked area
	double maxVal;
	//non-maxima suppression
	cv::minMaxLoc(eig, 0, &maxVal, 0, 0, tmpMask);
	//cv::threshold(eig, eig, maxVal*1e-12, 0, cv::THRESH_TOZERO);
	cv::dilate(eig, tmp, cv::Mat());

	cv::Size imgsize = image.size();

	std::vector<const float*> tmpCorners;

	// collect list of pointers to features - put them into temporary image
	for (int y = 1; y < imgsize.height - 1; y++)
	{
		const float* eig_data = (const float*)eig.ptr(y);
		const float* tmp_data = (const float*)tmp.ptr(y);
		const uchar* mask_data = tmpMask.data ? tmpMask.ptr(y) : 0;

		for (int x = 1; x < imgsize.width - 1; x++)
		{
			float val = eig_data[x];
			if (val >= maxVal * cornerThres /*&& val == tmp_data[x]*/ && (!mask_data || mask_data[x]))
				tmpCorners.push_back(eig_data + x);
		}
	}
	//cv::sort(tmpCorners,tmpCorners,CV_SORT_DESCENDING);
	std::sort(tmpCorners.begin(), tmpCorners.end(), [](const float*& a, const float*& b) {return (*a) >= (*b); });

	std::vector<cv::Point2f> corners;
	size_t i, j, total = tmpCorners.size(), ncorners = 0;
	if (minDistance >= 1)
	{
		// Partition the image into larger grids
		int w = image.cols;
		int h = image.rows;

		const int cell_size = cvRound(minDistance);
		const int grid_width = (w + cell_size - 1) / cell_size;
		const int grid_height = (h + cell_size - 1) / cell_size;

		std::vector<std::vector<cv::Point2f> > grid(grid_width * grid_height);

		minDistance *= minDistance;

		for (i = 0; i < total; i++)
		{
			int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
			int y = (int)(ofs / eig.step);
			int x = (int)((ofs - y * eig.step) / sizeof(float));

			bool good = true;

			int x_cell = x / cell_size;
			int y_cell = y / cell_size;

			int x1 = x_cell - 1;
			int y1 = y_cell - 1;
			int x2 = x_cell + 1;
			int y2 = y_cell + 1;

			// boundary check
			x1 = std::max(0, x1);
			y1 = std::max(0, y1);
			x2 = std::min(grid_width - 1, x2);
			y2 = std::min(grid_height - 1, y2);

			for (int yy = y1; yy <= y2; yy++)
			{
				for (int xx = x1; xx <= x2; xx++)
				{
					std::vector <cv::Point2f>& m = grid[yy * grid_width + xx];

					if (m.size())
					{
						for (j = 0; j < m.size(); j++)
						{
							float dx = x - m[j].x;
							float dy = y - m[j].y;

							if (dx * dx + dy * dy < minDistance)
							{
								good = false;
								goto break_out;
							}
						}
					}
				}
			}

		break_out:

			if (good)
			{
				// printf("%d: %d %d -> %d %d, %d, %d -- %d %d %d %d, %d %d, c=%d\n",
				//    i,x, y, x_cell, y_cell, (int)minDistance, cell_size,x1,y1,x2,y2, grid_width,grid_height,c);
				grid[y_cell * grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

				cv::Point2f p = cv::Point2f((float)x, (float)y);
				unsigned int pint = ((int)p.x) + ((int)p.y) * image.cols;
				auto itr2 = point_idx.find(pint);
				if (itr2 != point_idx.end()) {
					selectedIdx.push_back(point_idx.at(pint));
					//					corners.push_back(p);
					++ncorners;
				}


				if (maxCorners > 0 && (int)ncorners == maxCorners)
					break;
			}
		}
	}
	else
	{
		for (i = 0; i < total; i++)
		{
			int ofs = (int)((const uchar*)tmpCorners[i] - eig.data);
			int y = (int)(ofs / eig.step);
			int x = (int)((ofs - y * eig.step) / sizeof(float));

			cv::Point2f p = cv::Point2f((float)x, (float)y);
			unsigned int pint = ((int)p.x) + ((int)p.y) * image.cols;
			auto itr2 = point_idx.find(pint);
			if (itr2 != point_idx.end()) {
				selectedIdx.push_back(point_idx.at(pint));
				//					corners.push_back(p);
				++ncorners;
			}

			if (maxCorners > 0 && (int)ncorners == maxCorners)
				break;
		}
	}
	//selectedIdx.clear();
	//for (int j = 0;j<corners.size();j++) {
	//	cv::Point2f p = corners.at(j);
	//	unsigned int pint = ((int)p.x) + ((int)p.y)*image.cols;
	//	auto itr2 = point_idx.find(pint);
	//	if (itr2 != point_idx.end()) {
	//		selectedIdx.push_back(point_idx.at(pint));
	//	}
	//}

}



std::vector<int> nonLinearScaleSolving(std::vector<Eigen::Vector3d> points3d, std::vector<Eigen::Vector3d> bearingVect, cvl_toolkit::_6dof& motion, double tolerance) {

	double optimizedParaTrans[3] = { motion.x,motion.y,motion.z };
	double optimizedParaFix[6] = { motion.rx,motion.ry,motion.rz,motion.x,motion.y,motion.z };

	int itr = 0;
	double tolerance_ = tolerance;

	std::vector<int> inlierIdx, logInlierIdx;
	cvl_toolkit::_6dof prevMotion = {0,0,0,0,0,0};
	Eigen::Vector3d camCenter; camCenter << prevMotion.x, prevMotion.y, prevMotion.z;
	//Direction solving
	double theta, phi;
	double angles[2];
	omniTrans(motion.x, motion.y, motion.z, angles[0], angles[1]);
	std::vector<double> errs;
	for (int i = 0; i < points3d.size(); i++) {
		Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(motion.rx, motion.ry, motion.rz);
		/*Vector3d T;T << x, y, z;
		T = T - camCenter;
		T.normalized();*/
		Eigen::Vector3d T; T << sin(angles[0])*cos(angles[1]), sin(angles[0])*sin(angles[1]), cos(angles[0]);
		Eigen::Vector3d vj = points3d.at(i) - camCenter;
		Eigen::Vector3d vjd = R * bearingVect.at(i);
		vj = vj.normalized();

		double V = T.dot(vjd);
		double s = (vj).dot(T - V * vjd) / (1 - V * V);
		double alpha = (vj).dot(vjd - V * T) / (1 - V * V);

		double err = (((vj)-alpha * vjd) - s * T).norm();//distance error

		errs.push_back(err);
		//		inlierIdx.push_back(i);
	}
	std::vector<double>errs_(errs);

	std::partial_sort(errs_.begin(), errs_.begin() + errs_.size() / 3, errs_.end());
	double thres = errs_.at(errs_.size() / 3 - 1);
	for (int i = 0; i < errs.size(); i++) {
		if (errs.at(i) < thres) {
			inlierIdx.push_back(i);
		}
	}

	//scale calculation
	std::vector<double> scales;
	//	vector<double> consts;
	Eigen::Vector3d ti; ti << sin(angles[0])*cos(angles[1]), sin(angles[0])*sin(angles[1]), cos(angles[0]);
	//ti = ti - camCenter;
	//ti = ti.normalized();
	double scale_res = 0;
	Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(optimizedParaFix[0], optimizedParaFix[1], optimizedParaFix[2]);
	std::vector<int> trackingInlierIdx(inlierIdx);
	for (int idx = 0; idx < trackingInlierIdx.size(); idx++) {
		double dj = (points3d.at(trackingInlierIdx.at(idx)) - camCenter).norm();
		Eigen::Vector3d vj = (points3d.at(trackingInlierIdx.at(idx)) - camCenter).normalized();
		Eigen::Vector3d vjd = R * bearingVect.at(trackingInlierIdx.at(idx));
		double leng = (dj*vj.dot(ti - (ti.dot(vjd))*vjd)) / (1 - (ti.dot(vjd))*(ti.dot(vjd)));
		scales.push_back(leng);
		//		consts.push_back(-1.0*leng / (vj.dot(nv)*dj));
		scale_res += leng;
	}
	scale_res /= trackingInlierIdx.size();
	itr = 0;
	int cntThres = 100;
	double scale_sum = 0;

	std::vector<int> allInlierIdx;//tracking point idx

//linear solving
	std::random_device rnd;
	std::mt19937 mt(rnd());
	std::uniform_int_distribution<> randp(0, scales.size() - 1);
	int maxcnt = 0;
	int maxidx = -1;
	while (itr < 150) {
		int idx = randp(mt);
		double cscale = scales.at(idx);
		int ccnt = 0;
		for (auto scale : scales) {
			if (fabs(scale - cscale) < 0.01) {
				ccnt++;
			}
		}
		if (maxidx < 0 || maxcnt < ccnt) {
			maxidx = idx;
			maxcnt = ccnt;
		}
		if (ccnt > scales.size() / 2)break;
		itr++;
	}
	double nscale = scales.at(maxidx);

	//average
	itr = 0;
	tolerance_ = tolerance;
	std::vector<int> keepedII;
	while (itr < 2) {
		inlierIdx.clear();
		int sumcnt = 0;
		double sum = 0;
		double ave = 0;
		double sigma = 0.0;
		int inlIdx = 0;
		for (auto scale : scales) {
			if (fabs(scale - nscale) < tolerance_) {
				sum += scale;
				double preave = ave;
				ave = 1 / (sumcnt + 1.0) * (sumcnt * ave + scale);
				sigma = (sumcnt*(sigma + preave * preave) + scale * scale) / (sumcnt + 1.0) - ave * ave;
				sumcnt++;
				inlierIdx.push_back(trackingInlierIdx.at(inlIdx));
			}
			inlIdx++;
		}
		if (sumcnt == 0)break;
		int i = 0;
		nscale = sum / sumcnt;

		keepedII.clear();
		keepedII = std::vector<int>(inlierIdx);
		if (itr == 1)break;
		itr++;
		tolerance_ = 2 * sqrt(sigma);
	}

	inlierIdx = keepedII;
	//Matrix3d R1, R2;
	//Vector3d v1, v2, c1, c2, c12;//in the world coordinate

	//R2 = axisRot2R(motion.rx, motion.ry, motion.rz);
	//c2 << motion.x, motion.y, motion.z;

	Eigen::Vector3d m;

	ti = nscale * ti;
	motion.rx = optimizedParaFix[0];
	motion.ry = optimizedParaFix[1];
	motion.rz = optimizedParaFix[2];
	motion.x = prevMotion.x + ti(0);
	motion.y = prevMotion.y + ti(1);
	motion.z = prevMotion.z + ti(2);
	std::cout << "nscale,tolerance:" << nscale << "," << tolerance_ << "," << inlierIdx.size() << "," << itr << std::endl;
	if (nscale > 0.05) {
		std::cout << "large:" << nscale << "," << tolerance_ << "," << inlierIdx.size() << "," << itr << std::endl;
	}
	//return summary_;
	return inlierIdx;
	//return logInlierIdx;
};


int main(int argc, char* argv[]) {
	std::cout << "Kitty on your lap!!" << std::endl;

	std::ifstream ifs(argv[1]);
	nlohmann::json config;
	ifs >> config;
	ifs.close();

	std::string scaninfo = config["scan"].get<std::string>();
	std::string outputpath = config["output_json"].get<std::string>();
	makeFolder(outputpath);

	ifs.open(scaninfo);
	nlohmann::json scanconf;
	ifs >> scanconf;
	ifs.close();
	long long ptnum = scanconf["lidar_pn"].get<long long>();
	int startFrm = scanconf["first_frame"].get<int>();
	int endFrm = scanconf["end_frame"].get<int>();

	ifs.open(scanconf["lidar"].get<std::string>(), std::ios::binary);
	float* allpt = (float*)malloc(sizeof(float) * ptnum * 5);
	ifs.read((char*)allpt, sizeof(float) * ptnum * 5);
	ifs.close();

	std::vector<std::string> cam_list = scanconf["camera_list"].get<std::vector<std::string>>();
	std::vector<double> ts = scanconf["camera_ts"].get<std::vector<double>>();
	double timeoffset = scanconf["sync_offset"].get<double>();

	ifs.open(config["extparam"].get<std::string>());
	nlohmann::json extjson;
	ifs >> extjson;
	ifs.close();

	std::vector<double> rotv = extjson["camera2lidar"]["rotation"].get<std::vector<double>>()
		, transv = extjson["camera2lidar"]["translation"].get<std::vector<double>>();


	nlohmann::json caminfo = config["camerainfo"].get<nlohmann::json>();
	std::string type = caminfo["type"].get<std::string>();
	camera * cam;
	if (type.compare("panoramic") == 0) {
		int w = caminfo["w"].get<int>();
		int h = caminfo["h"].get<int>();
		cam = new omnidirectionalCamera(w, h);
	}
	else if (type.compare("perspective") == 0) {
		//intrinsic
		nlohmann::json camera_intrinsic;
		ifs.open(caminfo["intrinsic"].get<std::string>());
		ifs >> camera_intrinsic;
		ifs.close();

		cam = new perspectiveCamera(camera_intrinsic["cx"].get<double>(), camera_intrinsic["cy"].get<double>(), camera_intrinsic["fx"].get<double>(), camera_intrinsic["fy"].get<double>());
	}
	else {
		std::cout << "invalid camera type" << std::endl;
		return 0;
	}
	cv::Mat mask= cv::Mat();
	
	if (!config["mask"].is_null()) {
		mask = cv::imread(config["mask"].get<std::string>(),0);	
	}

	Eigen::Matrix3d extR; extR << rotv[0], rotv[1], rotv[2], rotv[3], rotv[4], rotv[5], rotv[6], rotv[7], rotv[8];
	Eigen::Vector3d extt; extt << transv[0], transv[1], transv[2];

	std::cout << extR << std::endl;
	std::cout << extt << std::endl;

	int imagecnt = 0;
	cv::Mat previmg;
	std::vector<Eigen::Vector3d> pts;
	std::vector<cv::Point2f> impts,dst;
	std::vector<uchar> status;
	std::vector<float> err;
	double 	thres_ = 0.10;
	cvl_toolkit::_6dof prevmot = { 0,0,0,0,0,0 };

	std::vector<cvl_toolkit::_6dof> movement;
	std::vector<double> ts_l,ts_c;
	for (int frm = startFrm; frm<=endFrm; frm++) {
		//load frames
		cv::Mat inputimg = cv::imread(cam_list.at(imagecnt), cv::ImreadModes::IMREAD_GRAYSCALE);


		if (frm > startFrm) {
			cvl_toolkit::_6dof mot = { 0,0,0,0,0,0 };// = cvl_toolkit::conversion::m2_6dof(initCamMotion.at(imgid));

			std::vector<cv::Point2f> corners;
			cv::goodFeaturesToTrack(previmg, corners, 3000, 0.001, 20, mask);
			//rotation
			cv::calcOpticalFlowPyrLK(previmg, inputimg, corners, dst, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

			std::vector<Eigen::Vector3d> bvs_f1,bvs_f2;

			for (int idx = 0; idx < dst.size(); idx++) {
				if (status[idx] != '\1')continue;
				Eigen::Vector3d pix1(corners.at(idx).x, corners.at(idx).y, 1);
				Eigen::Vector3d bv1 = cam->reverse_projection(pix1);
				
				Eigen::Vector3d pix2(dst.at(idx).x, dst.at(idx).y, 1);
				Eigen::Vector3d bv2 = cam->reverse_projection(pix2);
				bvs_f1.push_back(bv1);
				bvs_f2.push_back(bv2);
			}

			cameraPoseRelEst_non_lin(bvs_f1, bvs_f2, mot,1.0);
			
			std::vector<Eigen::Vector3d> bearingVectors1, bearingVectors2, bearingVectors3, bearingVectors4;

			cv::calcOpticalFlowPyrLK(previmg, inputimg, impts, dst, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

			for (int idx = 0; idx < dst.size(); idx++) {
				if (status[idx] != '\1')continue;

				Eigen::Vector3d pix(dst.at(idx).x, dst.at(idx).y,1);
				Eigen::Vector3d bv2 = cam->reverse_projection(pix);
				bearingVectors2.push_back(bv2);
				bearingVectors1.push_back(pts.at(idx));
			}

			//initial
			
			std::vector<int> inlier;
			inlier = nonLinearScaleSolving(bearingVectors1, bearingVectors2 ,mot, thres_);


			Eigen::Matrix3d tempR = cvl_toolkit::conversion::axisRot2R(mot.rx, mot.ry, mot.rz);
			Eigen::Vector3d prevT; prevT << prevmot.x, prevmot.y, prevmot.z;
			Eigen::Matrix3d keyR = cvl_toolkit::conversion::axisRot2R(prevmot.rx, prevmot.ry, prevmot.rz);

			Eigen::Vector3d trans; trans << mot.x, mot.y, mot.z;
			trans = keyR * trans + prevT;
			mot.x = trans(0);
			mot.y = trans(1);
			mot.z = trans(2);
			tempR = keyR * tempR;
			cvl_toolkit::conversion::R2axisRot(tempR, mot.rx, mot.ry, mot.rz);
			
			prevmot = mot;
		}

		double tdelta = 0.1;
		double timecamera = ts.at(imagecnt);
		ts_c.push_back(timecamera);
		double timelidar = timecamera - timeoffset- tdelta;
		ts_l.push_back(timelidar+tdelta);

		long long lidaridx = findIdx(allpt,timelidar,ptnum);

		std::vector<Eigen::Vector3d> ptc;
		std::vector<cv::Point2f> srcc;
		std::vector<int> selectedIdx;

		while (allpt[lidaridx*5+4]<timelidar+ tdelta) {
			double ix, iy;
			Eigen::Vector3d pt(allpt[lidaridx * 5], allpt[lidaridx * 5 + 1], allpt[lidaridx * 5 + 2]);
			if (pt.norm() <= 0.05) { 
				lidaridx++;
				continue; 
			}
			pt = extR * pt + extt;

			cam->projection(pt, ix, iy);

			ptc.push_back(pt);

			srcc.push_back(cv::Point2f(ix, iy));
			lidaridx++;
		}

		goodFeatureToTrack_onProjection__(inputimg, srcc, selectedIdx, 3.0, 0, 0.0);
		impts.clear();
		pts.clear();
		for (int sIdx = 0; sIdx < selectedIdx.size(); sIdx++) {
			impts.push_back(srcc.at(selectedIdx.at(sIdx)));
			pts.push_back(ptc.at(selectedIdx.at(sIdx)));
		}

		movement.push_back(prevmot);
		previmg = inputimg;
		imagecnt++;
	}

	//output
	std::vector< std::vector<double>> movement_out_l;
	std::vector< std::vector<double>> movement_out_c;
	Eigen::Matrix4d extM;
	extM.block(0, 0, 3, 3) = extR;
	extM.block(0, 3, 3, 1) = extt;
	extM.block(3, 0, 1, 4) << 0, 0, 0, 1;
	for (int i = 0; i < movement.size(); i++) {
		std::vector<double> camerapose,lidarpose;
		cvl_toolkit::_6dof mot = movement.at(i);
		Eigen::Matrix4d camera_mot_m = cvl_toolkit::conversion::_6dof2m(mot);
		Eigen::Matrix4d lidar_mot = camera_mot_m * extM;
		
		Eigen::Matrix3d cRot = camera_mot_m.block(0, 0, 3, 3);
		Eigen::Vector4d qc = cvl_toolkit::conversion::dcm2q(cRot);
		if (qc(3) < 0)qc = -qc;
		camerapose.push_back(qc(0));		camerapose.push_back(qc(1));		camerapose.push_back(qc(2));
		camerapose.push_back(camera_mot_m(0,3));		camerapose.push_back(camera_mot_m(1, 3));		camerapose.push_back(camera_mot_m(2, 3));

		Eigen::Matrix3d lRot = lidar_mot.block(0, 0, 3, 3);
		Eigen::Vector4d ql = cvl_toolkit::conversion::dcm2q(lRot);
		if (ql(3) < 0)ql = -ql;
		lidarpose.push_back(ql(0));		lidarpose.push_back(ql(1));		lidarpose.push_back(ql(2));
		lidarpose.push_back(lidar_mot(0, 3));		lidarpose.push_back(lidar_mot(1, 3));		lidarpose.push_back(lidar_mot(2, 3));
		movement_out_l.push_back(lidarpose);
		movement_out_c.push_back(camerapose);
	}
	nlohmann::json jout;
	jout["basepcap"] = scanconf["lidar"].get<std::string>();
	jout["_lidarmove"] = movement_out_l;
	jout["_lidar_ts"] = ts_l;
	jout["_cameramove"] = movement_out_c;
	jout["_camera_ts"] = ts_c;
	std::ofstream ofs(outputpath);
	ofs << std::setw(4) << jout << std::endl;
	ofs.close();


	free(allpt);
	return 0;
}
