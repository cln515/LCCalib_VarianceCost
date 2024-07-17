//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <mb_calibration.h>

void cameraPoseRelEst_non_lin(std::vector<Eigen::Vector3d> bvs1, std::vector<Eigen::Vector3d> bvs2, cvl_toolkit::_6dof& mot, double tolerance) {
	double optPara[5];
	optPara[0] = mot.rx;
	optPara[1] = mot.ry;
	optPara[2] = mot.rz;
	//optPara[3] = mot.x;
	//optPara[4] = mot.y;
	//optPara[5] = mot.z;//redundant
	int itr = 0;
	double tolerance_ = tolerance;
	Eigen::Vector3d mtrans(mot.x,mot.y,mot.z);
	if (mtrans.norm() == 0)mtrans(2) = 1.0;
	cvl_toolkit::conversion::xyz2polar(mtrans, optPara[3], optPara[4]);
	while (itr < 10) {
		double sigma2 = 0;
		int inl = 0;
		ceres::Problem problem;
		ceres::CauchyLoss* loss;
		for (int i = 0; i < bvs1.size(); i++) {
			depthUnknownDirectionCostFunc* p = new depthUnknownDirectionCostFunc(bvs1.at(i), bvs2.at(i));
			double residual[1];
			p->operator()(optPara, residual);
			if (residual[0] * residual[0] < tolerance_*tolerance_) {
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<depthUnknownDirectionCostFunc, ceres::CENTRAL, 1, 5>(
					p);
				problem.AddResidualBlock(c, new ceres::HuberLoss(tolerance), optPara);
				inl++;
				sigma2 = (inl*(sigma2)+residual[0] * residual[0]) / (inl + 1.0);
			}
			else {
				delete p;
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 1e5;
		options.function_tolerance = 1e-8;
		options.parameter_tolerance = 1e-8;
		options.linear_solver_type = ceres::DENSE_QR;
		ceres::Solver::Summary summary;

		ceres::Solve(options, &problem, &summary);
		if (itr >= 1 && sigma2 < sin(0.002)*sin(0.002)) { std::cout << inl << std::endl; break; }
		tolerance_ = sqrt(sigma2);


		itr++;

	}
	mot.rx = optPara[0];
	mot.ry = optPara[1];
	mot.rz = optPara[2];
	Eigen::Vector3d ti; ti << sin(optPara[3])*cos(optPara[4]), sin(optPara[3])*sin(optPara[4]), cos(optPara[3]);//prev cam coordinates
	std::cout << itr << std::endl;
	mot.x = ti(0);
	mot.y = ti(1);
	mot.z = ti(2);
}


void cameraPoseEst_lin(std::vector<Eigen::Vector3d> bvs1, std::vector<Eigen::Vector3d> bvs2, cvl_toolkit::_6dof& mot, double threshold, int max_itr) {
	opengv::bearingVectors_t bvs1_, bvs2_;
	bvs1_ = opengv::bearingVectors_t(bvs1.begin(), bvs1.end());
	bvs2_ = opengv::bearingVectors_t(bvs2.begin(), bvs2.end());
	opengv::relative_pose::CentralRelativeAdapter adapter(bvs1_, bvs2_);
	opengv::sac::Ransac<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;
	std::shared_ptr<opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem> absposeproblem_ptr(
		new opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem(
			adapter,
			opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::SEVENPT));
	ransac.sac_model_ = absposeproblem_ptr;
	ransac.threshold_ = threshold;
	ransac.max_iterations_ = max_itr;
	ransac.computeModel();
	double optPara[6];

	cvl_toolkit::conversion::R2axisRot(ransac.model_coefficients_.block(0, 0, 3, 3), mot.x, mot.y, mot.z);
	mot.x = ransac.model_coefficients_(0, 3);
	mot.y = ransac.model_coefficients_(1, 3);
	mot.z = ransac.model_coefficients_(2, 3);
}


std::vector<int> nonLinearAbsSolving(std::vector<Eigen::Vector3d> points3d, std::vector<Eigen::Vector3d> bearingVect, cvl_toolkit::_6dof& motion, double tolerance) {
	double optimizedPara[6] = { motion.rx,motion.ry,motion.rz,motion.x,motion.y,motion.z };
	int itr = 0;
	double tolerance_ = tolerance;
	std::vector<int> inlier;
	while (itr < 150) {
		double sigma2 = 0;
		int inl = 0;
		ceres::Problem problem;

		for (int i = 0; i < points3d.size(); i++) {
			projectionCostFunc* p = new projectionCostFunc(points3d.at(i), bearingVect.at(i));
			double residual[1];
			p->operator()(optimizedPara, residual);
			if (residual[0] < tolerance_) {
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<projectionCostFunc, ceres::CENTRAL, 1, 6>(
					p);
				problem.AddResidualBlock(c, new ceres::HuberLoss(tolerance_ / 3), optimizedPara);
				inl++;
				double dev = residual[0] * residual[0] / points3d.at(i).norm();
				sigma2 = (inl*(sigma2)+dev) / (inl + 1.0);
				//cout<<cameraIdMat.coeff(i,j)<<endl;
				inlier.push_back(i);
			}
			else {
				delete p;
			}
		}
		ceres::Solver::Options options;
		options.max_num_iterations = 1e4;
		options.function_tolerance = 1e-7;
		options.parameter_tolerance = 1e-7;
		options.linear_solver_type = ceres::DENSE_QR;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (itr >= 1 && tolerance_ < sin(0.1)) {
			std::cout << inl << std::endl;
			std::cout << itr << std::endl;
			break;
		}
		//break;
		inlier.clear();
		tolerance_ = sqrt(sigma2) * 2;
		itr++;
	}
	motion.rx = optimizedPara[0];
	motion.ry = optimizedPara[1];
	motion.rz = optimizedPara[2];
	motion.x = optimizedPara[3];
	motion.y = optimizedPara[4];
	motion.z = optimizedPara[5];
	return inlier;
}

cvl_toolkit::_6dof absoluteTransRansac(opengv::bearingVectors_t bearingVectors, opengv::points_t points, Eigen::Matrix3d R) {
	//_6dof absolutePoseRansac(std::vector<Vector3d> bearingVectors,std::vector<Vector3d> points){
	opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
	//ransac
	opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
	std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
		new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
			adapter,
			opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::TWOPT));
	//Eigen::Matrix3d rot;
	adapter.setR(R);
	ransac.sac_model_ = absposeproblem_ptr;
	ransac.threshold_ = 1.0 - cos(atan(sqrt(2.0)*0.5 / 400.0));//reprojection error(sqrt(2.0)+0.5)pixel / focal length;
															   //ransac.threshold_ = 7e-7;
	ransac.max_iterations_ = 250;
	//cout<<ransac.probability_<<endl;
	ransac.computeModel();
	cvl_toolkit::_6dof optimizedPara;
	//Eigen::Matrix3d cpRot = ransac.model_coefficients_.block(0, 0, 3, 3);
	cvl_toolkit::conversion::R2axisRot(R, optimizedPara.rx, optimizedPara.ry, optimizedPara.rz);
	optimizedPara.x = ransac.model_coefficients_(0, 3);
	optimizedPara.y = ransac.model_coefficients_(1, 3);
	optimizedPara.z = ransac.model_coefficients_(2, 3);
	return optimizedPara;
}

std::vector<int>  nonLinearAbsTransSolving(std::vector<Eigen::Vector3d> points3d, std::vector<Eigen::Vector3d> bearingVect, cvl_toolkit::_6dof& motion, double tolerance) {
	double optimizedPara[3] = { motion.x,motion.y,motion.z };
	double fixedPara[3] = { motion.rx, motion.ry, motion.rz };

	int itr = 0;
	double tolerance_ = tolerance;

	std::vector<int> inlier;
	std::stringstream ss;
	while (itr < 150) {
		double sigma2 = 0;
		int inl = 0;
		ceres::Problem problem;

		for (int i = 0; i < points3d.size(); i++) {
			projectionCostFunc_T* p = new projectionCostFunc_T(points3d.at(i), bearingVect.at(i), fixedPara);
			double residual[1];
			p->operator()(optimizedPara, residual);
			if (residual[0] < tolerance_) {
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<projectionCostFunc_T, ceres::CENTRAL, 1, 3>(
					p);
				problem.AddResidualBlock(c, new ceres::HuberLoss(tolerance_ / 3), optimizedPara);
				inl++;
				double dev = residual[0] * residual[0] / points3d.at(i).norm();
				sigma2 = (inl*(sigma2)+dev) / (inl + 1.0);

				inlier.push_back(i);
			}
			else {
				delete p;
			}
		}
		ceres::Solver::Options options;
		options.max_num_iterations = 1e5;
		options.function_tolerance = 1e-8;
		options.parameter_tolerance = 1e-8;
		options.linear_solver_type = ceres::DENSE_QR;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		ss << itr << "," << inl << "," << sigma2;
		if (itr >= 1 && tolerance_ < sin(0.1)) {
			std::cout << inl << std::endl;
			std::cout << itr << std::endl;
			break;
		}
		//break;
		inlier.clear();
		tolerance_ = sqrt(sigma2) * 2;
		itr++;
	}
	//dataDumper::dumper::dumpTextLn("log.txt", ss.str());
	motion.x = optimizedPara[0];
	motion.y = optimizedPara[1];
	motion.z = optimizedPara[2];
	return inlier;
}



std::vector<Eigen::Matrix4d> initCameraMotionComputation(std::vector<cv::Mat> imgs, std::vector<int> motionList, std::vector<cv::Mat>imgsg, camera* pc, cv::Mat mask) {
	std::vector<Eigen::Matrix4d> cameraMotion;
	//key point detection
	std::vector<cv::Mat> descriptors;
	int motionNumber = motionList.size() / 2;
	std::vector<bool> included(imgs.size(), false);
	for (int i = 0; i < motionList.size(); i++) {
		included.at(motionList.at(i)) = true;
	}
	auto algorithm = cv::AKAZE::create();


	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
	std::vector<cv::DMatch> match12, match13, match34, match12_, match13_, match_;

	std::vector<std::vector<cv::KeyPoint>> keypoints;
	for (int i = 0; i < imgs.size(); i++) {
		std::cout << "image " << i << "keypoint detection" << std::endl;
		std::vector<cv::KeyPoint> keypoint;
		cv::Mat descriptor;
		if (included.at(i)) {

			cv::Mat miniim;
			//if (!bPers) {
			//	cv::resize(imgs.at(i), miniim, cv::Size(1000, 500));
			//}
			//else {
			miniim = imgs.at(i);
			//}
			if (mask.cols == 0) {
				algorithm->detect(miniim, keypoint);
				algorithm->compute(miniim, keypoint, descriptor);
			}
			else {
				algorithm->detect(miniim, keypoint,mask);
				algorithm->compute(miniim, keypoint, descriptor);

			}
		}
		keypoints.push_back(keypoint);
		descriptors.push_back(descriptor);
	}




	for (int i = 0; i < motionNumber; i++) {
		int mbase = motionList.at(i * 2);
		int mdst = motionList.at(i * 2 + 1);
		matcher->match(descriptors.at(mbase), descriptors.at(mdst), match12);
		std::vector<Eigen::Vector3d> bearingVectors1, bearingVectors2;
		opengv::bearingVector_t bv;

		for (int j = 0; j < match12.size(); j++) {
			double ix = keypoints.at(mbase).at(match12.at(j).queryIdx).pt.x;
			double iy = keypoints.at(mbase).at(match12.at(j).queryIdx).pt.y;
	
			Eigen::Vector3d v; v << ix, iy, 1;
			bv = pc->reverse_projection(v);
	
			bearingVectors1.push_back(bv);
			ix = keypoints.at(mdst).at(match12.at(j).trainIdx).pt.x;
			iy = keypoints.at(mdst).at(match12.at(j).trainIdx).pt.y;
			v << ix, iy, 1;
			bv = pc->reverse_projection(v);
	
			bearingVectors2.push_back(bv);
		}
		cvl_toolkit::_6dof mot;
		cameraPoseEst_lin(bearingVectors1, bearingVectors2, mot, sin(1.0e-2), 10000);
		cameraPoseRelEst_non_lin(bearingVectors1, bearingVectors2, mot, 0.1);




		//bearingVectors1.clear();
		//bearingVectors2.clear();

		//std::vector<uchar> status;
		//std::vector<float> err;
		//std::vector<cv::Point2f> src, src_, dst;
		//cv::Mat img1g = imgsg.at(mbase);
		//cv::Mat img2g = imgsg.at(mdst);
		//double minDistance = img1g.cols / 1000.0;
		//cv::goodFeaturesToTrack(img1g, src, 0, 1e-12, minDistance, cv::noArray(), 3, true);

		//cv::calcOpticalFlowPyrLK(img1g, img2g, src, dst, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));//, cv::OPTFLOW_USE_INITIAL_FLOW);
		//int height = img1g.size().height;
		//for (int j = 0; j < src.size(); j++) {
		//	if (status.at(j) != '\1' || err.at(j) > 30)continue;
		//	double ix = src.at(j).x;
		//	double iy = src.at(j).y;
		//	Eigen::Vector3d v; v << ix, iy, 1;
		//	bv = pc->reverse_projection(v);
		//	bearingVectors1.push_back(bv);
		//	ix = dst.at(j).x;
		//	iy = dst.at(j).y;

		//	v << ix, iy, 1;
		//	bv = pc->reverse_projection(v);
		//	bearingVectors2.push_back(bv);
		//}
		//cameraPoseEst_lin(bearingVectors1, bearingVectors2, mot, sin(1.0e-2), 10000);
		//cameraPoseRelEst_non_lin(bearingVectors1, bearingVectors2, mot, 0.05);

		Eigen::Matrix4d estA = cvl_toolkit::conversion::_6dof2m(mot);
		cameraMotion.push_back(estA);
		std::cout << estA << std::endl;
	}
	return cameraMotion;
}


Eigen::Matrix4d calibrationFromMotion(std::vector<Eigen::Matrix4d>& motA, std::vector<Eigen::Matrix4d> motB) {
	//Rotation solver
	Eigen::MatrixXd KA(3, motA.size()), KB(3, motA.size());
	std::cout << "axis of rotation Eigen::Matrix" << std::endl;


	struct rotErrCostFunc {
	public:
		rotErrCostFunc(Eigen::Matrix3d& RA_, Eigen::Matrix3d& RB_)
		{
			RA = RA_;
			RB = RB_;
			double angle;
			Eigen::Vector3d axis;
			Eigen::Matrix3d rot = RA;
			cvl_toolkit::conversion::mat2axis_angle(rot, axis, angle);
			weight = sqrt(1 - cos(angle));
		};
		bool operator()(const double* parameters, double* residual) const {
			double rx = parameters[0];
			double ry = parameters[1];
			double rz = parameters[2];

			Eigen::Matrix3d R = cvl_toolkit::conversion::axisRot2R(rx, ry, rz);
			Eigen::Matrix3d err = Eigen::Matrix3d::Identity() - RA * R * (R * RB).inverse();

			residual[0] = weight * err.norm();//eyeDirec.cross(plot).norm();
			return true;
		}
	private:
		Eigen::Matrix3d RA;
		Eigen::Matrix3d RB;
		double weight;
	};


	double opt[3];
	ceres::Problem problem;
	for (int i = 0; i < motA.size(); i++) {
		double angle1, angle2;
		std::cout << motA.at(i) << std::endl;
		std::cout << motB.at(i) << std::endl;
		Eigen::Vector3d pepax; cvl_toolkit::conversion::mat2axis_angle(motA.at(i).block(0, 0, 3, 3), pepax, angle1);
		Eigen::Vector3d holax; cvl_toolkit::conversion::mat2axis_angle(motB.at(i).block(0, 0, 3, 3), holax, angle2);
		KA.col(i) = pepax;
		KB.col(i) = holax;
		Eigen::Matrix3d RA = motA.at(i).block(0, 0, 3, 3);
		Eigen::Matrix3d RB = motB.at(i).block(0, 0, 3, 3);
		axisAlignCostFunc* p = new axisAlignCostFunc(RA, RB);
		ceres::CostFunction* c = new ceres::NumericDiffCostFunction<axisAlignCostFunc, ceres::CENTRAL, 1, 3>(
			p);
		problem.AddResidualBlock(c, new ceres::HuberLoss(1.0e-2), opt);
	}
	std::cout << "KA,KB" << std::endl;
	std::cout << KA << std::endl;
	std::cout << KB << std::endl;
	Eigen::MatrixXd KBKA = KB * KA.transpose();
	//calc rotation
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(KBKA, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d Hm;
	Eigen::Matrix3d uvt = svd.matrixU()*svd.matrixV().transpose();
	Hm << 1, 0, 0,
		0, 1, 0,
		0, 0, uvt.determinant();
	Eigen::Matrix3d R = svd.matrixV()*Hm*svd.matrixU().transpose();
	cvl_toolkit::conversion::R2axisRot(R, opt[0], opt[1], opt[2]);
	ceres::Solver::Options options;
	options.max_num_iterations = 1e4;
	options.function_tolerance = 1e-6;
	options.parameter_tolerance = 1e-6;
	options.linear_solver_type = ceres::DENSE_QR;
	ceres::Solver::Summary summary;
	R = cvl_toolkit::conversion::axisRot2R(opt[0], opt[1], opt[2]);
	std::cout << KA - R * KB << std::endl;
	ceres::Solve(options, &problem, &summary);
	R = cvl_toolkit::conversion::axisRot2R(opt[0], opt[1], opt[2]);
	std::cout << KA - R * KB << std::endl;


	//solve least square problem
	Eigen::MatrixXd A(motA.size() * 3, 3 + motA.size());
	A.setZero();
	Eigen::VectorXd B(motA.size() * 3);
	for (int i = 0; i < motA.size(); i++) {
		Eigen::Vector3d tpep, thol;
		tpep << motA.at(i)(0, 3), motA.at(i)(1, 3), motA.at(i)(2, 3);
		tpep = tpep.normalized();
		thol << motB.at(i)(0, 3), motB.at(i)(1, 3), motB.at(i)(2, 3);
		Eigen::Vector3d rightt = -R * thol;
		Eigen::Matrix3d leftm = Eigen::Matrix3d::Identity() - motA.at(i).block(0, 0, 3, 3);
		A.block(i * 3, 0, 3, 3) = leftm;
		A.block(i * 3, 3 + i, 3, 1) = -tpep;
		B(i * 3) = rightt(0);
		B(i * 3 + 1) = rightt(1);
		B(i * 3 + 2) = rightt(2);
	}
	Eigen::VectorXd ans(3 + motA.size());
	ans = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);
	std::cout << "x,y,z,t1,t2,t3..." << std::endl;
	std::cout << ans.transpose() << std::endl;
	for (int i = 0; i < motA.size(); i++) {
		motA.at(i).block(0, 3, 3, 1) = ans(i + 3) *motA.at(i).block(0, 3, 3, 1).normalized();
	}
	Eigen::Vector3d t; t << ans(0), ans(1), ans(2);


	Eigen::Matrix4d ret; ret.block(0, 0, 3, 3) = R;
	ret.block(0, 3, 3, 1) = t;
	ret.block(3, 0, 1, 4) << 0, 0, 0, 1;

	return ret;
}



std::vector<Eigen::Matrix4d> cameraMotionComputationWithDepth(std::vector<cv::Mat> imgs_gray, std::vector<Eigen::Matrix4d>& initCamMotion, 
	std::vector<cvl_toolkit::plyObject> po_s, std::vector<Eigen::Matrix4d> lidarPos, std::vector<int> motionList, Eigen::Matrix4d extParam, double thres, camera* pc, cv::Mat mask) {
	std::vector<Eigen::Matrix4d> retMotion;
	int motionNumber = motionList.size() / 2;
	opengv::bearingVector_t bv;
	double ret[8];
	Eigen::Matrix3d extR = extParam.block(0, 0, 3, 3);
	Eigen::Vector3d extt = extParam.block(0, 3, 3, 1);
	for (int imgid = 0; imgid < motionNumber; imgid++) {
		//makeCorresponds
		std::vector<uchar> status;
		std::vector<float> err;
		int mbase = motionList.at(imgid * 2);
		int mdst = motionList.at(imgid * 2 + 1);
		Eigen::Matrix3d Ra = initCamMotion.at(imgid).block(0, 0, 3, 3);
		Eigen::Vector3d ta = initCamMotion.at(imgid).block(0, 3, 3, 1);
		std::vector<cv::Point2f> src, dst;
		cv::Mat img1g = imgs_gray.at(mbase);
		cv::Mat img2g = imgs_gray.at(mdst);

		//projection
		std::vector<Eigen::Vector3d> bearingVectors1, bearingVectors2, bearingVectors3, bearingVectors4;
		std::vector<Eigen::Vector3d> bearingVectors1c, bearingVectors2c, bearingVectorsBase, bearingVectorsBaseback;
		std::vector<cv::Point2f> dst2;
		std::vector<cv::Point2f> srcc, dstc;
		cvl_toolkit::plyObject lidarpts = po_s.at(mbase);
		float*vps = lidarpts.getVertecesPointer();
		Eigen::Matrix4d mrev = lidarPos.at(mbase).inverse();
		Eigen::Matrix4d mrev2 = lidarPos.at(mdst).inverse();
		std::cout << mrev << std::endl;
		unsigned char* maskptr = (unsigned char*)mask.data;
		for (int i = 0; i < lidarpts.getVertexNumber(); i++) {
			Eigen::Vector3d pt, pt2, pt3;
			pt << vps[i * 3], vps[i * 3 + 1], vps[i * 3 + 2];
			pt3 = extR * (mrev2.block(0, 0, 3, 3)*pt + mrev2.block(0, 3, 3, 1)) + extt;//(lidar(aligned, global))->lidar2 local->camera2
			pt = extR * (mrev.block(0, 0, 3, 3)*pt + mrev.block(0, 3, 3, 1)) + extt; //(lidar(aligned, global))->lidar1 local->camera cooridate
			if (pt.norm() < 0.5)continue;

			//projection point computation
			double ix, iy;
			cvl_toolkit::conversion::panorama_bearing2pix(pt, img1g.cols, img1g.rows, ix, iy);
			cv::Point2f p(ix, iy);
			if (iy <= 0 || ix <= 0 || iy > img1g.rows - 1 || ix > img1g.cols - 1)continue;
			//mask
			if (mask.cols > 0) {
				if (maskptr[(int)ix+ (int)iy * img1g.cols] == 0)continue;
			}
			bearingVectors1c.push_back(pt);
			pt2 = (Ra.inverse()*(pt - ta));
			bearingVectors2c.push_back(pt3);
			srcc.push_back(p);

			//initial point conputation for tracking
			pt3 = pt3;
			cvl_toolkit::conversion::panorama_bearing2pix(pt3, img1g.cols, img1g.rows, ix, iy);
			cv::Point2f p3(ix, iy);
			dstc.push_back(p3);//lidar1->lidar2->camera2 
		}

		std::vector<int> selectedIdx;
		cvl_toolkit::goodFeatureToTrack_onProjection(img1g, srcc, selectedIdx, 3.0, 0, 0.0);
		for (int sIdx = 0; sIdx < selectedIdx.size(); sIdx++) {
			src.push_back(srcc.at(selectedIdx.at(sIdx)));
			bearingVectors1.push_back(bearingVectors1c.at(selectedIdx.at(sIdx)));
			bearingVectors3.push_back(bearingVectors2c.at(selectedIdx.at(sIdx)));
			dst.push_back(dstc.at(selectedIdx.at(sIdx)));
		}
		//cv::calcOpticalFlowPyrLK(img1g, img2g, src, dst, status, err, cv::Size(21, 21), 0);
		//cv::Mat disp2; img1g.copyTo(disp2);
		//for (int didx = 0; didx < src.size(); didx++) {
		//	cv::circle(disp2, src.at(didx), 2, cv::Scalar(255));
		//}
		//cv::imshow("neko", disp2); cv::waitKey();

		//cv::Mat disp; img2g.copyTo(disp);
		//for (int didx = 0; didx < dst.size(); didx++) {
		//	cv::circle(disp,dst.at(didx),2,cv::Scalar(255));
		//}
		//cv::imshow("neko", disp); cv::waitKey();
		cv::calcOpticalFlowPyrLK(img1g, img2g, src, dst, status, err, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

		for (int idx = 0; idx < dst.size(); idx++) {
			Eigen::Vector3d bv2;
			cvl_toolkit::conversion::panorama_pix2bearing(dst.at(idx).x, dst.at(idx).y, img2g.size().width, img2g.size().height, bv2);
			bearingVectors2.push_back(bv2);
		}

		//initial
		cvl_toolkit::_6dof mot = cvl_toolkit::conversion::m2_6dof(initCamMotion.at(imgid));
		double thres_ = thres;
		if (thres_ == 0) {
			opengv::bearingVectors_t bearingVectors_t(bearingVectors2.begin(), bearingVectors2.end());
			opengv::points_t points_t(bearingVectors1.begin(), bearingVectors1.end());
			Eigen::Matrix3d rot = cvl_toolkit::conversion::axisRot2R(mot.rx, mot.ry, mot.rz);
			mot = absoluteTransRansac(bearingVectors_t, points_t, rot);
			std::cout << "check" << std::endl;
			thres_ = 0.13;
		}
		std::vector<int> inlier;
		inlier = nonLinearAbsSolving(bearingVectors1, bearingVectors2, mot, thres_);
		//		inlier = nonLinearAbsTransSolving(bearingVectors1, bearingVectors2, mot, thres_);

		//if (thres == 0) {
		//	std::stringstream ss;
		//	ss << "check" << mbase;// << ".ply";
		//	cv::Mat img1d_;
		//	cv::cvtColor(img1g, img1d_, CV_GRAY2BGR);
		//	for (int j = 0; j < inlier.size(); j++) {
		//		cv::circle(img1d_, src_.at(originalId.at(inlier.at(j))), 2, cv::Scalar(255, 0, 0, 255));
		//		cv::circle(img1d_, dst.at(originalId.at(inlier.at(j))), 2, cv::Scalar(0, 0, 255, 255));
		//		cv::line(img1d_, src_.at(originalId.at(inlier.at(j))), dst.at(originalId.at(inlier.at(j))), cv::Scalar(0, 0, 255, 255));
		//	}
		//	//cv::imwrite(dataDumper::dumper::getPrefix() + ss.str() + ".png", img1d_);
		//}

		retMotion.push_back(cvl_toolkit::conversion::_6dof2m(mot));
	}
	return retMotion;
}




Eigen::Matrix4d calibrationFromMotion2(std::vector<Eigen::Matrix4d> motA, std::vector<Eigen::Matrix4d> motB) {
	//Rotation solver
	Eigen::MatrixXd KA(3, motA.size()), KB(3, motA.size());
	std::cout << "axis of rotation Eigen::Matrix" << std::endl;

	double opt[3];
	ceres::Problem problem;
	for (int i = 0; i < motA.size(); i++) {
		double angle1, angle2;
		std::cout << motA.at(i) << std::endl;
		std::cout << motB.at(i) << std::endl;
		Eigen::Vector3d pepax; cvl_toolkit::conversion::mat2axis_angle(motA.at(i).block(0, 0, 3, 3), pepax, angle1);
		Eigen::Vector3d holax; cvl_toolkit::conversion::mat2axis_angle(motB.at(i).block(0, 0, 3, 3), holax, angle2);
		KA.col(i) = pepax;
		KB.col(i) = holax;
		Eigen::Matrix3d RA = motA.at(i).block(0, 0, 3, 3);
		Eigen::Matrix3d RB = motB.at(i).block(0, 0, 3, 3);
		axisAlignCostFunc* p = new axisAlignCostFunc(RA, RB);
		ceres::CostFunction* c = new ceres::NumericDiffCostFunction<axisAlignCostFunc, ceres::CENTRAL, 1, 3>(
			p);
		problem.AddResidualBlock(c, new ceres::HuberLoss(1.0e-3), opt);

	}
	std::cout << "KA,KB" << std::endl;
	std::cout << KA << std::endl;
	std::cout << KB << std::endl;
	Eigen::MatrixXd KBKA = KB * KA.transpose();
	//calc rotation
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(KBKA, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d Hm;
	Eigen::Matrix3d uvt = svd.matrixU()*svd.matrixV().transpose();
	Hm << 1, 0, 0,
		0, 1, 0,
		0, 0, uvt.determinant();
	Eigen::Matrix3d R = svd.matrixV()*Hm*svd.matrixU().transpose();
	cvl_toolkit::conversion::R2axisRot(R, opt[0], opt[1], opt[2]);
	ceres::Solver::Options options;
	options.max_num_iterations = 1e4;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
	options.linear_solver_type = ceres::DENSE_QR;
	ceres::Solver::Summary summary;
	R = cvl_toolkit::conversion::axisRot2R(opt[0], opt[1], opt[2]);
	std::cout << KA - R * KB << std::endl;
	ceres::Solve(options, &problem, &summary);
	R = cvl_toolkit::conversion::axisRot2R(opt[0], opt[1], opt[2]);
	std::cout << KA - R * KB << std::endl;


	struct motionAlignCostFunc {
	public:
		motionAlignCostFunc(Eigen::Matrix4d& ma_, Eigen::Matrix4d& mb_, Eigen::Matrix3d& R_)
		{
			ma = ma_;
			mb = mb_;
			R = R_;
			double angle;
			Eigen::Vector3d axis;
			Eigen::Matrix3d rot = mb.block(0, 0, 3, 3);
			cvl_toolkit::conversion::mat2axis_angle(rot, axis, angle);
			weight = angle;//sqrt(1-cos(angle));
		};
		bool operator()(const double* parameters, double* residual) const {
			double x = parameters[0];
			double y = parameters[1];
			double z = parameters[2];

			Eigen::Vector3d t; t << x, y, z;

			Eigen::Vector3d v1 = (ma.block(0, 0, 3, 3)*t + ma.block(0, 3, 3, 1));
			Eigen::Vector3d v2 = (R*mb.block(0, 3, 3, 1) + t);
			Eigen::Vector3d err = v1 - v2;
			residual[0] = err(0);//eyeDirec.cross(plot).norm();
			residual[1] = err(1);
			residual[2] = err(2);
			return true;
		}
		double weight;
	private:
		Eigen::Matrix4d ma, mb;
		Eigen::Matrix3d R;


	};
	//	double opt[3];
	ceres::Problem problem2;


	//solve least square problem
	Eigen::MatrixXd A(motA.size() * 3, 3);
	A.setZero();
	Eigen::VectorXd B(motA.size() * 3);
	for (int i = 0; i < motA.size(); i++) {
		Eigen::Vector3d tpep, thol;
		tpep << motA.at(i)(0, 3), motA.at(i)(1, 3), motA.at(i)(2, 3);
		thol << motB.at(i)(0, 3), motB.at(i)(1, 3), motB.at(i)(2, 3);
		Eigen::Vector3d rightt = tpep - R * thol;
		Eigen::Matrix3d leftm = Eigen::Matrix3d::Identity() - motA.at(i).block(0, 0, 3, 3);
		A.block(i * 3, 0, 3, 3) = leftm;
		B(i * 3) = rightt(0);
		B(i * 3 + 1) = rightt(1);
		B(i * 3 + 2) = rightt(2);
	}
	Eigen::Vector3d t = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);

	opt[0] = t(0);
	opt[1] = t(1);
	opt[2] = t(2);
	for (int i = 0; i < motA.size(); i++) {
		motionAlignCostFunc* p = new motionAlignCostFunc(motA.at(i), motB.at(i), R);

		double resid[3];
		p->operator()(opt, resid);

		ceres::CostFunction* c = new ceres::NumericDiffCostFunction<motionAlignCostFunc, ceres::CENTRAL, 3, 3>(
			p);
		problem2.AddResidualBlock(c, new ceres::HuberLoss(1e-3), opt);
		std::cout << i << ":" << resid[0] << "," << resid[1] << "," << resid[2] << std::endl;





	}
	options.max_num_iterations = 1e4;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
	options.linear_solver_type = ceres::DENSE_QR;
	ceres::Solve(options, &problem2, &summary);
	t << opt[0], opt[1], opt[2];

	Eigen::Matrix4d ret; ret.block(0, 0, 3, 3) = R;
	ret.block(0, 3, 3, 1) = t;
	ret.block(3, 0, 1, 4) << 0, 0, 0, 1;

	return ret;
}

