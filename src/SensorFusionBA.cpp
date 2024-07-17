//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#pragma warning(disable:4996)
#include "SensorFusionBA.h"


bool SensorFusionBA::LoadMotion(string motionFile) {
	std::ifstream ifs(motionFile);
	nlohmann::json config;
	ifs >> config;
	ifs.close();

	std::vector<std::vector<double>> org_motion = config["_cameramove"].get<std::vector<std::vector<double>>>();
//	cammot_timestamp = config["_camera_ts"].get <std::vector<double>>();

	for (int i = 0; i < org_motion.size(); i++) {
		double qx = org_motion[i][0];
		double qy = org_motion[i][1];
		double qz = org_motion[i][2];
		double qxyz = qx * qx + qy * qy + qz * qz;
		double qw = qxyz >= 1 ? 0 : sqrt(1 - qxyz);

		if (qw == 0) {
			qx = qx / sqrt(qxyz);
			qy = qy / sqrt(qxyz);
			qz = qz / sqrt(qxyz);
		}
		Eigen::Vector4d q(qx, qy, qz, qw);
		Eigen::Vector3d t(org_motion[i][3], org_motion[i][4], org_motion[i][5]);
		Eigen::Matrix3d R = q2dcm(q);
		Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
		M.block(0, 0, 3, 3) = R;
		M.block(0, 3, 3, 1) = t;

		_6dof m = m2_6dof(M);

		framePos.push_back(m);
	}
	return true;
}

void SensorFusionBA::WriteMotion(std::string file) {
	//output
	std::vector< std::vector<double>> movement_out_l;
	std::vector< std::vector<double>> movement_out_c;
	Eigen::Matrix4d extM = inputs.extCalib;
	//extM.block(0, 0, 3, 3) = input.extPar;
	//extM.block(0, 3, 3, 1) = extt;
	//extM.block(3, 0, 1, 4) << 0, 0, 0, 1;
	std::vector<double> ts_l, ts_c;
	double timeoffset = inputs.psl->getTimeOffset();
	for (int i = 0; i < framePos.size(); i++) {
		std::vector<double> camerapose, lidarpose;
		_6dof mot = framePos.at(i);
		Eigen::Matrix4d camera_mot_m = _6dof2m(mot);
		Eigen::Matrix4d lidar_mot = camera_mot_m * extM;

		Eigen::Matrix3d cRot = camera_mot_m.block(0, 0, 3, 3);
		Eigen::Vector4d qc = dcm2q(cRot);
		if (qc(3) < 0)qc = -qc;
		camerapose.push_back(qc(0));		camerapose.push_back(qc(1));		camerapose.push_back(qc(2));
		camerapose.push_back(camera_mot_m(0, 3));		camerapose.push_back(camera_mot_m(1, 3));		camerapose.push_back(camera_mot_m(2, 3));

		Eigen::Matrix3d lRot = lidar_mot.block(0, 0, 3, 3);
		Eigen::Vector4d ql = dcm2q(lRot);
		if (ql(3) < 0)ql = -ql;
		lidarpose.push_back(ql(0));		lidarpose.push_back(ql(1));		lidarpose.push_back(ql(2));
		lidarpose.push_back(lidar_mot(0, 3));		lidarpose.push_back(lidar_mot(1, 3));		lidarpose.push_back(lidar_mot(2, 3));
		movement_out_l.push_back(lidarpose);
		movement_out_c.push_back(camerapose);

		double timecamera = inputs.imTimeStamp[i];
		ts_c.push_back(timecamera);
		double timelidar = timecamera - timeoffset - 0.05;
		ts_l.push_back(timelidar + 0.05);
	}
	nlohmann::json jout;
	jout["_lidarmove"] = movement_out_l;
	jout["_lidar_ts"] = ts_l;
	jout["_cameramove"] = movement_out_c;
	jout["_camera_ts"] = ts_c;
	std::ofstream ofs(file);
	ofs << std::setw(4) << jout << std::endl;
	ofs.close();
}

void SensorFusionBA::WriteMotionAscii(string file) {
	ofstream ofs(file, ios::binary);
	__int64 datanum = framePos.size();
	ofs << datanum << endl;
	//ofs.write((char*)&datanum, sizeof(__int64));
	for (int i = 0;i < datanum;i++) {
		_6dof dof = framePos.at(i);
		Matrix4d c_mot = _6dof2m(dof), s_mot;
		s_mot = inputs.extCalib.inverse()*c_mot*inputs.extCalib;
		dof = m2_6dof(s_mot);

		double curtime = inputs.imTimeStamp[i];
		ofs << curtime << "," << dof << endl;
	}
	ofs.close();
}

void SensorFusionBA::BatchBA(int baseFrame,int frameNum, int skipNum) {
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(baseFrame));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g,cv::COLOR_RGB2GRAY);

	//stage 1, image-image correspondence
	
	vector<cv::Point2f> originP,nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;
	vector<vector<cv::Point2f>> origin3P(frameNum+1);
	vector <vector<Vector3d>> originW3p(frameNum + 1);//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	double startTime = inputs.imTimeStamp[baseFrame]-1/32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	while(inputs.psl->getNextPointData(dat)){
		Vector3d p,pt;p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t=dat[4];
		if (t > inputs.imTimeStamp[baseFrame + framecnt*skipNum] + 1 / 32.0) {
			

			if (framecnt < frameNum) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[baseFrame + framecnt * skipNum] - 1 / 32.0;
				inputs.psl->seekByTime(startTime_);
				free(dat);
				continue;
			}
			else {
				free(dat);
				break;
			}
			
		}
		if (ref < 1e-4) {
			free(dat);
			continue;
		}
		cv::Scalar color;color = cv::Scalar(255, 0, 0);
		if (t < inputs.imTimeStamp[baseFrame + framecnt * skipNum]) {
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum - 1));
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0,3,3,1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum -1];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1-rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else{
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum + 1));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum +1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		cv::circle(image1, projp, 0, color);
		origin3P.at(framecnt).push_back(projp);
		originW3p.at(framecnt).push_back(pt);
		free(dat);
	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	cv::imwrite(ss.str(), image1);
	double** optParas = (double**)malloc(sizeof(double*)*frameNum);
	for (int i = 0;i < frameNum;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(baseFrame + (i + 1) * skipNum));
		optParas[i] = (double*)malloc(sizeof(double)*6);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParas[i][0] = diffMot.rx;
		optParas[i][1] = diffMot.ry;
		optParas[i][2] = diffMot.rz;
		optParas[i][3] = diffMot.x;
		optParas[i][4] = diffMot.y;
		optParas[i][5] = diffMot.z;
	}

	ceres::Problem problem;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < frameNum;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(baseFrame + (i + 1)*skipNum));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		//2d-2d
		images_g.push_back(image2g);
	}
		//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
		//2d-3d
				
		//point selection
	for (int i = 0;i < frameNum+1;i++) {
		vector<int> selected;
		goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, 5, 0);
		vector<cv::Point2f> fps;
		vector<Vector3d> wps;
		for (auto itr : selected) {
			cv::Point2f pt = origin3P.at(i).at(itr);
			Vector3d pt3w = originW3p.at(i).at(itr);
			fps.push_back(pt);
			wps.push_back(pt3w);
		}
		origin3P.at(i) = vector<cv::Point2f>(fps);
		originW3p.at(i) = vector<Vector3d>(wps);
	}
	double weight3d[1];
	weight3d[0] = 1.0;
	for (int i = 0;i < frameNum + 1;i++) {
		//forward
		for (int j = i+1;j < frameNum + 1;j++) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1') {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);
					
					if (i == 0) {
						BADepthKnownCostFunc_anckor *f = new BADepthKnownCostFunc_anckor(originW3p.at(i).at(k), nxt3v,weight3d);
						double residual[3];
						f->operator()(optParas[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_anckor, ceres::CENTRAL, 3,  6>(f);
							problem.AddResidualBlock(c, NULL, optParas[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc *f = new BADepthKnownCostFunc(originW3p.at(i).at(k), nxt3v, weight3d);
						double residual[3];
						f->operator()(optParas[i-1], optParas[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc, ceres::CENTRAL, 3, 6, 6>(f);
							problem.AddResidualBlock(c, NULL, optParas[i - 1],optParas[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
				}
			}
		}
		//backward
		for (int j = i - 1;j >= 0;j--) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1') {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (j == 0) {
						BADepthKnownCostFunc_anckor_rev *f = new BADepthKnownCostFunc_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d);
						double residual[3];
						f->operator()(optParas[i - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_anckor_rev, ceres::CENTRAL, 3, 6>(f);
							problem.AddResidualBlock(c, NULL, optParas[i - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc *f = new BADepthKnownCostFunc(originW3p.at(i).at(k), nxt3v, weight3d);
						double residual[3];
						f->operator()(optParas[i - 1], optParas[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc, ceres::CENTRAL, 3, 6, 6>(f);
							problem.AddResidualBlock(c, NULL, optParas[i - 1], optParas[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
				}
			}
		}
	}
	

	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g,optParas);
	double** sfm_point=(double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;
	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParas[itr2.first - 1][0], optParas[itr2.first - 1][1], optParas[itr2.first - 1][2]);
						c1 << optParas[itr2.first - 1][3], optParas[itr2.first - 1][4], optParas[itr2.first - 1][5];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else {
					R2 = axisRot2R(optParas[itr2.first - 1][0], optParas[itr2.first - 1][1], optParas[itr2.first - 1][2]);
					c2 << optParas[itr2.first - 1][3], optParas[itr2.first - 1][4], optParas[itr2.first - 1][5];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm()>0.005) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };
			
		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm_b.ply");

	}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	for (auto itr : f_pts) {
	/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
		sfm_point[idx][0] = 1;
		sfm_point[idx][1] = 1;
		sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (itr2.first == 0) {
				BaFeatureProjCostFunction_Ancker* f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				double residual[1];

				f->operator()(sfm_point[idx], residual);
				if (residual[0] * residual[0] < 0.005) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
					problem.AddResidualBlock(c, NULL, sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}
			else {
				BaFeatureProjCostFunction* f = new BaFeatureProjCostFunction(nxt3v, weight2d);

				double residual[1];

				f->operator()(optParas[itr2.first-1], sfm_point[idx], residual);
				if (residual[0] * residual[0] < 0.005) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction, ceres::CENTRAL, 3, 6, 3>(f);
					problem.AddResidualBlock(c, NULL, optParas[itr2.first-1], sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}
			
		}
		idx++;
	}
	cout << "3d:" << count3d << "   2d:" << count << endl;
	weight3d[0] = 1.0 / sqrt(count3d);
	weight2d[0] = 1.0 / sqrt(count);
	ceres::Solver::Options options;
	options.max_num_iterations = 1e2;
	options.function_tolerance = 1e-6;
	options.parameter_tolerance = 1e-6;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	Matrix4d diffMatrix=Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx*3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i*3] = sfm_point[i][0];
			vtx[i * 3+1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;
		
		bp.setVertecesPointer(vtx,idx);
		bp.setReflectancePointer(rfl,idx);
		bp.writePlyFileAuto("sfm.ply");
	
	}

	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < frameNum) {
			if (i == baseFrame + (updateIdx)*skipNum) {
				diffMatrix = diffMatrixLocal* diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin,mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.x = optParas[updateIdx - 1][0];m1.y = optParas[updateIdx - 1][1]; m1.z = optParas[updateIdx - 1][2]; m1.x = optParas[updateIdx - 1][3];;m1.y = optParas[updateIdx - 1][4];m1.z = optParas[updateIdx - 1][5];
	
				}
				//morigin = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum));
				_6dof m2; m2.rx = optParas[updateIdx][0];m2.ry = optParas[updateIdx][1]; m2.rz = optParas[updateIdx][2]; m2.x = optParas[updateIdx][3];;m2.y = optParas[updateIdx][4];m2.z = optParas[updateIdx][5];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum))*mupdate*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum)).inverse();
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - baseFrame) % skipNum) / (double)skipNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0,3,3,1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix =m * baseDiffMatrix;
			if (i == baseFrame + (updateIdx+1)*skipNum-1) {
				updateIdx++;
			}
			//cout << i << endl;
			//cout << baseDiffMatrix << endl << endl;
		}
		_6dof temp = framePos.at(i);
		//cout << i << endl;
		//cout << baseDiffMatrix << endl << endl;
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);
		
		temp = m2_6dof(updated);
		framePos.at(i) = temp;
	}

	for (int i = 0;i < frameNum;i++) {
		free(optParas[i]);
	}
	free(optParas);



}




void SensorFusionBA::BatchBA_scale(int baseFrame, int frameNum, int skipNum) {
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(baseFrame));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;
	vector<vector<cv::Point2f>> origin3P(frameNum + 1);
	vector <vector<Vector3d>> originW3p(frameNum + 1);//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	double startTime = inputs.imTimeStamp[baseFrame] - 1 / 32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	while (inputs.psl->getNextPointData(dat)) {
		Vector3d p, pt;p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t = dat[4];
		if (t > inputs.imTimeStamp[baseFrame + framecnt * skipNum] + 1 / 32.0 || (baseFrame + framecnt * skipNum == framePos.size()-1 && t > inputs.imTimeStamp[baseFrame + framecnt * skipNum])) {


			if (framecnt < frameNum) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[baseFrame + framecnt * skipNum] - 1 / 32.0;
				inputs.psl->seekByTime(startTime_);
				free(dat);
				continue;
			}
			else {
				free(dat);
				break;
			}

		}
		if (ref < 1e-2) {
			free(dat);
			continue;
		}
		if (p.norm() < distLower || p.norm() > distUpper) {
			free(dat);
			continue;
		}
		cv::Scalar color;color = cv::Scalar(255, 0, 0);
		if (t < inputs.imTimeStamp[baseFrame + framecnt * skipNum]) {
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum - 1));
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);

			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum - 1];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1 - rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else {
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum + 1));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum + 1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		cv::circle(image1, projp, 0, color);
		origin3P.at(framecnt).push_back(projp);
		originW3p.at(framecnt).push_back(pt);
		free(dat);
	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*frameNum);
	double** optParasR = (double**)malloc(sizeof(double*)*frameNum);
	double* optScale = (double*)malloc(sizeof(double)*frameNum);
	double optScale_[1] = { 1.0 };
	double scale = 0;
	for (int i = 0;i < frameNum;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(baseFrame + (i + 1) * skipNum));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
	}
	
	ceres::Problem problem_d,problem,problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < frameNum;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(baseFrame + (i + 1)*skipNum));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection
	for (int i = 0;i < frameNum + 1;i++) {
		vector<int> selected;
		goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, 10, 0);
		vector<cv::Point2f> fps;
		vector<Vector3d> wps;
		for (auto itr : selected) {
			cv::Point2f pt = origin3P.at(i).at(itr);
			Vector3d pt3w = originW3p.at(i).at(itr);
			fps.push_back(pt);
			wps.push_back(pt3w);
		}
		origin3P.at(i) = vector<cv::Point2f>(fps);
		originW3p.at(i) = vector<Vector3d>(wps);
	}
	double weight3d[1];
	weight3d[0] = 1.0;
	vector<BADepthKnownCostFunc_scale*> costFncs;
	for (int i = 0;i < frameNum + 1;i++) {
		//forward
		for (int j = i + 1;j < frameNum + 1;j++) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1') {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (i == 0) {
						BADepthKnownCostFunc_direction_anckor *f = new BADepthKnownCostFunc_direction_anckor(originW3p.at(i).at(k), nxt3v , weight3d, optParasR[j-1]);
						double residual[1];
						f->operator()(optParasT[j - 1], residual);
						if (residual[0] * residual[0] < 1e-4) {

							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_direction_anckor, ceres::CENTRAL, 1, 3>(f);

							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[j - 1]);

							count3d++;
						}

						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc_direction *f = new BADepthKnownCostFunc_direction(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i-1], optParasR[j - 1]);
						double residual[1];
						f->operator()(optParasT[i - 1], optParasT[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_direction, ceres::CENTRAL, 1, 3, 3>(f);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}

					int idxs[2];
					idxs[0] = i;
					idxs[1] = j;
					BADepthKnownCostFunc_scale* f = new BADepthKnownCostFunc_scale(originW3p.at(i).at(k), nxt3v, weight3d,idxs,optParasR,optParasT,false);
					double residual[3];
					f->operator()(optScale_,  residual);
					//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
					if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_scale, ceres::CENTRAL, 3, 1>(f);
						problem2.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optScale_);
						costFncs.push_back(f);
					}
					
				}
			}
		}
		//backward
		for (int j = i - 1;j >= 0;j--) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1') {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (j == 0) {
						BADepthKnownCostFunc_direction_anckor_rev *f = new BADepthKnownCostFunc_direction_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d,optParasR[i-1]);
						double residual[1];
						f->operator()(optParasT[i - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0]  < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_direction_anckor_rev, ceres::CENTRAL, 1, 3>(f);
							problem.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optParasT[i - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc_direction *f = new BADepthKnownCostFunc_direction(originW3p.at(i).at(k), nxt3v, weight3d,optParasR[i-1], optParasR[j - 1]);
						double residual[1];
						f->operator()(optParasT[i - 1], optParasT[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0]  < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_direction, ceres::CENTRAL, 1, 3, 3>(f);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					int idxs[2];
					idxs[0] = i;
					idxs[1] = j;
					//cout << i << "," << j << "," << l << endl;
					BADepthKnownCostFunc_scale* f = new BADepthKnownCostFunc_scale(originW3p.at(i).at(k), nxt3v, weight3d, idxs, optParasR, optParasT, true);
					double residual[3];
					f->operator()(optScale_, residual);
					//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
					if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_scale, ceres::CENTRAL, 3,1>(f);
						problem2.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optScale_);
						costFncs.push_back(f);
					}
					
				}
			}
		}
	}
	scaleAnckor* sa_func = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func);
	problem.AddResidualBlock(c_anc, NULL, optParasT[frameNum - 1]);
	scaleAnckor* sa_func2 = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func2);
	problem_d.AddResidualBlock(c_anc2, NULL, optParasT[frameNum - 1]);
	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g, optParasR);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else {
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.005) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm_b.ply");
		bp.release();
	}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (/*itr2.first < 0*/itr2.first == 0) {
				BaFeatureProjCostFunction_Ancker* f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				double residual[3];

				f->operator()(sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.005) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					BaFeatureProjCostFunction_Ancker* f2 = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);
					ceres::CostFunction* c2 = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f2);
					problem.AddResidualBlock(c2, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}
			else {
				BaFeatureProjCostFunction_* f = new BaFeatureProjCostFunction_(nxt3v, weight2d);

				double residual[3];

				f->operator()(optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx], residual);
				if (residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] < 0.005) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_, ceres::CENTRAL, 3, 3, 3, 3>(f);
					problem_d.AddResidualBlock(c,new ceres::CauchyLoss(2e-3), optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
					BaFeatureProjCostFunctionT* f2 = new BaFeatureProjCostFunctionT(nxt3v, weight2d , optParasR[itr2.first - 1]);
					ceres::CostFunction* c2 = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunctionT, ceres::CENTRAL, 3, 3, 3>(f2);
					problem.AddResidualBlock(c2, new ceres::CauchyLoss(2e-3), optParasT[itr2.first - 1], sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}

		}
		idx++;
	}

	cout << "3d:" << count3d << "   2d:" << count << endl;
	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options options;
	options.max_num_iterations = 1e2;
	options.function_tolerance = 1e-6;
	options.parameter_tolerance = 1e-6;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem_d, &summary);
	ceres::Solve(options, &problem, &summary);
	ceres::Solve(options, &problem2, &summary);
	
	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);
	
	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < frameNum) {
			if (i == baseFrame + (updateIdx)*skipNum) {
				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				//morigin = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum));
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0]*mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum))*mupdate*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum)).inverse();
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - baseFrame) % skipNum) / (double)skipNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == baseFrame + (updateIdx + 1)*skipNum - 1) {
				updateIdx++;
			}
			//cout << i << endl;
			//cout << baseDiffMatrix << endl << endl;
		}
		_6dof temp = framePos.at(i);
		//cout << i << endl;
		//cout << baseDiffMatrix << endl << endl;
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		//cout << temp << endl;
		framePos.at(i) = temp;
	}

	for (int i = 0;i < frameNum;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return;

}


void SensorFusionBA::BatchBA_TL(int baseFrame, int frameNum, int skipNum) {
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(baseFrame));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;
	vector<vector<cv::Point2f>> origin3P(frameNum + 1);
	vector <vector<Vector3d>> originW3p(frameNum + 1);//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	double startTime = inputs.imTimeStamp[baseFrame] - 1 / 32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	while (inputs.psl->getNextPointData(dat)) {
		Vector3d p, pt;p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t = dat[4];
		if (t > inputs.imTimeStamp[baseFrame + framecnt * skipNum] + 1 / 32.0 || (baseFrame + framecnt * skipNum == framePos.size() - 1 && t > inputs.imTimeStamp[baseFrame + framecnt * skipNum])) {


			if (framecnt < frameNum) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[baseFrame + framecnt * skipNum] - 1 / 32.0;
				inputs.psl->seekByTime(startTime_);
				free(dat);
				continue;
			}
			else {
				free(dat);
				break;
			}

		}
		if (ref < 1e-2) {
			free(dat);
			continue;
		}
		if (p.norm() < distLower || p.norm() > distUpper) {
			free(dat);
			continue;
		}
		cv::Scalar color;color = cv::Scalar(255, 0, 0);
		if (t < inputs.imTimeStamp[baseFrame + framecnt * skipNum]) {
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum - 1));
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);

			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum - 1];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1 - rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else {
			if (baseFrame + framecnt * skipNum + 1 >= framePos.size()) {
				cout << framecnt << endl;
				cout << skipNum << endl;
				cout << framePos.size() << endl;
			}
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum + 1));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum + 1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		cv::circle(image1, projp, 0, color);
		origin3P.at(framecnt).push_back(projp);
		originW3p.at(framecnt).push_back(pt);
		free(dat);
	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*frameNum);
	double** optParasR = (double**)malloc(sizeof(double*)*frameNum);
	double* optScale = (double*)malloc(sizeof(double)*frameNum);
	double optScale_[1] = { 1.0 };
	double scale = 0;
	for (int i = 0;i < frameNum;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(baseFrame + (i + 1) * skipNum));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
	}

	ceres::Problem problem_d, problem, problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < frameNum;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(baseFrame + (i + 1)*skipNum));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}


	/*
	scaleAnckor* sa_func = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func);
	problem.AddResidualBlock(c_anc, NULL, optParasT[frameNum - 1]);*/
	scaleAnckor* sa_func2 = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func2);
	problem_d.AddResidualBlock(c_anc2, NULL, optParasT[frameNum - 1]);
	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g,optParasR);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else {
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.005) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm_b.ply");
		bp.release();
	}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (/*itr2.first < 0*/itr2.first == 0) {
				BaFeatureProjCostFunction_Ancker* f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				double residual[3];

				f->operator()(sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.01) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					BaFeatureProjCostFunction_Ancker* f2 = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);
					ceres::CostFunction* c2 = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f2);
					//problem.AddResidualBlock(c2, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}
			else {
				BaFeatureProjCostFunction_* f = new BaFeatureProjCostFunction_(nxt3v, weight2d);

				double residual[3];

				f->operator()(optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.01) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_, ceres::CENTRAL, 3, 3, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
					BaFeatureProjCostFunctionT* f2 = new BaFeatureProjCostFunctionT(nxt3v, weight2d, optParasR[itr2.first - 1]);
					ceres::CostFunction* c2 = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunctionT, ceres::CENTRAL, 3, 3, 3>(f2);
					//problem.AddResidualBlock(c2, new ceres::CauchyLoss(2e-3), optParasT[itr2.first - 1], sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}

		}
		idx++;
	}

	double weight3d[1];
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection
	for (int i = 0;i < frameNum + 1;i++) {
		vector<int> selected;
		goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, 10, 0);
		vector<cv::Point2f> fps;
		vector<Vector3d> wps;
		for (auto itr : selected) {
			cv::Point2f pt = origin3P.at(i).at(itr);
			Vector3d pt3w = originW3p.at(i).at(itr);
			fps.push_back(pt);
			wps.push_back(pt3w);
		}
		origin3P.at(i) = vector<cv::Point2f>(fps);
		originW3p.at(i) = vector<Vector3d>(wps);
	}

	vector<BADepthKnownCostFunc_scale*> costFncs;
	for (int i = 0;i < frameNum + 1;i++) {
		//forward
		for (int j = i + 1;j < frameNum + 1;j++) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1') {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (i == 0) {
						BADepthKnownCostFunc_PrjT_anckor *f = new BADepthKnownCostFunc_PrjT_anckor(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1]);
						double residual[3];
						f->operator()(optParasT[j - 1], residual);
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {

							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor, ceres::CENTRAL, 3, 3>(f);

							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[j - 1]);

							count3d++;
						}

						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						double residual[3];
						f->operator()(optParasT[i - 1], optParasT[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}

					//int idxs[2];
					//idxs[0] = i;
					//idxs[1] = j;
					//BADepthKnownCostFunc_scale* f = new BADepthKnownCostFunc_scale(originW3p.at(i).at(k), nxt3v, weight3d, idxs, optParasR, optParasT, false);
					//double residual[3];
					//f->operator()(optScale_, residual);
					////cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
					//if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
					//	ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_scale, ceres::CENTRAL, 3, 1>(f);
					//	problem2.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optScale_);
					//	costFncs.push_back(f);
					//}

				}
			}
		}
		//backward
		for (int j = i - 1;j >= 0;j--) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1') {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (j == 0) {
						BADepthKnownCostFunc_PrjT_anckor_rev *f = new BADepthKnownCostFunc_PrjT_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1]);
						double residual[3];
						f->operator()(optParasT[i - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor_rev, ceres::CENTRAL, 3, 3>(f);
							problem.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optParasT[i - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						double residual[3];
						f->operator()(optParasT[i - 1], optParasT[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					//int idxs[2];
					//idxs[0] = i;
					//idxs[1] = j;
					////cout << i << "," << j << "," << l << endl;
					//BADepthKnownCostFunc_scale* f = new BADepthKnownCostFunc_scale(originW3p.at(i).at(k), nxt3v, weight3d, idxs, optParasR, optParasT, true);
					//double residual[3];
					//f->operator()(optScale_, residual);
					////cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
					//if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
					//	ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_scale, ceres::CENTRAL, 3, 1>(f);
					//	problem2.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optScale_);
					//	costFncs.push_back(f);
					//}

				}
			}
		}
	}

	cout << "3d:" << count3d << "   2d:" << count << endl;

	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options options;
	options.max_num_iterations = 3e2;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
	options.num_threads = 12;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem_d, &summary);
	ceres::Solve(options, &problem, &summary);
	//ceres::Solve(options, &problem2, &summary);

	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);

	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < frameNum) {
			if (i == baseFrame + (updateIdx)*skipNum) {
				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				//morigin = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum));
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0] * mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum))*mupdate*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum)).inverse();
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - baseFrame) % skipNum) / (double)skipNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == baseFrame + (updateIdx + 1)*skipNum - 1) {
				updateIdx++;
			}
			//cout << i << endl;
			//cout << baseDiffMatrix << endl << endl;
		}
		_6dof temp = framePos.at(i);
		//cout << i << endl;
		//cout << baseDiffMatrix << endl << endl;
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		//cout << temp << endl;
		framePos.at(i) = temp;
	}

	for (int i = 0;i < frameNum;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return;

}


void SensorFusionBA::BatchBA_Global(int baseFrame, int frameNum, int skipNum) {
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(baseFrame));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;

	int frame_thres = 80;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;
	vector<vector<cv::Point2f>> origin3P(frameNum + 1);
	vector <vector<Vector3d>> originW3p(frameNum + 1);//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	double startTime = inputs.imTimeStamp[baseFrame] - 1 / 32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	while (inputs.psl->getNextPointData(dat)) {
		Vector3d p, pt;p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t = dat[4];
		if (t > inputs.imTimeStamp[baseFrame + framecnt * skipNum] + 1 / 32.0 || (baseFrame + framecnt * skipNum == framePos.size() - 1 && t > inputs.imTimeStamp[baseFrame + framecnt * skipNum])) {


			if (framecnt < frameNum) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[baseFrame + framecnt * skipNum] - 1 / 32.0;
				inputs.psl->seekByTime(startTime_);
				free(dat);
				continue;
			}
			else {
				free(dat);
				break;
			}

		}
		if (ref < 1e-2) {
			free(dat);
			continue;
		}
		if (p.norm() < distLower || p.norm() > distUpper) {
			free(dat);
			continue;
		}
		cv::Scalar color;color = cv::Scalar(255, 0, 0);
		if (t < inputs.imTimeStamp[baseFrame + framecnt * skipNum]) {
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum - 1));
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);

			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum - 1];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1 - rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else {
			if (baseFrame + framecnt * skipNum + 1 >= framePos.size()) {
				cout << framecnt << endl;
				cout << skipNum << endl;
				cout << framePos.size() << endl;
			}
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum + 1));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum + 1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		cv::circle(image1, projp, 0, color);
		origin3P.at(framecnt).push_back(projp);
		originW3p.at(framecnt).push_back(pt);
		free(dat);
	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*frameNum);
	double** optParasR = (double**)malloc(sizeof(double*)*frameNum);
	double* optScale = (double*)malloc(sizeof(double)*frameNum);
	double optScale_[1] = { 1.0 };
	double scale = 0;
	for (int i = 0;i < frameNum;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(baseFrame + (i + 1) * skipNum));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
	}

	ceres::Problem problem_d, problem, problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < frameNum;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(baseFrame + (i + 1)*skipNum));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}

	scaleAnckor* sa_func2 = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func2);
	problem_d.AddResidualBlock(c_anc2, NULL, optParasT[frameNum - 1]);
	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g, optParasR,skipNum);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;
	

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else {
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.005) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm_b.ply");
		bp.release();
	}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (/*itr2.first < 0*/itr2.first == 0) {
				BaFeatureProjCostFunction_Ancker* f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				double residual[3];

				f->operator()(sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.01) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					BaFeatureProjCostFunction_Ancker* f2 = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);
					ceres::CostFunction* c2 = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f2);
					//problem.AddResidualBlock(c2, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}
			else {
				BaFeatureProjCostFunction_* f = new BaFeatureProjCostFunction_(nxt3v, weight2d);

				double residual[3];

				f->operator()(optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.01) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_, ceres::CENTRAL, 3, 3, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
					BaFeatureProjCostFunctionT* f2 = new BaFeatureProjCostFunctionT(nxt3v, weight2d, optParasR[itr2.first - 1]);
					ceres::CostFunction* c2 = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunctionT, ceres::CENTRAL, 3, 3, 3>(f2);
					//problem.AddResidualBlock(c2, new ceres::CauchyLoss(2e-3), optParasT[itr2.first - 1], sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}

		}
		idx++;
	}

	double weight3d[1];
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection
	int originskip = 10;
	for (int i = 0;i < frameNum + 1;i++) {
		
		if (i % originskip == 0) {
			vector<int> selected;
			goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, 30, 3);
			vector<cv::Point2f> fps;
			vector<Vector3d> wps;
			for (auto itr : selected) {
				cv::Point2f pt = origin3P.at(i).at(itr);
				Vector3d pt3w = originW3p.at(i).at(itr);
				fps.push_back(pt);
				wps.push_back(pt3w);
			}
			origin3P.at(i) = vector<cv::Point2f>(fps);
			originW3p.at(i) = vector<Vector3d>(wps);
		}
		else {
			origin3P.at(i).clear();
			originW3p.at(i).clear();
		}
	}

	vector<BADepthKnownCostFunc_scale*> costFncs;
	for (int i = 0;i < frameNum + 1;i++) {
		if (originW3p.at(i).size() == 0)continue;
		//forward
		for (int j = i + 1;j < min(frameNum + 1,i+frame_thres);j++) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1' && err.at(k) < 15) {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (i == 0) {
						BADepthKnownCostFunc_PrjT_anckor *f = new BADepthKnownCostFunc_PrjT_anckor(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1]);
						double residual[3];
						f->operator()(optParasT[j - 1], residual);
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {

							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor, ceres::CENTRAL, 3, 3>(f);

							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[j - 1]);

							count3d++;
						}

						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						double residual[3];
						f->operator()(optParasT[i - 1], optParasT[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}

				}
			}
		}
		//backward
		for (int j = i - 1;j >= max(0, i - frame_thres);j--) {
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1'&& err.at(k) < 15) {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (j == 0) {
						BADepthKnownCostFunc_PrjT_anckor_rev *f = new BADepthKnownCostFunc_PrjT_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1]);
						double residual[3];
						f->operator()(optParasT[i - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor_rev, ceres::CENTRAL, 3, 3>(f);
							problem.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optParasT[i - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						double residual[3];
						f->operator()(optParasT[i - 1], optParasT[j - 1], residual);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 1e-4) {
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
						}
						else {
							delete f;
						}
					}

				}
			}
		}
	}

	cout << "3d:" << count3d << "   2d:" << count << endl;

	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options options;
	options.max_num_iterations = 1e3;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
	options.num_threads = 12;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem_d, &summary);
	ceres::Solve(options, &problem, &summary);
	//ceres::Solve(options, &problem2, &summary);

	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);

	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < frameNum) {
			if (i == baseFrame + (updateIdx)*skipNum) {
				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0] * mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum))*mupdate*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum)).inverse();
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - baseFrame) % skipNum) / (double)skipNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == baseFrame + (updateIdx + 1)*skipNum - 1) {
				updateIdx++;
			}
		}
		_6dof temp = framePos.at(i);
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		framePos.at(i) = temp;
	}

	for (int i = 0;i < frameNum;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return;

}


string SensorFusionBA::BatchBA_Global_seq(int baseFrame, int frameNum, int skipNum, int overlap) {
	stringstream returnlog;
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(baseFrame));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;

	int frame_thres = 800/skipNum;// threshold for frame 
	int anchor_under = overlap;// anchor
	int originskip = 5/skipNum;
	if (originskip <= 0)originskip = 1;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;
	vector<vector<cv::Point2f>> origin3P(frameNum + 1);
	vector <vector<Vector3d>> originW3p(frameNum + 1);//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	double startTime = inputs.imTimeStamp[baseFrame] - 1 / 32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	while (inputs.psl->getNextPointData(dat)) {
		Vector3d p, pt;p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t = dat[4];
		if (t > inputs.imTimeStamp[baseFrame + framecnt * skipNum] + 1 / 32.0 || (baseFrame + framecnt * skipNum == framePos.size() - 1 && t > inputs.imTimeStamp[baseFrame + framecnt * skipNum])) {


			if (framecnt < frameNum) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[baseFrame + framecnt * skipNum] - 1 / 32.0;
				inputs.psl->seekByTime(startTime_);
				free(dat);
				continue;
			}
			else {
				free(dat);
				break;
			}

		}
		if (ref < 1e-2) {
			free(dat);
			continue;
		}
		if (p.norm() < distLower || p.norm() > distUpper) {
			free(dat);
			continue;
		}
		cv::Scalar color;color = cv::Scalar(255, 0, 0);
		if (t < inputs.imTimeStamp[baseFrame + framecnt * skipNum]) {
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum - 1));
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);

			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum - 1];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1 - rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else {
			if (baseFrame + framecnt * skipNum + 1 >= framePos.size()) {
				cout << framecnt << endl;
				cout << skipNum << endl;
				cout << framePos.size() << endl;
			}
			Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum + 1));
			Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
			double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum + 1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		cv::circle(image1, projp, 0, color);
		origin3P.at(framecnt).push_back(projp);
		originW3p.at(framecnt).push_back(pt);
		free(dat);
	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*frameNum);
	double** optParasR = (double**)malloc(sizeof(double*)*frameNum);
	double* optScale = (double*)malloc(sizeof(double)*frameNum);
	double optScale_[1] = { 1.0 };
	double scale = 0;
	for (int i = 0;i < frameNum;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(baseFrame + (i + 1) * skipNum));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
	}

	ceres::Problem problem_d, problem, problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < frameNum;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(baseFrame + (i + 1)*skipNum));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}

	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g, optParasR, skipNum);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else if(itr2.first > overlap){
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.05) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	//{
	//	float* vtx;
	//	float* rfl;
	//	vtx = (float*)malloc(sizeof(float)*idx * 3);
	//	rfl = (float*)malloc(sizeof(float)*idx);
	//	for (int i = 0;i < idx;i++) {
	//		vtx[i * 3] = sfm_point[i][0];
	//		vtx[i * 3 + 1] = sfm_point[i][1];
	//		vtx[i * 3 + 2] = sfm_point[i][2];
	//		rfl[i] = 0.5;
	//	}
	//	BasicPly bp;

	//	bp.setVertecesPointer(vtx, idx);
	//	bp.setReflectancePointer(rfl, idx);
	//	bp.writePlyFileAuto("sfm_b.ply");
	//	bp.release();
	//}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	vector<int> hist = vector<int>(images_g.size()+1, 0);
	vector<int> hist2 = vector<int>(images_g.size() + 1, 0);
	vector<BaFeatureProjCostFunction_*> cost_2d;
	vector<BaFeatureProjCostFunction_Ancker*> cost_2d_a;
	vector<BADepthKnownCostFunc_PrjT*> cost_3d;
	vector<BADepthKnownCostFunc_PrjT_anckor*> cost_3d_a;
	vector<BADepthKnownCostFunc_PrjT_anckor_rev*> cost_3d_ar;
	double thres2d = 0.05*0.05;
	double thres3d = 0.05*0.05;
	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			hist2[itr2.first] += 1;
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (/*itr2.first < 0*/itr2.first <= anchor_under) {
				double ancm[6];
				
				BaFeatureProjCostFunction_Ancker* f;
				if (itr2.first==0) {
					f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				}
				else {
					ancm[0] = optParasR[itr2.first - 1][0];
					ancm[1] = optParasR[itr2.first - 1][1];
					ancm[2] = optParasR[itr2.first - 1][2];
					ancm[3] = optParasT[itr2.first - 1][0];
					ancm[4] = optParasT[itr2.first - 1][1];
					ancm[5] = optParasT[itr2.first - 1][2];
					f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d, ancm);				
				}

				cost_2d_a.push_back(f);
				f->set(sfm_point[idx]);
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
				f->outlierRejection(thres2d);
				problem_d.AddResidualBlock(c, new ceres::CauchyLoss(1e-3), sfm_point[idx]);
				count++;
				hist[itr2.first] += 1;
			}
			else {
				BaFeatureProjCostFunction_* f = new BaFeatureProjCostFunction_(nxt3v, weight2d);
				cost_2d.push_back(f);
				f->set(optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_, ceres::CENTRAL, 3, 3, 3, 3>(f);
				f->outlierRejection(thres2d);
				problem_d.AddResidualBlock(c, new ceres::CauchyLoss(1e-3), optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
				count++;
				hist[itr2.first] += 1;
			}

		}
		idx++;
	}
	int histidx = 0;
	returnlog << endl<<endl;
	bool setAnchor = false;
	for (auto itr = hist.begin();hist.end() != itr;itr++) {
		returnlog << *itr <<"/"<<hist2.at(histidx)<< endl;
		histidx++;
		if (*itr<5) setAnchor = true;
	}

	if (anchor_under == 0 /*|| setAnchor*/) {
		scaleAnckor* sa_func2 = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
		ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func2);
		problem_d.AddResidualBlock(c_anc2, NULL, optParasT[frameNum - 1]);
	}

	double weight3d[1];
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection

	for (int i = 0;i < frameNum + 1;i++) {

		if (i % originskip == 0) {
			vector<int> selected;
			goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, images_g.at(i).size().height/800.0, 100);
			vector<cv::Point2f> fps;
			vector<Vector3d> wps;
			for (auto itr : selected) {
				cv::Point2f pt = origin3P.at(i).at(itr);
				Vector3d pt3w = originW3p.at(i).at(itr);
				fps.push_back(pt);
				wps.push_back(pt3w);
			}
			origin3P.at(i) = vector<cv::Point2f>(fps);
			originW3p.at(i) = vector<Vector3d>(wps);
		}
		else {
			origin3P.at(i).clear();
			originW3p.at(i).clear();
		}
	}

	vector<BADepthKnownCostFunc_scale*> costFncs;
	hist.clear();
	hist.resize((frameNum + 1)*(frameNum + 1));
	for (int i = 0;i < frameNum + 1;i++) {
		if (originW3p.at(i).size() == 0)continue;
		//forward
		for (int j = i + 1;j < min(frameNum + 1, i + frame_thres);j++) {
			if (j <= anchor_under)continue;//i<j<=anchor_under
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			//cout << i<<","<<j << endl;
			//cout << framePos.at(baseFrame + i * skipNum) << endl;
			//cout << framePos.at(baseFrame + j * skipNum) << endl;
			//cout << trFrame << endl;

			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1' && err.at(k) < 40) {

					Vector3d nxt3v;

					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (i <= anchor_under) {
						BADepthKnownCostFunc_PrjT_anckor *f;
						if (i == 0) {
							f = new BADepthKnownCostFunc_PrjT_anckor(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1]);
						}
						else {
							double ancm[6];
							ancm[0] = optParasR[i - 1][0];
							ancm[1] = optParasR[i - 1][1];
							ancm[2] = optParasR[i - 1][2];
							ancm[3] = optParasT[i - 1][0];
							ancm[4] = optParasT[i - 1][1];
							ancm[5] = optParasT[i - 1][2];
							f = new BADepthKnownCostFunc_PrjT_anckor(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1],ancm);
						}
						cost_3d_a.push_back(f);
						f->set(optParasT[j - 1]);

							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor, ceres::CENTRAL, 3, 3>(f);
							f->outlierRejection(thres3d);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[j - 1]);

							count3d++;
							hist[(i)*(frameNum+1)+(j)]+=1;

					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						double residual[3];
						cost_3d.push_back(f);
						f->set(optParasT[i - 1], optParasT[j - 1]);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
							ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
							f->outlierRejection(thres3d);
							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
							count3d++;
							hist[(i)*(frameNum + 1) + (j)] += 1;
						
					}

				}
			}
		}
		//backward
		for (int j = i - 1;j >= max(0, i - frame_thres);j--) {
			if (i <= anchor_under)continue;
			Matrix4d trFrame = _6dof2m(framePos.at(baseFrame + j * skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + i * skipNum));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			for (int k = 0;k < originW3p.at(i).size();k++) {
				if (status.at(k) == '\1'&& err.at(k) < 40) {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (j <= anchor_under) {

						BADepthKnownCostFunc_PrjT_anckor_rev *f;
						
						if(j == 0) f = new BADepthKnownCostFunc_PrjT_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1]);
						else {
							double ancm[6];
							ancm[0] = optParasR[j - 1][0];
							ancm[1] = optParasR[j - 1][1];
							ancm[2] = optParasR[j - 1][2];
							ancm[3] = optParasT[j - 1][0];
							ancm[4] = optParasT[j - 1][1];
							ancm[5] = optParasT[j - 1][2];
							f = new BADepthKnownCostFunc_PrjT_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1],ancm);
						}
						cost_3d_ar.push_back(f);
						f->set(optParasT[i - 1]);
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor_rev, ceres::CENTRAL, 3, 3>(f);
						f->outlierRejection(thres3d);
						problem.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optParasT[i - 1]);
						count3d++;
						hist[(i)*(frameNum + 1) + (j)] += 1;

					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						cost_3d.push_back(f);
						f->set(optParasT[i - 1], optParasT[j - 1]);
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
						f->outlierRejection(thres3d);
						problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
						count3d++;
														hist[(i)*(frameNum + 1) + (j)]+=1;
					}
				}
			}
		}
	}
	histidx = 0;
	returnlog << endl << endl;
	for (auto itr = hist.begin();hist.end() != itr;itr++) {
		returnlog << *itr << ",";
		histidx++;
		if (histidx == frameNum + 1) {
			histidx = 0;
			returnlog << endl;
		}

	}
	cout << "3d:" << count3d << "   2d:" << count << endl;

	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options optionsrough,optionsfine;
	optionsrough.max_num_iterations = 1e2;
	optionsrough.function_tolerance = 1e-5;
	optionsrough.parameter_tolerance = 1e-5;
	optionsrough.initial_trust_region_radius = 1e6;
	optionsrough.num_threads = 12;
	optionsfine.max_num_iterations = 1e3;
	optionsfine.function_tolerance = 1e-7;
	optionsfine.parameter_tolerance = 1e-7;
	optionsfine.initial_trust_region_radius = 1e6;
	optionsfine.num_threads = 12;
	ceres::Solver::Summary summary;
	double thres2dfine = 0.01*0.01,thres3dfine = 0.01*0.01;
	for (int i = 0;i < frameNum;i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}
	ceres::Solve(optionsrough, &problem_d, &summary);
	for (auto itr : cost_2d)itr->outlierRejection(thres2dfine);
	for (auto itr : cost_2d_a)itr->outlierRejection(thres2dfine);
	ceres::Solve(optionsfine, &problem_d, &summary);
	for (int i = 0;i < frameNum ;i++) {
		returnlog << optParasT[i][0] << "," << optParasT[i][1] << ","<<optParasT[i][2] << endl;
	}
	ceres::Solve(optionsrough, &problem, &summary);
	for (auto itr : cost_3d)itr->outlierRejection(thres3dfine);
	for (auto itr : cost_3d_a)itr->outlierRejection(thres3dfine);
	for (auto itr : cost_3d_ar)itr->outlierRejection(thres3dfine);
	ceres::Solve(optionsfine, &problem, &summary);
	for (int i = 0;i < frameNum;i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}
	//ceres::Solve(options, &problem2, &summary);

	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);

	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	//position update
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < frameNum) {
			//interpolation for frame down sampling
			if (i == baseFrame + (updateIdx)*skipNum) {
				
				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0] * mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum))*mupdate*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum)).inverse();
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - baseFrame) % skipNum) / (double)skipNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == baseFrame + (updateIdx + 1)*skipNum - 1) {
				updateIdx++;
			}
		}
		_6dof temp = framePos.at(i);
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		framePos.at(i) = temp;
	}

	for (int i = 0;i < frameNum;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return returnlog.str();

}






string SensorFusionBA::BatchBA_Global_seq(vector<int> sampledFrame, int overlap) {
	stringstream returnlog;
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(sampledFrame.at(0)));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;
	double tolTime = 1 / 5.0;
	int frame_thres = 800;// threshold for frame 
	int anchor_under = overlap;// anchor
	int originskip = 1;
	if (originskip <= 0)originskip = 1;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;

	vector<vector<cv::Point2f>> origin3P(sampledFrame.size());
	vector <vector<Vector3d>> originW3p(sampledFrame.size());//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	int baseFrame = sampledFrame.at(0);
	double startTime = inputs.imTimeStamp[baseFrame] - tolTime;
	inputs.psl->seekByTime(startTime);
	std::cout << startTime << std::endl;

	float dat[5];
	int baseframe_ = baseFrame;
	int framecnt = 0;
	while (inputs.psl->getNextPointData(dat)) {
		Vector3d p, pt; p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t = dat[4];
//		std::cout << t << std::endl;

		if (t > inputs.imTimeStamp[sampledFrame.at(framecnt)] + tolTime || (sampledFrame.at(framecnt) == framePos.size() - 1 && t > inputs.imTimeStamp[sampledFrame.at(framecnt)])) {
		
			if (framecnt < sampledFrame.size() - 1) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[sampledFrame.at(framecnt)] - tolTime;
				inputs.psl->seekByTime(startTime_);

				continue;
			}
			else {

				break;
			}

		}
		if (ref <= 0.00) {

			continue;
		}
		if (p.norm() < distLower || p.norm() > distUpper) {

			continue;
		}
		cv::Scalar color; color = cv::Scalar(255, 0, 0);

		if (t < inputs.imTimeStamp[sampledFrame.at(framecnt)]) {
			Matrix4d mbase = _6dof2m(framePos.at(sampledFrame.at(framecnt)));
			Matrix4d mprev = sampledFrame.at(framecnt)>0?_6dof2m(framePos.at(sampledFrame.at(framecnt) - 1)):mbase;
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);

			Vector4d q = dcm2q(diffR);
			Vector4d q_b; q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[sampledFrame.at(framecnt) - 1];
			double frameEndTime = inputs.imTimeStamp[sampledFrame.at(framecnt)];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1 - rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else {
			if (sampledFrame.at(framecnt) + 1 >= framePos.size()) {
				cout << framecnt << endl;
				cout << framePos.size() << endl;
			}
			Matrix4d mbase = _6dof2m(framePos.at(sampledFrame.at(framecnt) + 1));
			Matrix4d mprev = _6dof2m(framePos.at(sampledFrame.at(framecnt)));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b; q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[sampledFrame.at(framecnt)];
			double frameEndTime = inputs.imTimeStamp[sampledFrame.at(framecnt) + 1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		//cv::circle(image1, projp, 0, color);
		if (!stat.bMask || stat.mask.at<uchar>(projp) >= 128) {
			origin3P.at(framecnt).push_back(projp);
			originW3p.at(framecnt).push_back(pt);
		}

	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*sampledFrame.size());
	double** optParasR = (double**)malloc(sizeof(double*)*sampledFrame.size());
	double* optScale = (double*)malloc(sizeof(double)*sampledFrame.size());
	double optScale_[1] = { 1.0 };
	double scale = 0;
	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(sampledFrame.at(i + 1)));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
	}

	ceres::Problem problem_d, problem, problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(sampledFrame.at(i + 1)));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}

	vector< map<int, cv::Point2f>> f_pts = stat.bMask ? pointTracking(images_g, optParasR, 1, stat.mask, dcpara, fitv) : pointTracking(images_g, optParasR,1,cv::Mat(),dcpara,fitv);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else if (itr2.first > overlap) {
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.05) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}if (p(0)!=p(0) || p(1) != p(1) || p(2) != p(2)) {
			std::cout << p.transpose() << ":" << s <<"," << t << ":"<<std::endl;
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	//{
	//	float* vtx;
	//	float* rfl;
	//	vtx = (float*)malloc(sizeof(float)*idx * 3);
	//	rfl = (float*)malloc(sizeof(float)*idx);
	//	for (int i = 0;i < idx;i++) {
	//		vtx[i * 3] = sfm_point[i][0];
	//		vtx[i * 3 + 1] = sfm_point[i][1];
	//		vtx[i * 3 + 2] = sfm_point[i][2];
	//		rfl[i] = 0.5;
	//	}
	//	BasicPly bp;

	//	bp.setVertecesPointer(vtx, idx);
	//	bp.setReflectancePointer(rfl, idx);
	//	bp.writePlyFileAuto("sfm_b.ply");
	//	bp.release();
	//}

	idx = 0;
	int count = 0;
	double weight2d[1]; weight2d[0] = 1.0;
	vector<int> hist = vector<int>(images_g.size() + 1, 0);
	vector<int> hist2 = vector<int>(images_g.size() + 1, 0);
	vector<BaFeatureProjCostFunction_*> cost_2d;
	vector<BaFeatureProjCostFunction_Ancker*> cost_2d_a;
	vector<BADepthKnownCostFunc_PrjT*> cost_3d;
	vector<BADepthKnownCostFunc_PrjT_anckor*> cost_3d_a;
	vector<BADepthKnownCostFunc_PrjT_anckor_rev*> cost_3d_ar;
	double thres2d = 0.1*0.1;
	double thres3d = 0.001*0.001;

	AdjustableHuberLoss* my2dloss = new AdjustableHuberLoss(10);
	AdjustableHuberLoss* my3dloss = new AdjustableHuberLoss(10);

	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			hist2[itr2.first] += 1;
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (/*itr2.first < 0*/itr2.first <= anchor_under) {
				double ancm[6];

				BaFeatureProjCostFunction_Ancker* f;
				if (itr2.first == 0) {
					f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				}
				else {
					ancm[0] = optParasR[itr2.first - 1][0];
					ancm[1] = optParasR[itr2.first - 1][1];
					ancm[2] = optParasR[itr2.first - 1][2];
					ancm[3] = optParasT[itr2.first - 1][0];
					ancm[4] = optParasT[itr2.first - 1][1];
					ancm[5] = optParasT[itr2.first - 1][2];
					f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d, ancm);
				}

				cost_2d_a.push_back(f);
				f->set(sfm_point[idx]);
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
				f->outlierRejection(thres2d);
				problem_d.AddResidualBlock(c, my2dloss, sfm_point[idx]);
				count++;
				hist[itr2.first] += 1;
			}
			else {
				BaFeatureProjCostFunction_* f = new BaFeatureProjCostFunction_(nxt3v, weight2d);
				cost_2d.push_back(f);
				f->set(optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
				f->fixTrans(bfixTrans);
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_, ceres::CENTRAL, 3, 3, 3, 3>(f);
				f->outlierRejection(thres2d);
				problem_d.AddResidualBlock(c, my2dloss, optParasR[itr2.first - 1], optParasT[itr2.first - 1], sfm_point[idx]);
				count++;
				hist[itr2.first] += 1;
			}

		}
		idx++;
	}
	int histidx = 0;
	returnlog << endl << endl;
	bool setAnchor = false;
	for (auto itr = hist.begin(); hist.end() != itr; itr++) {
		returnlog << *itr << "/" << hist2.at(histidx) << endl;
		histidx++;
		if (*itr < 5) setAnchor = true;
	}

	if (anchor_under == 0 && !bfixTrans /*|| setAnchor*/) {
		scaleAnckor* sa_func2 = new scaleAnckor(optParasT[sampledFrame.size() - 2][0] * optParasT[sampledFrame.size() - 2][0] + optParasT[sampledFrame.size() - 2][1] * optParasT[sampledFrame.size() - 2][1] + optParasT[sampledFrame.size() - 2][2] * optParasT[sampledFrame.size() - 2][2]);
		ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func2);
		problem_d.AddResidualBlock(c_anc2, NULL, optParasT[sampledFrame.size() - 2]);
	}

	double weight3d[1];
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection

	for (int i = 0; i < sampledFrame.size(); i++) {

		if (i % originskip == 0) {
			vector<int> selected;
			goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, images_g.at(i).size().height / 200.0, 200);
			vector<cv::Point2f> fps;
			vector<Vector3d> wps;
			for (auto itr : selected) {
				cv::Point2f pt = origin3P.at(i).at(itr);
				Vector3d pt3w = originW3p.at(i).at(itr);
				fps.push_back(pt);
				wps.push_back(pt3w);
			}
			origin3P.at(i) = vector<cv::Point2f>(fps);
			originW3p.at(i) = vector<Vector3d>(wps);
		}
		else {
			origin3P.at(i).clear();
			originW3p.at(i).clear();
		}
	}

	vector<BADepthKnownCostFunc_scale*> costFncs;
	hist.clear();
	hist.resize((sampledFrame.size())*(sampledFrame.size()));
	for (int i = 0; i < sampledFrame.size(); i++) {
		if (originW3p.at(i).size() == 0)continue;
		//forward
		for (int j = i + 1; j < min(sampledFrame.size(), i + frame_thres); j++) {
			if (j <= anchor_under)continue;//i<j<=anchor_under
			Matrix4d trFrame = _6dof2m(framePos.at(sampledFrame.at(j))).inverse()*_6dof2m(framePos.at(sampledFrame.at(i)));
			//cout << i<<","<<j << endl;
			//cout << framePos.at(baseFrame + i * skipNum) << endl;
			//cout << framePos.at(baseFrame + j * skipNum) << endl;
			//cout << trFrame << endl;

			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}

			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, 
				cv::Size(11, 11), 0, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

			for (int k = 0; k < originW3p.at(i).size(); k++) {
				if (status.at(k) == '\1' && err.at(k) < 10) {

					Vector3d nxt3v;

					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (i <= anchor_under) {
						BADepthKnownCostFunc_PrjT_anckor *f;
						if (i == 0) {
							f = new BADepthKnownCostFunc_PrjT_anckor(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1]);
						}
						else {
							double ancm[6];
							ancm[0] = optParasR[i - 1][0];
							ancm[1] = optParasR[i - 1][1];
							ancm[2] = optParasR[i - 1][2];
							ancm[3] = optParasT[i - 1][0];
							ancm[4] = optParasT[i - 1][1];
							ancm[5] = optParasT[i - 1][2];
							f = new BADepthKnownCostFunc_PrjT_anckor(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1], ancm);
						}
						cost_3d_a.push_back(f);
						f->set(optParasT[j - 1]);

						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor, ceres::CENTRAL, 3, 3>(f);
						f->outlierRejection(thres3d);
						problem.AddResidualBlock(c, my3dloss, optParasT[j - 1]);

						count3d++;
						hist[(i)*(sampledFrame.size()) + (j)] += 1;

					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						double residual[3];
						cost_3d.push_back(f);
						f->set(optParasT[i - 1], optParasT[j - 1]);
						//cout << residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] << endl;
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
						f->outlierRejection(thres3d);
						problem.AddResidualBlock(c, my3dloss, optParasT[i - 1], optParasT[j - 1]);
						count3d++;
						hist[(i)*(sampledFrame.size()) + (j)] += 1;

					}

				}
			}
		}
		//backward
		for (int j = i - 1; j >= max(0, i - frame_thres); j--) {
			if (i <= anchor_under)continue;
			Matrix4d trFrame = _6dof2m(framePos.at(sampledFrame.at(j))).inverse()*_6dof2m(framePos.at(sampledFrame.at(i)));
			next3Pts.clear();
			for (auto w3p : originW3p.at(i)) {
				Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
				double ix, iy;
				omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
				cv::Point2f projp(ix, iy);
				next3Pts.push_back(projp);
			}
			vector<cv::Point2f> nptcopy(next3Pts);
			cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, 
				cv::Size(11, 11), 0, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			/*if (i == 15) {
				for (int k = 0; k < next3Pts.size(); k += 5) {
					std::cout << k << ":" << origin3P.at(i).at(k) << ":" << nptcopy.at(k) << ":" << next3Pts.at(k) << std::endl;
				}
				std::cout << std::endl;
			}*/
			for (int k = 0; k < originW3p.at(i).size(); k++) {
				if (status.at(k) == '\1'&& err.at(k) < 10) {

					Vector3d nxt3v;
					rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

					if (j <= anchor_under) {

						BADepthKnownCostFunc_PrjT_anckor_rev *f;

						if (j == 0) f = new BADepthKnownCostFunc_PrjT_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1]);
						else {
							double ancm[6];
							ancm[0] = optParasR[j - 1][0];
							ancm[1] = optParasR[j - 1][1];
							ancm[2] = optParasR[j - 1][2];
							ancm[3] = optParasT[j - 1][0];
							ancm[4] = optParasT[j - 1][1];
							ancm[5] = optParasT[j - 1][2];
							f = new BADepthKnownCostFunc_PrjT_anckor_rev(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], ancm);
						}
						cost_3d_ar.push_back(f);
						f->set(optParasT[i - 1]);
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor_rev, ceres::CENTRAL, 3, 3>(f);
						f->outlierRejection(thres3d);
						problem.AddResidualBlock(c, my3dloss, optParasT[i - 1]);
						count3d++;
						hist[(i)*(sampledFrame.size()) + (j)] += 1;

					}
					else {
						BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
						cost_3d.push_back(f);
						f->set(optParasT[i - 1], optParasT[j - 1]);
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
						f->outlierRejection(thres3d);
						problem.AddResidualBlock(c, my3dloss, optParasT[i - 1], optParasT[j - 1]);
						count3d++;
						hist[(i)*(sampledFrame.size()) + (j)] += 1;
					}
				}
			}
		}
	}
	histidx = 0;
	returnlog << endl << endl;
	for (auto itr = hist.begin(); hist.end() != itr; itr++) {
		returnlog << *itr << ",";
		histidx++;
		if (histidx == sampledFrame.size()) {
			histidx = 0;
			returnlog << endl;
		}

	}
	cout << "3d:" << count3d << "   2d:" << count << endl;

	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options optionsrough, optionsfine;
	optionsrough.max_num_iterations = 1e2;
	optionsrough.function_tolerance = 1e-5;
	optionsrough.parameter_tolerance = 1e-5;
	optionsrough.initial_trust_region_radius = 1e6;
	optionsrough.num_threads = 12;
	optionsfine.max_num_iterations = 1e3;
	optionsfine.function_tolerance = 1e-7;
	optionsfine.parameter_tolerance = 1e-7;
	optionsfine.initial_trust_region_radius = 1e6;
	optionsfine.num_threads = 12;
	ceres::Solver::Summary summary;
	double thres2dfine = 0.01*0.01, thres3dfine = 0.01*0.01;
	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}

	double stdev = 0;
	int inlcnt = 0;

	//compute stdev
	for (auto itr : cost_2d) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	for (auto itr : cost_2d_a) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	stdev /= inlcnt;
	stdev = sqrt(stdev);
	my2dloss->resetScale(stdev);
	std::cout << "init 2D loss,inlier/all: " << stdev << "," << inlcnt << "/" << cost_2d.size() + cost_2d_a.size() << std::endl;
	ceres::Solve(optionsrough, &problem_d, &summary);
	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}
	//compute stdev again
	stdev = 0; inlcnt = 0;
	for (auto itr : cost_2d) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	for (auto itr : cost_2d_a) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	stdev /= inlcnt;
	stdev = sqrt(stdev);
	thres2dfine = (2 * stdev) * (2 * stdev);//set threshold to sigma
	inlcnt = 0;
	for (auto itr : cost_2d) { bool bin = itr->outlierRejection(thres2dfine); if (bin)inlcnt++; }
	for (auto itr : cost_2d_a) { bool bin = itr->outlierRejection(thres2dfine); if (bin)inlcnt++; }
	my2dloss->resetScale(stdev);
	std::cout << "fine 2D loss,inlier/all: " << stdev << "," << inlcnt << "/" << cost_2d.size() + cost_2d_a.size() << std::endl;
	ceres::Solve(optionsfine, &problem_d, &summary);
	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
		returnlog << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}


	//compute stdev (3d)
	stdev = 0; inlcnt = 0;
	for (auto itr : cost_3d) {
		if (!itr->bInlier())continue;
		double var = itr->squaredCostEveluate();
		stdev += var;
		//std::cout << var << ",";
		inlcnt++;
	}
	for (auto itr : cost_3d_a) {
		if (!itr->bInlier())continue;
		double var = itr->squaredCostEveluate();
		stdev += var;
		//std::cout << var << ",";
		inlcnt++;
	}
	for (auto itr : cost_3d_ar) {
		if (!itr->bInlier())continue;
		double var = itr->squaredCostEveluate();
		stdev += var;
		//std::cout << var << ",";
		inlcnt++;
	}
	stdev /= inlcnt;
	stdev = sqrt(stdev);

	thres3dfine = (2 * stdev) * (2 * stdev);
	inlcnt = 0;
	std::vector<double> berr,aerr;
	for (auto itr : cost_3d) {
		bool bin = itr->outlierRejection(thres3dfine);
		if (bin) { inlcnt++; }
	}
	for (auto itr : cost_3d_a) { bool bin = itr->outlierRejection(thres3dfine); if (bin) inlcnt++; }
	for (auto itr : cost_3d_ar) { bool bin = itr->outlierRejection(thres3dfine); if (bin) inlcnt++; }

	my3dloss->resetScale(stdev);
	std::cout << "init 3d thres,loss,inlier/all: " << thres3dfine<< "," << stdev << "," << inlcnt << "/" << cost_3d.size() + cost_3d_a.size() + cost_3d_ar.size() << std::endl;


	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}

	//cost_3d.at(100)->debug();
	//cost_3d.at(200)->debug();
	//cost_3d.at(300)->debug();
	//cost_3d.at(500)->debug();
	//cost_3d.at(1000)->debug();

	ceres::Solve(optionsrough, &problem, &summary);
	//cost_3d.at(100)->debug();
	//cost_3d.at(200)->debug();
	//cost_3d.at(300)->debug();
	//cost_3d.at(500)->debug();
	//cost_3d.at(1000)->debug();

	for (int i = 0; i < sampledFrame.size() - 1; i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}
	//compute stdev (3d)
	stdev = 0; inlcnt = 0;
	for (auto itr : cost_3d) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	for (auto itr : cost_3d_a) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	for (auto itr : cost_3d_ar) {
		if (!itr->bInlier())continue;
		stdev += itr->squaredCostEveluate();
		inlcnt++;
	}
	stdev /= inlcnt;
	stdev = sqrt(stdev);

	thres3dfine = (2 * stdev) * (2 * stdev);//set threshold to 2 sigma
	inlcnt = 0;
	for (auto itr : cost_3d) { bool bin = itr->outlierRejection(thres3dfine);if(bin) inlcnt++; }
	for (auto itr : cost_3d_a) { bool bin = itr->outlierRejection(thres3dfine); if (bin) inlcnt++; }
	for (auto itr : cost_3d_ar) { bool bin = itr->outlierRejection(thres3dfine); if (bin) inlcnt++; }

	my3dloss->resetScale(stdev);
	std::cout << "fine 3d loss,inlier/all: " << stdev << "," << inlcnt << "/" << cost_3d.size() + cost_3d_a.size() + cost_3d_ar.size() << std::endl;

	ceres::Solve(optionsfine, &problem, &summary);

	for (int i = 0;i < sampledFrame.size()-1;i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}
	//ceres::Solve(options, &problem2, &summary);

	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);

	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	//position update
	int skippedNum = 1;
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < sampledFrame.size()-1) {
			//interpolation for frame down sampling
			if (i == sampledFrame.at(updateIdx)/*baseFrame + (updateIdx)*skipNum*/) {

				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0] * mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(sampledFrame.at(updateIdx)))*mupdate*_6dof2m(framePos.at(sampledFrame.at(updateIdx+1))).inverse();
				skippedNum = sampledFrame.at(updateIdx + 1)- sampledFrame.at(updateIdx);
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - sampledFrame.at(updateIdx))) / (double)skippedNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == sampledFrame.at(updateIdx + 1) - 1) {
				updateIdx++;
			}
		}
		_6dof temp = framePos.at(i);
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		framePos.at(i) = temp;
	}

	for (int i = 0;i < sampledFrame.size()-1;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return returnlog.str();

}



string SensorFusionBA::BatchBA_Global_seq2(vector<int> sampledFrame, int overlap) {
	stringstream returnlog;
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(sampledFrame.at(0)));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;

	int frame_thres = 8;// threshold for frame 
	int anchor_under = overlap;// anchor
	int originskip = 1;
	if (originskip <= 0)originskip = 1;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;

	vector<vector<cv::Point2f>> origin3P(sampledFrame.size());
	vector <vector<Vector3d>> originW3p(sampledFrame.size());//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	int baseFrame = sampledFrame.at(0);
	double startTime = inputs.imTimeStamp[baseFrame] - 1 / 32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	while (inputs.psl->getNextPointData(dat)) {
		Vector3d p, pt;p << dat[0], dat[1], dat[2];
		double ref = dat[3];
		double t = dat[4];
		if (t > inputs.imTimeStamp[sampledFrame.at(framecnt)] + 1 / 32.0 || (sampledFrame.at(framecnt) == framePos.size() - 1 && t > inputs.imTimeStamp[sampledFrame.at(framecnt)])) {


			if (framecnt < sampledFrame.size() - 1) {
				framecnt++;
				double startTime_ = inputs.imTimeStamp[sampledFrame.at(framecnt)] - 1 / 32.0;
				inputs.psl->seekByTime(startTime_);
				free(dat);
				continue;
			}
			else {
				free(dat);
				break;
			}

		}
		if (ref < 1e-2) {
			free(dat);
			continue;
		}
		if (p.norm() < distLower || p.norm() > distUpper) {
			free(dat);
			continue;
		}
		cv::Scalar color;color = cv::Scalar(255, 0, 0);
		if (t < inputs.imTimeStamp[sampledFrame.at(framecnt)]) {
			Matrix4d mbase = _6dof2m(framePos.at(sampledFrame.at(framecnt)));
			Matrix4d mprev = _6dof2m(framePos.at(sampledFrame.at(framecnt) - 1));
			Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
			Matrix4d mdiff = mdiff_.inverse();
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);

			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[sampledFrame.at(framecnt) - 1];
			double frameEndTime = inputs.imTimeStamp[sampledFrame.at(framecnt)];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q + rate * q_b;
			q_ = q_.normalized();
			Vector3d t_ = (1 - rate) * diffT;
			Matrix3d interpolatedR = q2dcm(q_);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
			//p - camera
			pt = interpolatedR * pt + interpolatedT;//
			//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
			color = cv::Scalar(255, 0, 0);
		}
		else {
			if (sampledFrame.at(framecnt) + 1 >= framePos.size()) {
				cout << framecnt << endl;
				cout << framePos.size() << endl;
			}
			Matrix4d mbase = _6dof2m(framePos.at(sampledFrame.at(framecnt) + 1));
			Matrix4d mprev = _6dof2m(framePos.at(sampledFrame.at(framecnt)));
			Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
			Matrix3d diffR = mdiff.block(0, 0, 3, 3);
			Vector4d q = dcm2q(diffR);
			Vector4d q_b;q_b << 0, 0, 0, 1;
			Vector3d diffT = mdiff.block(0, 3, 3, 1);
			double frameStartTime = inputs.imTimeStamp[sampledFrame.at(framecnt)];
			double frameEndTime = inputs.imTimeStamp[sampledFrame.at(framecnt) + 1];
			double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
			Vector4d q_ = (1 - rate)*q_b + rate * q;
			Vector4d q_2 = q_.normalized();
			//cout << q.transpose() << "," << q_2.transpose() << endl;
			Vector3d t_ = rate * diffT;
			Matrix3d interpolatedR = q2dcm(q_2);
			Vector3d interpolatedT = t_;
			//p - sensor
			pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
																						  //p - camera
			pt = interpolatedR * pt + interpolatedT;//
			color = cv::Scalar(0, 0, 255);
		}
		double ix, iy;
		omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
		cv::Point2f projp(ix, iy);
		cv::circle(image1, projp, 0, color);
		origin3P.at(framecnt).push_back(projp);
		originW3p.at(framecnt).push_back(pt);
		free(dat);
	}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*sampledFrame.size());
	double** optParasR = (double**)malloc(sizeof(double*)*sampledFrame.size());
	int ctl_pt = ((sampledFrame.size()-overlap) / 15) + 2;
	double** optParas_bend = (double**)malloc(sizeof(double*)*(ctl_pt-1));
	double** optt_bend = (double**)malloc(sizeof(double*)*(ctl_pt - 1));
	double* optScale = (double*)malloc(sizeof(double)*sampledFrame.size());
	double optScale_[1] = { 1.0 };
	double scale = 0;
	vector<double>w1a,w2a,accleng;
	vector<int>ctl_pt_idx;
	double acclensum = 0;
	accleng.push_back(0);
	for (int i = 0;i < sampledFrame.size() - 1;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(sampledFrame.at(i + 1)));
		Matrix4d trFrame_ = _6dof2m(framePos.at(sampledFrame.at(i))).inverse()*_6dof2m(framePos.at(sampledFrame.at(i + 1)));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
		if (i >= anchor_under) {
			acclensum += trFrame_.block(0, 3, 3, 1).norm();
		}
		accleng.push_back(acclensum);
	}
	double splitlen = acclensum / (ctl_pt - 1);
	for (int i = 0;i < ctl_pt-1;i++) {
		optParas_bend[i] = (double*)malloc(sizeof(double) * 6);
		optt_bend[i] = (double*)malloc(sizeof(double) * 3);
		memset(optParas_bend[i], 0, sizeof(double) * 6);
		memset(optt_bend[i], 0, sizeof(double) * 3);
	}
	double sqrt2 = sqrt(2);
	for (int i = 0;i < sampledFrame.size();i++) {
		if (i >= anchor_under) {
			int idx = accleng.at(i)/splitlen;
			double w2 = (accleng.at(i) - idx * splitlen) / splitlen;
			//if (w2 < 0.5) {
			//	w2 = sqrt2 * w2*sqrt(w2);
			//}
			//else {
			//	w2 = 1 - w2;
			//	w2 =1 - sqrt2 * w2*sqrt(w2);
			//}
			double w1 = 1 - w2;

			if (idx >= ctl_pt - 1) {
				idx = ctl_pt - 2;
				w2 = 1;
				w1 = 0;
			}
			w1a.push_back(w1);
			w2a.push_back(w2);
			ctl_pt_idx.push_back(idx);
		}
		else {
			w1a.push_back(0);
			w2a.push_back(0);
			ctl_pt_idx.push_back(-1);
		}
	}

	ceres::Problem problem_d, problem, problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < sampledFrame.size() - 1;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(sampledFrame.at(i + 1)));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}

	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g, optParasR);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else if (itr2.first > overlap) {
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.05) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	//{
	//	float* vtx;
	//	float* rfl;
	//	vtx = (float*)malloc(sizeof(float)*idx * 3);
	//	rfl = (float*)malloc(sizeof(float)*idx);
	//	for (int i = 0;i < idx;i++) {
	//		vtx[i * 3] = sfm_point[i][0];
	//		vtx[i * 3 + 1] = sfm_point[i][1];
	//		vtx[i * 3 + 2] = sfm_point[i][2];
	//		rfl[i] = 0.5;
	//	}
	//	BasicPly bp;

	//	bp.setVertecesPointer(vtx, idx);
	//	bp.setReflectancePointer(rfl, idx);
	//	bp.writePlyFileAuto("sfm_b.ply");
	//	bp.release();
	//}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	vector<int> hist = vector<int>(images_g.size() + 1, 0);
	vector<int> hist2 = vector<int>(images_g.size() + 1, 0);
	vector<BaFeatureProjCostFunction_fbend*> cost_2d;
	vector<BaFeatureProjCostFunction_Ancker*> cost_2d_a;
	vector<BaFeatureProjCostFunction_fbend_anchor*> cost_2d_a2;
	vector<BADepthKnownCostFunc_PrjT*> cost_3d;
	vector<BADepthKnownCostFunc_PrjT_anckor*> cost_3d_a;
	vector<BADepthKnownCostFunc_PrjT_anckor_rev*> cost_3d_ar;
	double thres2d = 0.05*0.05;
	double thres3d = 0.05*0.05;
	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			hist2[itr2.first] += 1;
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);

			//weight computation


			if (/*itr2.first < 0*/itr2.first <= anchor_under) {
				double ancm[6];

				BaFeatureProjCostFunction_Ancker* f;
				if (itr2.first == 0) {
					f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				}
				else {
					ancm[0] = optParasR[itr2.first - 1][0];
					ancm[1] = optParasR[itr2.first - 1][1];
					ancm[2] = optParasR[itr2.first - 1][2];
					ancm[3] = optParasT[itr2.first - 1][0];
					ancm[4] = optParasT[itr2.first - 1][1];
					ancm[5] = optParasT[itr2.first - 1][2];
					f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d, ancm);
				}

				cost_2d_a.push_back(f);
				f->set(sfm_point[idx]);
				ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
				f->outlierRejection(thres2d);
				problem_d.AddResidualBlock(c, new ceres::CauchyLoss(1e-3), sfm_point[idx]);
				count++;
				hist[itr2.first] += 1;
			}
			else {
				_6dof mot = { optParasR[itr2.first - 1][0] ,optParasR[itr2.first - 1][1] ,optParasR[itr2.first - 1][2],optParasT[itr2.first - 1][0],optParasT[itr2.first - 1][1],optParasT[itr2.first - 1][2]};
				if (ctl_pt_idx.at(itr2.first)>0) {
					BaFeatureProjCostFunction_fbend* f = new BaFeatureProjCostFunction_fbend(nxt3v, weight2d, mot, w1a.at(itr2.first), w2a.at(itr2.first));
					cost_2d.push_back(f);
					f->set(optParas_bend[ctl_pt_idx.at(itr2.first) - 1], optParas_bend[ctl_pt_idx.at(itr2.first)], sfm_point[idx]);
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_fbend, ceres::CENTRAL, 3, 6,6,3>(f);
					f->outlierRejection(thres2d);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(1e-3), optParas_bend[ctl_pt_idx.at(itr2.first) - 1], optParas_bend[ctl_pt_idx.at(itr2.first)], sfm_point[idx]);
					count++;
					hist[itr2.first] += 1;
				}
				else {

						BaFeatureProjCostFunction_fbend_anchor* f = new BaFeatureProjCostFunction_fbend_anchor(nxt3v, weight2d, mot, w1a.at(itr2.first), w2a.at(itr2.first));
						cost_2d_a2.push_back(f);
						f->set( optParas_bend[ctl_pt_idx.at(itr2.first)], sfm_point[idx]);
						ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_fbend_anchor, ceres::CENTRAL, 3, 6, 3>(f);
						f->outlierRejection(thres2d);
						problem_d.AddResidualBlock(c, new ceres::CauchyLoss(1e-3), optParas_bend[ctl_pt_idx.at(itr2.first)], sfm_point[idx]);
						count++;
						hist[itr2.first] += 1;

				
				}
			}

		}
		idx++;
	}
	int histidx = 0;
	returnlog << endl << endl;
	bool setAnchor = false;
	for (auto itr = hist.begin();hist.end() != itr;itr++) {
		returnlog << *itr << "/" << hist2.at(histidx) << endl;
		histidx++;
		if (*itr < 5) setAnchor = true;
	}

	if (anchor_under == 0 /*|| setAnchor*/) {
		scaleAnckor_b* sa_func2 = new scaleAnckor_b(optParasT[sampledFrame.size() - 2]);
		ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor_b, ceres::CENTRAL, 1, 6>(sa_func2);
		problem_d.AddResidualBlock(c_anc2, NULL, optParas_bend[ctl_pt-2]);
	}

	double weight3d[1];
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection

	for (int i = 0;i < sampledFrame.size();i++) {

		if (i % originskip == 0) {
			vector<int> selected;
			goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, images_g.at(i).size().height / 800.0, 100);
			vector<cv::Point2f> fps;
			vector<Vector3d> wps;
			for (auto itr : selected) {
				cv::Point2f pt = origin3P.at(i).at(itr);
				Vector3d pt3w = originW3p.at(i).at(itr);
				fps.push_back(pt);
				wps.push_back(pt3w);
			}
			origin3P.at(i) = vector<cv::Point2f>(fps);
			originW3p.at(i) = vector<Vector3d>(wps);
		}
		else {
			origin3P.at(i).clear();
			originW3p.at(i).clear();
		}
	}

	vector<BADepthKnownCostFunc_scale*> costFncs;
	hist.clear();
	hist.resize((sampledFrame.size())*(sampledFrame.size()));
	//for (int i = 0;i < sampledFrame.size();i++) {
	//	if (originW3p.at(i).size() == 0)continue;
	//	//forward
	//	for (int j = i + 1;j < min(sampledFrame.size(), i + frame_thres);j++) {
	//		if (j <= anchor_under)continue;//i<j<=anchor_under
	//		Matrix4d trFrame = _6dof2m(framePos.at(sampledFrame.at(j))).inverse()*_6dof2m(framePos.at(sampledFrame.at(i)));
	//		//cout << i<<","<<j << endl;
	//		//cout << framePos.at(baseFrame + i * skipNum) << endl;
	//		//cout << framePos.at(baseFrame + j * skipNum) << endl;
	//		//cout << trFrame << endl;

	//		next3Pts.clear();
	//		for (auto w3p : originW3p.at(i)) {
	//			Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
	//			double ix, iy;
	//			omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
	//			cv::Point2f projp(ix, iy);
	//			next3Pts.push_back(projp);
	//		}

	//		cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);

	//		for (int k = 0;k < originW3p.at(i).size();k++) {
	//			if (status.at(k) == '\1' && err.at(k) < 40) {

	//				Vector3d nxt3v;

	//				rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

	//				if (i <= anchor_under) {
	//					BADepthKnownCostFunc_PrjT_anckor2 *f;
	//					if (i == 0) {
	//						f = new BADepthKnownCostFunc_PrjT_anckor2(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1], optParasT[j - 1],w1a.at(j),w2a.at(j));
	//					}
	//					else {
	//						double ancm[6];
	//						ancm[0] = optParasR[i - 1][0];
	//						ancm[1] = optParasR[i - 1][1];
	//						ancm[2] = optParasR[i - 1][2];
	//						ancm[3] = optParasT[i - 1][0];
	//						ancm[4] = optParasT[i - 1][1];
	//						ancm[5] = optParasT[i - 1][2];
	//						f = new BADepthKnownCostFunc_PrjT_anckor2(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[j - 1], optParasT[j - 1], ancm, w1a.at(j), w2a.at(j));
	//					}
	//					//cost_3d_a.push_back(f);

	//					ceres::CostFunction* c;
	//					if (ctl_pt_idx.at(j) == 0) {
	//						f->set(NULL, optt_bend[ctl_pt_idx.at(j)]);
	//						c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anchor_wrap_anc, ceres::CENTRAL, 3, 3>(new BADepthKnownCostFunc_PrjT_anchor_wrap_anc(f));
	//						problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[ctl_pt_idx.at(j)]);
	//					}
	//					else {
	//						f->set(optt_bend[ctl_pt_idx.at(j) - 1], optt_bend[ctl_pt_idx.at(j)]);

	//						c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anchor_wrap, ceres::CENTRAL, 3, 3, 3>(new BADepthKnownCostFunc_PrjT_anchor_wrap(f));
	//						problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[ctl_pt_idx.at(j) - 1], optt_bend[ctl_pt_idx.at(j)]);
	//					}
	//					
	//					
	//					f->outlierRejection(thres3d);


	//					count3d++;
	//					hist[(i)*(sampledFrame.size()) + (j)] += 1;

	//				}
	//				else {
	//					int idx1 = ctl_pt_idx.at(i);
	//					int idx2 = ctl_pt_idx.at(j);

	//					if (idx1 == idx2) {//case1
	//						BADepthKnownCostFunc_PrjT4 *f = new BADepthKnownCostFunc_PrjT4(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1],
	//							optParasT[i - 1], optParasT[j - 1], w1a.at(i), w2a.at(i), w1a.at(j), w2a.at(j));
	//						double residual[3];
	//						//cost_3d.push_back(f);
	//						ceres::CostFunction* c;
	//						if (idx1 == 0) {
	//							f->set(NULL, optt_bend[ctl_pt_idx.at(j)], NULL, optt_bend[ctl_pt_idx.at(j)]);
	//							c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_wrap_1a, ceres::CENTRAL, 3, 3>(new BADepthKnownCostFunc_PrjT_wrap_1a(f));
	//							f->outlierRejection(thres3d);
	//							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3),optt_bend[idx1]);
	//							count3d++;
	//							hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//						}
	//						else {
	//							f->set(optt_bend[ctl_pt_idx.at(j)-1], optt_bend[ctl_pt_idx.at(j)], optt_bend[ctl_pt_idx.at(j)-1], optt_bend[ctl_pt_idx.at(j)]);
	//							c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_wrap_1, ceres::CENTRAL, 3, 3, 3>(new BADepthKnownCostFunc_PrjT_wrap_1(f));
	//							f->outlierRejection(thres3d);
	//							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[idx1 - 1], optt_bend[idx1]);
	//							count3d++;
	//							hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//						}
	//					}
	//					else if (idx1 + 1==idx2) {//case 2
	//						BADepthKnownCostFunc_PrjT4 *f = new BADepthKnownCostFunc_PrjT4(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1],
	//							optParasT[i - 1], optParasT[j - 1], w1a.at(i), w2a.at(i), w1a.at(j), w2a.at(j));
	//						double residual[3];
	//						//cost_3d.push_back(f);
	//						ceres::CostFunction* c;
	//						if (idx1 == 0) {
	//							f->set(NULL, optt_bend[idx1], optt_bend[idx1], optt_bend[idx2]);
	//							c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_wrap_2a, ceres::CENTRAL, 3, 3, 3>(new BADepthKnownCostFunc_PrjT_wrap_2a(f));
	//							f->outlierRejection(thres3d);
	//							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[idx1], optt_bend[idx2]);
	//							count3d++;
	//							hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//						}
	//						else {
	//							f->set(optt_bend[idx1-1], optt_bend[idx1], optt_bend[idx1], optt_bend[idx2]);
	//							c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_wrap_2, ceres::CENTRAL, 3, 3,3, 3>(new BADepthKnownCostFunc_PrjT_wrap_2(f));
	//							f->outlierRejection(thres3d);
	//							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[idx1-1], optt_bend[idx1], optt_bend[idx2]);
	//							count3d++;
	//							hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//						}
	//					}
	//					else{//case 3
	//						BADepthKnownCostFunc_PrjT4 *f = new BADepthKnownCostFunc_PrjT4(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1],
	//							optParasT[i - 1], optParasT[j - 1], w1a.at(i), w2a.at(i), w1a.at(j), w2a.at(j));
	//						double residual[3];
	//						//cost_3d.push_back(f);
	//						ceres::CostFunction* c;
	//						if (idx1 == 0) {
	//							f->set(NULL, optt_bend[idx1], optt_bend[idx2-1], optt_bend[idx2]);
	//							c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_wrap_3a, ceres::CENTRAL, 3,3, 3, 3>(new BADepthKnownCostFunc_PrjT_wrap_3a(f));
	//							f->outlierRejection(thres3d);
	//							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[idx1], optt_bend[idx2]);
	//							count3d++;
	//							hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//						}
	//						else {
	//							f->set(optt_bend[idx1 - 1], optt_bend[idx1], optt_bend[idx2 - 1], optt_bend[idx2]);
	//							c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_wrap_3, ceres::CENTRAL, 3,3, 3, 3, 3>(new BADepthKnownCostFunc_PrjT_wrap_3(f));
	//							f->outlierRejection(thres3d);
	//							problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optt_bend[idx1 - 1], optt_bend[idx1], optt_bend[idx2 - 1], optt_bend[idx2]);
	//							count3d++;
	//							hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//						}
	//					}



	//				}

	//			}
	//		}
	//	}
	//	continue;
	//	//backward
	//	for (int j = i - 1;j >= max(0, i - frame_thres);j--) {
	//		if (i <= anchor_under)continue;
	//		Matrix4d trFrame = _6dof2m(framePos.at(sampledFrame.at(j))).inverse()*_6dof2m(framePos.at(sampledFrame.at(i)));
	//		next3Pts.clear();
	//		for (auto w3p : originW3p.at(i)) {
	//			Vector3d nw3p = trFrame.block(0, 0, 3, 3)*w3p + trFrame.block(0, 3, 3, 1);
	//			double ix, iy;
	//			omniTrans(nw3p(0), nw3p(1), nw3p(2), iy, ix, image1g.size().height);
	//			cv::Point2f projp(ix, iy);
	//			next3Pts.push_back(projp);
	//		}

	//		cv::calcOpticalFlowPyrLK(images_g.at(i), images_g.at(j), origin3P.at(i), next3Pts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//		for (int k = 0;k < originW3p.at(i).size();k++) {
	//			if (status.at(k) == '\1'&& err.at(k) < 40) {

	//				Vector3d nxt3v;
	//				rev_omniTrans(next3Pts.at(k).x, next3Pts.at(k).y, images_g.at(i).size().width, images_g.at(i).size().height, nxt3v);

	//				if (j <= anchor_under) {

	//					BADepthKnownCostFunc_PrjT_anckor_rev2 *f;

	//					if (j == 0) f = new BADepthKnownCostFunc_PrjT_anckor_rev2(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasT[i - 1],w1a.at(i), w2a.at(i));
	//					else {
	//						double ancm[6];
	//						ancm[0] = optParasR[j - 1][0];
	//						ancm[1] = optParasR[j - 1][1];
	//						ancm[2] = optParasR[j - 1][2];
	//						ancm[3] = optParasT[j - 1][0];
	//						ancm[4] = optParasT[j - 1][1];
	//						ancm[5] = optParasT[j - 1][2];
	//						f = new BADepthKnownCostFunc_PrjT_anckor_rev2(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasT[i - 1], ancm, w1a.at(i), w2a.at(i));
	//					}
	//					//cost_3d_ar.push_back(f);
	//					f->set(optt_bend[ctl_pt_idx.at(i) - 1], optt_bend[ctl_pt_idx.at(i)]);
	//					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT_anckor_rev2, ceres::CENTRAL, 3,3, 3>(f);
	//					f->outlierRejection(thres3d);
	//					problem.AddResidualBlock(c, new  ceres::CauchyLoss(2e-3), optt_bend[ctl_pt_idx.at(i) - 1], optt_bend[ctl_pt_idx.at(i)]);
	//					count3d++;
	//					hist[(i)*(sampledFrame.size()) + (j)] += 1;

	//				}
	//				else {
	//					BADepthKnownCostFunc_PrjT *f = new BADepthKnownCostFunc_PrjT(originW3p.at(i).at(k), nxt3v, weight3d, optParasR[i - 1], optParasR[j - 1]);
	//					cost_3d.push_back(f);
	//					f->set(optParasT[i - 1], optParasT[j - 1]);
	//					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BADepthKnownCostFunc_PrjT, ceres::CENTRAL, 3, 3, 3>(f);
	//					f->outlierRejection(thres3d);
	//					problem.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasT[i - 1], optParasT[j - 1]);
	//					count3d++;
	//					hist[(i)*(sampledFrame.size()) + (j)] += 1;
	//				}
	//			}
	//		}
	//	}
	//}
	histidx = 0;
	returnlog << endl << endl;
	for (auto itr = hist.begin();hist.end() != itr;itr++) {
		returnlog << *itr << ",";
		histidx++;
		if (histidx == sampledFrame.size()) {
			histidx = 0;
			returnlog << endl;
		}

	}
	cout << "3d:" << count3d << "   2d:" << count << endl;

	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options optionsrough, optionsfine;
	optionsrough.max_num_iterations = 1e2;
	optionsrough.function_tolerance = 1e-5;
	optionsrough.parameter_tolerance = 1e-5;
	optionsrough.initial_trust_region_radius = 1e6;
	optionsrough.num_threads = 12;
	optionsfine.max_num_iterations = 1e3;
	optionsfine.function_tolerance = 1e-7;
	optionsfine.parameter_tolerance = 1e-7;
	optionsfine.initial_trust_region_radius = 1e6;
	optionsfine.num_threads = 12;
	ceres::Solver::Summary summary;
	double thres2dfine = 0.01*0.01, thres3dfine = 0.01*0.01;
	for (int i = 0;i < sampledFrame.size() - 1;i++) {
		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
	}
	ceres::Solve(optionsrough, &problem_d, &summary);
	for (auto itr : cost_2d)itr->outlierRejection(thres2dfine);
	for (auto itr : cost_2d_a)itr->outlierRejection(thres2dfine);
	for (auto itr : cost_2d_a2)itr->outlierRejection(thres2dfine);
	ceres::Solve(optionsfine, &problem_d, &summary);
	for (int i = 1;i < sampledFrame.size();i++) {
		int idx=ctl_pt_idx.at(i);
		double w1 = w1a.at(i);
		double w2 = w2a.at(i);
		if (idx == -1)continue;
		else if (idx == 0) {

			double* paramend = optParas_bend[idx];
			Vector4d q1; q1 << 0,0,0, 1;
			q1 = q1.normalized();
			Vector4d q2; q2 << paramend[0], paramend[1], paramend[2], 1;
			q2 = q2.normalized();
			Vector4d qmix = w1 * q1 + w2 * q2;
			qmix = qmix.normalized();
			Matrix3d dR = q2dcm(qmix);

			Vector3d t1, t2;
			t1 << 0,0,0;
			t2 << paramend[3], paramend[4], paramend[5];
			Vector3d dt = w1 * t1 + w2 * t2;
			Vector3d t;t << optParasT[i - 1][0], optParasT[i - 1][1], optParasT[i - 1][2];
			Matrix3d R = axisRot2R(optParasR[i-1][0], optParasR[i-1][1], optParasR[i-1][2]);
			R = dR * R;
			R2axisRot(R, optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
			t = dt +  t;
			optParasT[i - 1][0] = t(0);
			optParasT[i - 1][1] = t(1);
			optParasT[i - 1][2] = t(2);

		}
		else {
			double* parambase=optParas_bend[idx-1];
			double* paramend = optParas_bend[idx];
			Vector4d q1; q1 << parambase[0], parambase[1], parambase[2], 1;
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
			Vector3d t;t << optParasT[i - 1][0], optParasT[i - 1][1], optParasT[i - 1][2];
			Matrix3d R = axisRot2R(optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
			R = dR * R;
			R2axisRot(R, optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
			t = dt + t;
			optParasT[i - 1][0] = t(0);
			optParasT[i - 1][1] = t(1);
			optParasT[i - 1][2] = t(2);
		}
		returnlog << optParasT[i-1][0] << "," << optParasT[i - 1][1] << "," << optParasT[i - 1][2] << endl;
	}
//	ceres::Solve(optionsrough, &problem, &summary);
//	for (auto itr : cost_3d)itr->outlierRejection(thres3dfine);
//	for (auto itr : cost_3d_a)itr->outlierRejection(thres3dfine);
//	for (auto itr : cost_3d_ar)itr->outlierRejection(thres3dfine);
//	ceres::Solve(optionsfine, &problem, &summary);
//	for (int i = 0;i < sampledFrame.size() - 1;i++) {
//		cout << optParasT[i][0] << "," << optParasT[i][1] << "," << optParasT[i][2] << endl;
//	}
//	for (int i = 1;i < sampledFrame.size();i++) {
//		int idx = ctl_pt_idx.at(i);
//		double w1 = w1a.at(i);
//		double w2 = w2a.at(i);
//		if (idx == -1)continue;
//		else if (idx == 0) {
//		
//			double* paramend = optt_bend[idx];
//			Vector3d t1, t2;
//			t1 << 0, 0, 0;
//			t2 << paramend[3], paramend[4], paramend[5];
//			Vector3d dt = w1 * t1 + w2 * t2;
//			Vector3d t;t << optParasT[i - 1][0], optParasT[i - 1][1], optParasT[i - 1][2];
////			Matrix3d R = axisRot2R(optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
////			R = dR * R;
////			R2axisRot(R, optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
//			cout << dt << endl;
//			t = dt + t;
//			optParasT[i - 1][0] = t(0);
//			optParasT[i - 1][1] = t(1);
//			optParasT[i - 1][2] = t(2);
//
//		}
//		else {
//			double* parambase = optt_bend[idx - 1];
//			double* paramend = optt_bend[idx];
//
//			Vector3d t1, t2;
//			t1 << parambase[3], parambase[4], parambase[5];
//			t2 << paramend[3], paramend[4], paramend[5];
//			Vector3d dt = w1 * t1 + w2 * t2;
//			Vector3d t;t << optParasT[i - 1][0], optParasT[i - 1][1], optParasT[i - 1][2];
//			//Matrix3d R = axisRot2R(optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
//			//R = dR * R;
//			//R2axisRot(R, optParasR[i - 1][0], optParasR[i - 1][1], optParasR[i - 1][2]);
//			t = dt + t;
//			optParasT[i - 1][0] = t(0);
//			optParasT[i - 1][1] = t(1);
//			optParasT[i - 1][2] = t(2);
//		}
//		//returnlog << optParasT[i - 1][0] << "," << optParasT[i - 1][1] << "," << optParasT[i - 1][2] << endl;
//	}
	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);

	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	//position update
	int skippedNum = 1;
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < sampledFrame.size() - 1) {
			//interpolation for frame down sampling
			if (i == sampledFrame.at(updateIdx)/*baseFrame + (updateIdx)*skipNum*/) {

				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0] * mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(sampledFrame.at(updateIdx)))*mupdate*_6dof2m(framePos.at(sampledFrame.at(updateIdx + 1))).inverse();
				skippedNum = sampledFrame.at(updateIdx + 1) - sampledFrame.at(updateIdx);
				diffMatrixLocal = mupdate;
			}

			double rate = (i - sampledFrame.at(updateIdx)) / (double)skippedNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == sampledFrame.at(updateIdx + 1) - 1) {
				updateIdx++;
			}
		}
		_6dof temp = framePos.at(i);
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		framePos.at(i) = temp;
	}

	for (int i = 0;i < sampledFrame.size() - 1;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return returnlog.str();

}



void SensorFusionBA::BatchBA_R(int baseFrame, int frameNum, int skipNum) {
	cv::Mat image1 = cv::imread(inputs.imgFileNameList.at(baseFrame));
	cv::Mat image1g;
	cv::cvtColor(image1, image1g, cv::COLOR_RGB2GRAY);
	cv::resize(image1g, image1g, cv::Size(2048, 1024));
	cv::resize(image1, image1, cv::Size(2048, 1024));
	double distUpper = 15.0;
	double distLower = 1.5;
	//stage 1, image-image correspondence

	vector<cv::Point2f> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	vector<cv::Point2f> next3Pts;
	vector<vector<cv::Point2f>> origin3P(frameNum + 1);
	vector <vector<Vector3d>> originW3p(frameNum + 1);//3d points in base camera frame coordinates.

	cv::goodFeaturesToTrack(image1g, originP, 2000, 1e-20, 10);

	nextPts = vector<cv::Point2f>(originP);

	//3d projection
	double startTime = inputs.imTimeStamp[baseFrame] - 1 / 32.0;
	inputs.psl->seekByTime(startTime);
	float* dat;
	int baseframe_ = baseFrame;
	int framecnt = 0;

	//while (inputs.psl->getNextPointData(dat)) {
	//	Vector3d p, pt;p << dat[0], dat[1], dat[2];
	//	double ref = dat[3];
	//	double t = dat[4];
	//	if (t > inputs.imTimeStamp[baseFrame + framecnt * skipNum] + 1 / 32.0 || (baseFrame + framecnt * skipNum == framePos.size() - 1 && t > inputs.imTimeStamp[baseFrame + framecnt * skipNum])) {


	//		if (framecnt < frameNum) {
	//			framecnt++;
	//			double startTime_ = inputs.imTimeStamp[baseFrame + framecnt * skipNum] - 1 / 32.0;
	//			inputs.psl->seekByTime(startTime_);
	//			free(dat);
	//			continue;
	//		}
	//		else {
	//			free(dat);
	//			break;
	//		}

	//	}
	//	if (ref < 1e-2) {
	//		free(dat);
	//		continue;
	//	}
	//	if (p.norm() < distLower || p.norm() > distUpper) {
	//		free(dat);
	//		continue;
	//	}
	//	cv::Scalar color;color = cv::Scalar(255, 0, 0);
	//	if (t < inputs.imTimeStamp[baseFrame + framecnt * skipNum]) {
	//		Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
	//		Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum - 1));
	//		Matrix4d mdiff_ = mprev.inverse()*mbase;//mvt->mft
	//		Matrix4d mdiff = mdiff_.inverse();
	//		Matrix3d diffR = mdiff.block(0, 0, 3, 3);

	//		Vector4d q = dcm2q(diffR);
	//		Vector4d q_b;q_b << 0, 0, 0, 1;
	//		Vector3d diffT = mdiff.block(0, 3, 3, 1);
	//		double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum - 1];
	//		double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
	//		double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
	//		Vector4d q_ = (1 - rate)*q + rate * q_b;
	//		q_ = q_.normalized();
	//		Vector3d t_ = (1 - rate) * diffT;
	//		Matrix3d interpolatedR = q2dcm(q_);
	//		Vector3d interpolatedT = t_;
	//		//p - sensor
	//		pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
	//		//p - camera
	//		pt = interpolatedR * pt + interpolatedT;//
	//		//cout << interpolatedR <<endl<< interpolatedT.transpose()<< endl;
	//		color = cv::Scalar(255, 0, 0);
	//	}
	//	else {
	//		Matrix4d mbase = _6dof2m(framePos.at(baseFrame + framecnt * skipNum + 1));
	//		Matrix4d mprev = _6dof2m(framePos.at(baseFrame + framecnt * skipNum));
	//		Matrix4d mdiff = mprev.inverse()*mbase;//mft->mvt
	//		Matrix3d diffR = mdiff.block(0, 0, 3, 3);
	//		Vector4d q = dcm2q(diffR);
	//		Vector4d q_b;q_b << 0, 0, 0, 1;
	//		Vector3d diffT = mdiff.block(0, 3, 3, 1);
	//		double frameStartTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum];
	//		double frameEndTime = inputs.imTimeStamp[baseFrame + framecnt * skipNum + 1];
	//		double rate = (t - frameStartTime) / (frameEndTime - frameStartTime);
	//		Vector4d q_ = (1 - rate)*q_b + rate * q;
	//		Vector4d q_2 = q_.normalized();
	//		//cout << q.transpose() << "," << q_2.transpose() << endl;
	//		Vector3d t_ = rate * diffT;
	//		Matrix3d interpolatedR = q2dcm(q_2);
	//		Vector3d interpolatedT = t_;
	//		//p - sensor
	//		pt = inputs.extCalib.block(0, 0, 3, 3) * p + inputs.extCalib.block(0, 3, 3, 1);//
	//																					  //p - camera
	//		pt = interpolatedR * pt + interpolatedT;//
	//		color = cv::Scalar(0, 0, 255);
	//	}
	//	double ix, iy;
	//	omniTrans(pt(0), pt(1), pt(2), iy, ix, image1g.size().height);
	//	cv::Point2f projp(ix, iy);
	//	cv::circle(image1, projp, 0, color);
	//	origin3P.at(framecnt).push_back(projp);
	//	originW3p.at(framecnt).push_back(pt);
	//	free(dat);
	//}
	stringstream ss;
	ss << "image" << baseFrame << ".jpg";
	//cv::imwrite(ss.str(), image1);
	double** optParasT = (double**)malloc(sizeof(double*)*frameNum);
	double** optParasR = (double**)malloc(sizeof(double*)*frameNum);
	double* optScale = (double*)malloc(sizeof(double)*frameNum);
	double optScale_[1] = { 1.0 };
	double scale = 0;
	for (int i = 0;i < frameNum;i++) {
		Matrix4d trFrame = _6dof2m(framePos.at(baseFrame)).inverse()*_6dof2m(framePos.at(baseFrame + (i + 1) * skipNum));
		optParasT[i] = (double*)malloc(sizeof(double) * 3);
		optParasR[i] = (double*)malloc(sizeof(double) * 3);
		_6dof diffMot = m2_6dof(trFrame);
		cout << diffMot << endl;
		optParasR[i][0] = diffMot.rx;
		optParasR[i][1] = diffMot.ry;
		optParasR[i][2] = diffMot.rz;
		optParasT[i][0] = diffMot.x;
		optParasT[i][1] = diffMot.y;
		optParasT[i][2] = diffMot.z;
		optScale[i] = 1.0;
		if (i == 0)scale = trFrame.block(0, 3, 3, 1).norm();
	}

	ceres::Problem problem_d, problem, problem2;

	vector<cv::Mat> images_g;
	images_g.push_back(image1g);
	int count3d = 0;
	for (int i = 0;i < frameNum;i++) {
		cv::Mat image2 = cv::imread(inputs.imgFileNameList.at(baseFrame + (i + 1)*skipNum));
		cv::Mat image2g;

		cv::cvtColor(image2, image2g, cv::COLOR_RGB2GRAY);
		cv::resize(image2g, image2g, cv::Size(2048, 1024));
		//2d-2d
		images_g.push_back(image2g);
	}


	/*
	scaleAnckor* sa_func = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func);
	problem.AddResidualBlock(c_anc, NULL, optParasT[frameNum - 1]);*/
	scaleAnckor* sa_func2 = new scaleAnckor(optParasT[frameNum - 1][0] * optParasT[frameNum - 1][0] + optParasT[frameNum - 1][1] * optParasT[frameNum - 1][1] + optParasT[frameNum - 1][2] * optParasT[frameNum - 1][2]);
	ceres::CostFunction* c_anc2 = new ceres::NumericDiffCostFunction<scaleAnckor, ceres::CENTRAL, 1, 3>(sa_func2);
	//problem_d.AddResidualBlock(c_anc2, NULL, optParasT[frameNum - 1]);
	vector< map<int, cv::Point2f>> f_pts = pointTracking(images_g, optParasR);
	double** sfm_point = (double**)malloc(sizeof(double*)*f_pts.size());
	int idx = 0;

	for (auto itr : f_pts) {
		sfm_point[idx] = (double*)malloc(sizeof(double) * 3);

		Matrix3d R1, R2;
		Vector3d v1, v2, c1, c2, c12;//in the world coordinate
		bool isFirst = true;
		bool isMulti = false;
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			else {
				if (isFirst) {
					if (itr2.first == 0) {
						R1 = Matrix3d::Identity();
						c1 << 0, 0, 0;
					}
					else {
						R1 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
						c1 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					}
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v1);
					isFirst = false;
				}
				else {
					R2 = axisRot2R(optParasR[itr2.first - 1][0], optParasR[itr2.first - 1][1], optParasR[itr2.first - 1][2]);
					c2 << optParasT[itr2.first - 1][0], optParasT[itr2.first - 1][1], optParasT[itr2.first - 1][2];
					rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, v2);
					isMulti = true;
				}
			}
		}
		if (!isMulti) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		c12 = c2 - c1;
		v1 = R1 * v1;
		v2 = R2 * v2;
		double s, t;
		s = (c12.dot(v1) - (c12.dot(v2))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));
		t = -(c12.dot(v2) - (c12.dot(v1))*(v1.dot(v2))) / (1 - v1.dot(v2)*v1.dot(v2));

		Vector3d p = (c1 + s * v1 + c2 + t * v2) / 2;//point calculated from all of lays from camera

		Vector3d pd = R1.transpose()*(p - c1);
		Vector3d err = v1.cross(pd.normalized());
		if (err.norm() > 0.005) {
			sfm_point[idx][0] = 0;
			sfm_point[idx][1] = 0;
			sfm_point[idx][2] = 0;
			idx++;
			continue;
		}
		POINT_3D m_p = { p(0),p(1),p(2) };

		sfm_point[idx][0] = p(0);
		sfm_point[idx][1] = p(1);
		sfm_point[idx][2] = p(2);
		idx++;
	}
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm_b.ply");
		bp.release();
	}

	idx = 0;
	int count = 0;
	double weight2d[1];weight2d[0] = 1.0;
	for (auto itr : f_pts) {
		/*	sfm_point[idx] = (double*)malloc(sizeof(double) * 3);
			sfm_point[idx][0] = 1;
			sfm_point[idx][1] = 1;
			sfm_point[idx][2] = 1;*/
		if (sfm_point[idx][0] == 0 && sfm_point[idx][1] == 0 && sfm_point[idx][2] == 0) {
			idx++;
			continue;
		};
		for (auto itr2 : itr) {
			if (itr2.second.x < 0) {
				continue;
			}
			Vector3d nxt3v;
			rev_omniTrans(itr2.second.x, itr2.second.y, image1g.size().width, image1g.size().height, nxt3v);
			if (/*itr2.first < 0*/itr2.first == 0) {
				BaFeatureProjCostFunction_Ancker* f = new BaFeatureProjCostFunction_Ancker(nxt3v, weight2d);

				double residual[3];

				f->operator()(sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.01) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunction_Ancker, ceres::CENTRAL, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), sfm_point[idx]);
					count++;
				}
				else {
					delete f;
				}
			}
			else {
				BaFeatureProjCostFunctionR* f = new BaFeatureProjCostFunctionR(nxt3v, weight2d, optParasT[itr2.first - 1]);

				double residual[3];

				f->operator()(optParasR[itr2.first - 1], sfm_point[idx], residual);
				if (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2] < 0.01) {
					ceres::CostFunction* c = new ceres::NumericDiffCostFunction<BaFeatureProjCostFunctionR, ceres::CENTRAL, 3, 3, 3>(f);
					problem_d.AddResidualBlock(c, new ceres::CauchyLoss(2e-3), optParasR[itr2.first - 1], sfm_point[idx]);

					count++;
				}
				else {
					delete f;
				}
			}

		}
		idx++;
	}

	double weight3d[1];
	//cv::calcOpticalFlowPyrLK(image1g, image2g, originP, nextPts, status, err, cv::Size(21, 21), 3, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
	//2d-3d

	//point selection
	//for (int i = 0;i < frameNum + 1;i++) {
	//	vector<int> selected;
	//	goodFeatureToTrack_onProjection(images_g.at(i), origin3P.at(i), selected, 10, 0);
	//	vector<cv::Point2f> fps;
	//	vector<Vector3d> wps;
	//	for (auto itr : selected) {
	//		cv::Point2f pt = origin3P.at(i).at(itr);
	//		Vector3d pt3w = originW3p.at(i).at(itr);
	//		fps.push_back(pt);
	//		wps.push_back(pt3w);
	//	}
	//	origin3P.at(i) = vector<cv::Point2f>(fps);
	//	originW3p.at(i) = vector<Vector3d>(wps);
	//}



	cout << "3d:" << count3d << "   2d:" << count << endl;

	weight3d[0] = 1.0;// / sqrt(count3d);
	weight2d[0] = 1.0;// / sqrt(count);
	ceres::Solver::Options options;
	options.max_num_iterations = 1e3;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
	ceres::Solver::Summary summary;
	options.num_threads = 12;
	ceres::Solve(options, &problem_d, &summary);
	//ceres::Solve(options, &problem, &summary);
	//ceres::Solve(options, &problem2, &summary);

	//for (auto itr : costFncs) {
	//	double residual[3];
	//	itr->operator()(optScale,residual);
	//	//cout << residual[0] * residual[0]+ residual[1] * residual[1] + residual[2] * residual[2] << endl;
	//
	//}

	Matrix4d diffMatrix = Matrix4d::Identity();
	Matrix4d diffMatrixLocal = Matrix4d::Identity();
	int updateIdx = 0;
	{
		float* vtx;
		float* rfl;
		vtx = (float*)malloc(sizeof(float)*idx * 3);
		rfl = (float*)malloc(sizeof(float)*idx);
		for (int i = 0;i < idx;i++) {
			vtx[i * 3] = sfm_point[i][0];
			vtx[i * 3 + 1] = sfm_point[i][1];
			vtx[i * 3 + 2] = sfm_point[i][2];
			rfl[i] = 0.5;
		}
		BasicPly bp;

		bp.setVertecesPointer(vtx, idx);
		bp.setReflectancePointer(rfl, idx);
		bp.writePlyFileAuto("sfm.ply");
		bp.release();

	}
	for (int i = 0;i < idx;i++) {
		free(sfm_point[i]);
	}
	free(sfm_point);

	double scale_ = sqrt(optParasT[0][0] * optParasT[0][0] + optParasT[0][1] * optParasT[0][1] + optParasT[0][2] * optParasT[0][2]);
	cout << scale * (1 / scale_) << endl;
	for (int i = baseFrame;i < framePos.size();i++) {
		Matrix4d baseDiffMatrix = diffMatrix;
		if (updateIdx < frameNum) {
			if (i == baseFrame + (updateIdx)*skipNum) {
				diffMatrix = diffMatrixLocal * diffMatrix;
				baseDiffMatrix = diffMatrix;
				_6dof m1;
				Matrix4d morigin, mupdate;
				if (updateIdx == 0) {

					m1 = { 0,0,0,0,0,0 };
				}
				else {
					m1.rx = optParasR[updateIdx - 1][0];m1.ry = optParasR[updateIdx - 1][1]; m1.rz = optParasR[updateIdx - 1][2]; m1.x = optParasT[updateIdx - 1][0];;m1.y = optParasT[updateIdx - 1][1];m1.z = optParasT[updateIdx - 1][2];

				}
				//morigin = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum)).inverse()*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum));
				_6dof m2; m2.rx = optParasR[updateIdx][0];m2.ry = optParasR[updateIdx][1]; m2.rz = optParasR[updateIdx][2]; m2.x = optParasT[updateIdx][0];;m2.y = optParasT[updateIdx][1];m2.z = optParasT[updateIdx][2];
				cout << m2_6dof(morigin) << endl;
				cout << m2 << endl;
				cout << optScale_[0] << endl;
				mupdate = _6dof2m(m1).inverse()*_6dof2m(m2);
				mupdate.block(0, 3, 3, 1) = optScale_[0] * mupdate.block(0, 3, 3, 1);
				mupdate = _6dof2m(framePos.at(baseFrame + (updateIdx)*skipNum))*mupdate*_6dof2m(framePos.at(baseFrame + (updateIdx + 1)*skipNum)).inverse();
				diffMatrixLocal = mupdate;
			}

			double rate = ((i - baseFrame) % skipNum) / (double)skipNum;
			Matrix3d diffR = diffMatrixLocal.block(0, 0, 3, 3);
			Vector4d q, qb;qb << 0, 0, 0, 1;
			q = dcm2q(diffR);
			q = (1 - rate)*qb + rate * q;
			q = q.normalized();
			Matrix3d diffR_ = q2dcm(q);
			Vector3d diffT = rate * diffMatrixLocal.block(0, 3, 3, 1);
			Matrix4d m;
			m.block(0, 0, 3, 3) = diffR_;
			m.block(0, 3, 3, 1) = diffT;
			m.block(3, 0, 1, 4) << 0, 0, 0, 1;
			baseDiffMatrix = m * baseDiffMatrix;
			if (i == baseFrame + (updateIdx + 1)*skipNum - 1) {
				updateIdx++;
			}
			//cout << i << endl;
			//cout << baseDiffMatrix << endl << endl;
		}
		_6dof temp = framePos.at(i);
		//cout << i << endl;
		//cout << baseDiffMatrix << endl << endl;
		Matrix4d updated = baseDiffMatrix * _6dof2m(temp);

		temp = m2_6dof(updated);
		//cout << temp << endl;
		framePos.at(i) = temp;
	}

	for (int i = 0;i < frameNum;i++) {
		free(optParasR[i]);
		free(optParasT[i]);
	}
	free(optParasR);
	free(optParasT);

	return;

}



void pointMatching(vector<cv::Mat> images) {
	
	auto algorithm = cv::AKAZE::create();
	vector<vector<cv::KeyPoint>> keypoints;
	vector<cv::Mat> descriptors;
	//feature point detection
	for (int i = 0;i < images.size();i++) {
		cv::Mat img = images.at(i);
		cout << "image " << i << "keypoint detection" << endl;
		vector<cv::KeyPoint> keypoint;
		cv::Mat descriptor;
		cv::Mat miniim;
		cv::resize(img, miniim, cv::Size(1000, 500));
		algorithm->detect(miniim, keypoint);
		algorithm->compute(miniim, keypoint, descriptor);
		keypoints.push_back(keypoint);
		descriptors.push_back(descriptor);
	}


	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
	std::vector<cv::DMatch> match12;
	//matching
	//blute force
	for (int i = 0;i < images.size();i++) {
		for (int j = i + 1;j < images.size();j++) {
			matcher->match(descriptors.at(i), descriptors.at(j), match12);
		}
	}
	
	vector<Vector3d> bearingVectors1, bearingVectors2;

	//synthesis
	vector<map<int,cv::Point2f>> ret;
	
}

vector<map<int, cv::Point2f>> pointTracking(vector<cv::Mat> images, double** mots,int skipNum,cv::Mat mask_,double circleParam,int fItv) {

	vector<map<int, cv::Point2f>> ret;
	vector<vector<cv::Point2f>> originP, nextPts;
	vector<float> err;
	vector<uchar> status;
	double circlepara = images.at(0).size().width/ circleParam;
	int frameth = 10000;//160 / skipNum;
	int addFrameItv = fItv;
	//feature point detection

	int offset = 0;
	vector<bool> rejection_array;
	for (int i = 0;i < images.size() - 1;i++) {		//i: current image
		if(i%addFrameItv==0){
			cv::Mat mask(images.at(i).size(), CV_8UC1);
			if (mask_.cols > 0) {
				mask = mask_.clone();
			}
			else {
				memset(mask.data, (unsigned char)255, images.at(i).size().height*images.at(i).size().width);
			}
			int idx = 0;
			int point_idx=0;
			for (auto itr_ : nextPts) {
				if (idx < i - frameth) {
					point_idx += itr_.size();
					idx++;
					continue;
				}
				for (auto itr : itr_) {
					if (rejection_array.at(point_idx)) {
						cv::circle(mask, itr, circlepara/3, cv::Scalar(0), -1);
					}
					point_idx++;
				}
				idx++;
			}

			vector<cv::Point2f> addedP;
			try {
				cv::goodFeaturesToTrack(images.at(i), addedP, 4000, 1e-20, circlepara, mask, 3, true);
			}
			catch(cv::Exception e){
				std::cout << e.msg << std::endl;
				exit(-1);
			}

			originP.push_back(vector<cv::Point2f>(addedP));
			nextPts.push_back(vector<cv::Point2f>(addedP));
		}
		else {
			originP.push_back(vector<cv::Point2f>());
			nextPts.push_back(vector<cv::Point2f>());
		}
		//vector<map<int, cv::Point2f>> added;
		Matrix3d i_rot;
		Matrix3d mm1, mm2;
		mm2 = axisRot2R(mots[i][0], mots[i][1], mots[i][2]);
		if (i > 0) {
			mm1 = axisRot2R(mots[i-1][0], mots[i-1][1], mots[i-1][2]);
		}
		else {
			mm1 = Matrix3d::Identity();
		}
		i_rot = mm2.transpose()*mm1;
		if (i - frameth > 0) {
			offset+=originP.at(i - frameth - 1).size();
		}
		int idx = offset;
		int checksum = 0;
		int allsum = 0;

		for (int j = max(i- frameth,0);j < i+1;j++) {//j: origin frame of feature point
			if (originP.at(j).size() == 0)continue;

			//nextPts.clear();

			vector<cv::Point2f> nextP;
			for (int k = 0;k < nextPts.at(j).size();k++) {
				Vector3d bv;
				rev_omniTrans(nextPts.at(j).at(k).x, nextPts.at(j).at(k).y, images.at(i).cols, images.at(i).rows, bv);
				bv = i_rot * bv;
				double ix, iy;
				omniTrans(bv(0), bv(1), bv(2),iy ,ix , images.at(j).rows);
				nextP.push_back(cv::Point2f(ix,iy));
			}
			nextPts.at(j) = vector<cv::Point2f>(nextP);
			cv::calcOpticalFlowPyrLK(images.at(j), images.at(i+1), originP.at(j), nextPts.at(j), status, err, cv::Size(21, 21), 1, cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
			allsum += originP.at(j).size();
			for (int k = 0;k < originP.at(j).size();k++) {
				if (j == originP.size() - 1) {
					map<int, cv::Point2f> newm;
					newm[j] = originP.at(j).at(k);
					newm[j + 1] = nextPts.at(j).at(k);
					ret.push_back(newm);
					checksum++;
					if (status.at(k) == '\1'&& err.at(k) < 10) {
						rejection_array.push_back(true);
					}
					else {
						rejection_array.push_back(false);
					}
				}
				else {
					if (status.at(k) == '\1'&& err.at(k) < 10) {
						ret.at(idx)[i + 1] = nextPts.at(j).at(k);
						checksum++;
					}
					else {
						ret.at(idx)[i + 1] = cv::Point2f(-1,-1);
						rejection_array.at(idx) = false;
					}
				}
				idx++;
			}
		}
		cout << checksum <<"/"<<allsum<< endl;
	}

	return ret;
}


void AdjustableHuberLoss::Evaluate(double s, double rho[3]) const {
	if (s > b_) {
		// Outlier region.
		// 'r' is always positive.
		const double r = sqrt(s);
		rho[0] = 2 * a_ * r - b_;
		rho[1] = a_ / r;
		rho[2] = -rho[1] / (2 * s);
	}
	else {
		// Inlier region.
		rho[0] = s;
		rho[1] = 1;
		rho[2] = 0;
	}
}