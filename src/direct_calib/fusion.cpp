//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <direct_calib/fusion.h>



void fusionscan::setscan(std::string scaninfo) {
	std::ifstream ifs(scaninfo);
	nlohmann::json config;
	ifs >> config;
	ifs.close();
	ptnum = config["lidar_pn"].get<long long>();

	ifs.open(config["lidar"].get<std::string>(),std::ios::binary);
	allpt = (float*)malloc(sizeof(float) * ptnum * 5);
	ifs.read((char*)allpt, sizeof(float) * ptnum * 5);
	ifs.close();

	filelist = config["camera_list"].get<std::vector<std::string>>();
	imgs = std::vector<cv::Mat>(filelist.size());
	mask = std::vector<cv::Mat>(filelist.size());
	edgeMat = std::vector<cv::Mat>(filelist.size());

	image_timestamp = config["camera_ts"].get<std::vector<double>>();

	startFrame = config["first_frame"].get<int>();
	endFrame = config["end_frame"].get<int>();

	sync_offset = config["sync_offset"].get<double>();
};

void fusionscan::setCamera(nlohmann::json baseConf,double c2f) {

	nlohmann::json caminfo = baseConf["camerainfo"].get<nlohmann::json>();
	std::string type = caminfo["type"].get<std::string>();

	if (type.compare("panoramic") == 0) {
		int w = caminfo["w"].get<int>() * c2f;
		int h = caminfo["h"].get<int>() * c2f;

		if (w % 2 == 1)w++;
		if (h % 2 == 1)h++;
		cam = new omnidirectionalCamera(w, h);

		pr = new PanoramaRenderer();
		pr->createContext(w, h);
		if (glewInit() != GLEW_OK) {
			std::cout << "glewInit() error" << std::endl;
			exit(-1);
		};
		pr->setEquirectangular();
		pr->setDepthFarClip(20.0);

	}
	else if (type.compare("perspective") == 0) {
		//intrinsic
		nlohmann::json camera_intrinsic;
		std::ifstream ifs(caminfo["intrinsic"].get<std::string>());
		ifs >> camera_intrinsic;
		ifs.close();

		cam = new perspectiveCamera(camera_intrinsic["cx"].get<double>(), camera_intrinsic["cy"].get<double>(), camera_intrinsic["fx"].get<double>(), camera_intrinsic["fy"].get<double>());
	}
	else {
		std::cout << "invalid camera type" << std::endl;
		exit(0);
	}
}

void fusionscan::setInitialMotion(std::string motionFile) {
	std::ifstream ifs(motionFile);
	nlohmann::json config;
	ifs >> config;
	ifs.close();

	std::vector<std::vector<double>> org_motion = config["_cameramove"].get<std::vector<std::vector<double>>>();
	cammot_timestamp = config["_camera_ts"].get <std::vector<double>>();
	
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
		Eigen::Vector4d q(qx,qy,qz,qw);
		Eigen::Vector3d t(org_motion[i][3], org_motion[i][4], org_motion[i][5]);
		Eigen::Matrix3d R = cvl_toolkit::conversion::q2dcm(q);
		Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
		M.block(0, 0, 3, 3) = R;
		M.block(0, 3, 3, 1) = t;
		cameramotion.push_back(M);
	}
	motionTranslationWeight.clear();
	motionRotationWeight.clear();
	double tsum = 0,qsum=0;

	//weight for motion deformation
	for (int i = 0; i < cameramotion.size() - 1; i++) {
		Eigen::Matrix4d Mprev = cameramotion.at(i);
		Eigen::Matrix4d Mcur = cameramotion.at(i + 1);
		Eigen::Matrix4d Mdiff = Mprev.inverse()*Mcur;
		double tdiff = Mdiff.block(0, 3, 3, 1).norm();
		Eigen::Matrix3d Mdiffrot = Mdiff.block(0, 0, 3, 3);
		Eigen::Vector4d q = cvl_toolkit::conversion::dcm2q(Mdiffrot);
		double qdiff = q.segment(0, 3).norm();
		motionTranslationWeight.push_back(tdiff);
		motionRotationWeight.push_back(qdiff);
		tsum += tdiff;
		qsum += qdiff;
	}

	//normalize
	for (int i = 0; i < motionTranslationWeight.size(); i++) {
		motionTranslationWeight.at(i) = tsum == 0 ? 1.0 / motionTranslationWeight.size() : motionTranslationWeight.at(i) / tsum;
		motionRotationWeight.at(i) = qsum == 0 ? 1.0 / motionRotationWeight.size() : motionRotationWeight.at(i) / tsum;
	}
	deformation(Eigen::Vector4d(0, 0, 0, 1), Eigen::Vector3d(0, 0, 0),1.0);
}


void fusionscan::visualize(const column_vector& m,std::string base,int outNum) {
	// motion deformation
	Eigen::Vector4d deformrot(m(0), m(1), m(2), 1);
	deformrot.normalize();
	deformation(deformrot, Eigen::Vector3d(m(3), m(4), m(5)),m(12));
	Eigen::Vector4d extqd(m(6), m(7), m(8), 1);
	extqd.normalize();
	Eigen::Matrix3d extrot = cvl_toolkit::conversion::q2dcm(extqd);
	extrot = extR * extrot;
	Eigen::Vector3d exttd(m(9), m(10), m(11));
	exttd = extt + exttd;
	//reconstruction
	std::vector<Eigen::Vector4d> pc;
	std::vector<double> timestamps;
	double stTS = image_timestamp.at(0);
	double endTS = image_timestamp.at(image_timestamp.size() - 2);

	{
		int downsamp = 15;
		for (int i = 0; i < usedPtIdx.size();i++) {
			long long fIdx= usedPtIdx.at(i);
			Eigen::Vector3d v(allpt[fIdx * 5 + 0], allpt[fIdx * 5 + 1], allpt[fIdx * 5 + 2]);
			if (v.norm() < 0.5)continue;
			double ts = allpt[fIdx * 5 + 4];

			Eigen::Matrix4d motion = timeMotion(ts+sync_offset);

			Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
			Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

			v = Rot * (extrot * v + exttd) + trans;
			Eigen::Vector4d vt(v(0), v(1), v(2), allpt[fIdx * 5 + 3]);
			pc.push_back(vt);
			timestamps.push_back(ts + sync_offset);
			fIdx++;
		}
	}
	std::vector<uchar> color;
	//std::vector<Eigen::Matrix4d> tf_cams, std::vector<uchar>& color, std::vector<double> timestamps, std::vector<double> imtimestamp, camera*cam, double thres
	
	colorMapping(pc, color, timestamps);

	for (int i = 0; i < outNum; i++) {
		int uidx; 
		if (outNum > 1) {
			uidx = (i*(usedIdx.size()-1))/ (outNum - 1);
		}
		else {
			uidx = 0;
		}
		std::string savePath = base + std::to_string(usedIdx.at(uidx)) + ".png";
		imageProjection(savePath, imgs.at(usedIdx.at(uidx)), pc, color, deformedMotion.at(usedIdx.at(uidx)),
			findIdx(stTS), findIdx(endTS), cam);


	}
	std::vector<float> vtx;
	std::vector<uchar> colorrgba;
	for (int idx = 0; idx < pc.size(); idx++) {
		vtx.push_back(pc.at(idx)(0));
		vtx.push_back(pc.at(idx)(1));
		vtx.push_back(pc.at(idx)(2));
		colorrgba.push_back(color[idx * 3 + 2]);
		colorrgba.push_back(color[idx * 3 + 1]);
		colorrgba.push_back(color[idx * 3 + 0]);
		colorrgba.push_back(255);
	}
	cvl_toolkit::plyObject po;
	po.setVertecesPointer(vtx.data(), vtx.size() / 3);
	po.setRgbaPointer(colorrgba.data(), vtx.size() / 3);
	po.writePlyFile(base + "_model.ply");
	
	if (bEdge) {
		{
			std::vector<Eigen::Vector4d> pc_edge;
			std::vector<double> timestamps_edge;

			for (int i = 0; i < usedEdgePtIdx.size(); i++) {
				long long fIdx = usedEdgePtIdx.at(i);
				Eigen::Vector3d v(allpt_edge[fIdx * 5 + 0], allpt_edge[fIdx * 5 + 1], allpt_edge[fIdx * 5 + 2]);
				double ts = allpt_edge[fIdx * 5 + 4];
				Eigen::Matrix4d motion = timeMotion(ts + sync_offset);

				Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
				Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

				v = Rot * (extrot * v + exttd) + trans;
				Eigen::Vector4d vt(v(0), v(1), v(2), allpt_edge[fIdx * 5 + 3]);
				pc_edge.push_back(vt);
				timestamps_edge.push_back(ts);
				fIdx++;
			}

			std::vector<float> vtx_edge;
			std::vector<float> edgeness;
			for (int idx = 0; idx < pc_edge.size(); idx++) {
				vtx_edge.push_back(pc_edge.at(idx)(0));
				vtx_edge.push_back(pc_edge.at(idx)(1));
				vtx_edge.push_back(pc_edge.at(idx)(2));
				edgeness.push_back(pc_edge.at(idx)(3));
			}
			cvl_toolkit::plyObject po2;
			po2.setVertecesPointer(vtx_edge.data(), vtx_edge.size() / 3);
			po2.setReflectancePointer(edgeness.data(), vtx_edge.size() / 3);
			po2.writePlyFile(base + "_edge.ply");

		}
	}
}



long long fusionscan::findIdx(float qt) {
	long long maxIdx = ptnum;
	__int64 startIdx = 0, endIdx = maxIdx - 1, centerIdx = startIdx;
	while (endIdx - startIdx > 1) {
		centerIdx = (endIdx + startIdx) / 2;

		double tc = allpt[centerIdx * 5 + 4];
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


long long fusionscan::edgeFindIdx(float qt) {
	long long maxIdx = ptnum_edge;
	__int64 startIdx = 0, endIdx = maxIdx - 1, centerIdx = startIdx;
	while (endIdx - startIdx > 1) {
		centerIdx = (endIdx + startIdx) / 2;

		double tc = allpt_edge[centerIdx * 5 + 4];
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

Eigen::Matrix4d fusionscan::timeMotion(double qt) {
	//get time
	__int64 startIdx = 0, endIdx = cammot_timestamp.size() - 1, centerIdx = startIdx;
	
	while (endIdx - startIdx > 1) {
		centerIdx = (endIdx + startIdx) / 2;

		double tc = cammot_timestamp[centerIdx];
		if (tc >= qt) {
			endIdx = centerIdx;
		};
		if (tc < qt) {
			startIdx = centerIdx;
		};
	}
	centerIdx = (endIdx + startIdx) / 2 + 1;
	
	double t1 = cammot_timestamp.at(centerIdx);
	Eigen::Matrix4d m1 = deformedMotion.at(centerIdx);
	int prevIdx = centerIdx == 0 ? 0 : centerIdx-1;
	double t2 = cammot_timestamp.at(prevIdx);
	Eigen::Matrix4d m2 = deformedMotion.at(prevIdx);
	
	//out of range
	if (t1 < qt) return m1;
	if (t2 > qt) return m2;
	//interpolation
	double w = (qt - t2)/(t1 - t2);//w of m1
	
	Eigen::Matrix3d R1 = m1.block(0,0,3,3);
	Eigen::Vector4d q1 = cvl_toolkit::conversion::dcm2q(R1);
	Eigen::Matrix3d R2 = m2.block(0, 0, 3, 3);
	Eigen::Vector4d q2 = cvl_toolkit::conversion::dcm2q(R2);
	Eigen::Vector3d tv1 = m1.block(0, 3, 3, 1);
	Eigen::Vector3d tv2 = m2.block(0, 3, 3, 1);

	Eigen::Vector4d q3 = w * q1 + (1 - w)*q2;
	Eigen::Matrix3d R3 = cvl_toolkit::conversion::q2dcm(q3);
	Eigen::Vector3d tv3 = w * tv1 + (1 - w)*tv2;
	Eigen::Matrix4d m3 = Eigen::Matrix4d::Identity();
	m3.block(0,0,3,3) = R3;
	m3.block(0, 3, 3, 1) = tv3;
	return m3;

}


void fusionscan::setImage(int usedimageNum,double c2f) {
	
	usedIdx.push_back(0);
	imgs.at(0) = cv::imread(filelist.at(0));
	//mask.at(0) = cv::Mat(imgs.at(0).size(),CV_32FC1,1.0);
	for (int i = 1; i < usedimageNum-1; i++) {
		int idx = filelist.size()*i/ usedimageNum;
		usedIdx.push_back(idx);
		imgs.at(idx) = cv::imread(filelist.at(idx));
		//mask.at(idx) = cv::Mat(imgs.at(0).size(), CV_32FC1, 1.0);
	}
	usedIdx.push_back(imgs.size()-1);
	imgs.at(imgs.size() - 1) = cv::imread(filelist.at(imgs.size() - 1));
//	mask.at(imgs.size() - 1) = cv::Mat(imgs.at(imgs.size() - 1).size(), CV_32FC1, 1.0);
	if (c2f < 1.0 && c2f >0.0) {
		for (int i = 0; i < usedimageNum; i++) {
			int j=usedIdx.at(i);
			int w = imgs.at(j).size().width*c2f;
			int h = imgs.at(j).size().height * c2f;
			if (w % 2 == 1)w++;
			if (h % 2 == 1)h++;

			cv::resize(imgs.at(j), imgs.at(j),cv::Size(w,h));
		}
		
	
	}
};

void fusionscan::setPoints(int downSample) {
	double stTS = image_timestamp.at(0);
	double endTS = image_timestamp.at(image_timestamp.size() - 2);

	{
		int downsamp = downSample;
		long long fIdx = findIdx(stTS - sync_offset);
		while (true) {
			double ts = allpt[fIdx * 5 + 4];
			if (ts + sync_offset > endTS) {
				break;
			}
			Eigen::Vector3d v(allpt[fIdx * 5 + 0], allpt[fIdx * 5 + 1], allpt[fIdx * 5 + 2]);
			if (v.norm() < 0.5) {

				fIdx += downSample;
				continue;
			};
			usedPtIdx.push_back(fIdx);
			fIdx += downSample;

		}
	}


};

void fusionscan::setInitCalib(nlohmann::json baseConf){
	
	std::ifstream ifs(baseConf["initial_extparam"].get<std::string>());
	nlohmann::json extjson;
	ifs >> extjson;
	ifs.close();

	std::vector<double> rotv = extjson["camera2lidar"]["rotation"].get<std::vector<double>>()
		, transv = extjson["camera2lidar"]["translation"].get<std::vector<double>>();

	; extR << rotv[0], rotv[1], rotv[2], rotv[3], rotv[4], rotv[5], rotv[6], rotv[7], rotv[8];
	; extt << transv[0], transv[1], transv[2];
}


double fusionscan::costComputation(const column_vector& m) {
	// motion deformation
	//double m_[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
	for (int i = 0; i < optvalue.size(); i++) {
		m_(optvalue[i]) = m(i);
	}

	Eigen::Vector4d deformrot(m_(0), m_(1), m_(2), 1);
	deformrot.normalize();
	deformation(deformrot, Eigen::Vector3d(m_(3), m_(4), m_(5)),m_(12));
	Eigen::Vector4d extqd(m_(6), m_(7), m_(8), 1);
	extqd.normalize();
	Eigen::Matrix3d extrot = cvl_toolkit::conversion::q2dcm(extqd);
	extrot = extR * extrot;
	Eigen::Vector3d exttd(m_(9), m_(10), m_(11));
	exttd = extt + exttd;

	//reconstruction
	std::vector<Eigen::Vector4d> pc;
	std::vector<float> timestamps;
	double stTS = image_timestamp.at(0);
	double endTS = image_timestamp.at(image_timestamp.size() - 2);

	{
		for (int i = 0; i < usedPtIdx.size(); i++) {
			long long fIdx = usedPtIdx.at(i);
			Eigen::Vector3d v(allpt[fIdx * 5 + 0], allpt[fIdx * 5 + 1], allpt[fIdx * 5 + 2]);
			double ts = allpt[fIdx * 5 + 4];
			//if (ts + sync_offset > endTS) {
			//	break;
			//}

			Eigen::Matrix4d motion = timeMotion(ts + sync_offset);

			Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
			Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

			v = Rot * (extrot * v + exttd) + trans;
			Eigen::Vector4d vt(v(0), v(1), v(2), 1);//1 or weight
			pc.push_back(vt);
			timestamps.push_back(ts);
			fIdx++;
		}
	}
	// variance Computation
	//std::cout << "log" << std::endl;
	//std::cout << m << std::endl;
	for (int i = 0; i < optvalue.size(); i++) {
		std::cout << m(i) <<" : ";
	}

	double cost = varianceCalc(pc);


	if (bEdge) {
		{
			pc.clear();
			timestamps.clear();
			for (int i = 0; i < usedEdgePtIdx.size(); i++) {
				long long fIdx = usedEdgePtIdx.at(i);
				Eigen::Vector3d v(allpt_edge[fIdx * 5 + 0], allpt_edge[fIdx * 5 + 1], allpt_edge[fIdx * 5 + 2]);
				double ts = allpt_edge[fIdx * 5 + 4];
				//if (ts + sync_offset > endTS) {
				//	break;
				//}

				Eigen::Matrix4d motion = timeMotion(ts + sync_offset);

				Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
				Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

				v = Rot * (extrot * v + exttd) + trans;
				Eigen::Vector4d vt(v(0), v(1), v(2), 1);// allpt_edge[fIdx * 5 + 3]);
				pc.push_back(vt);
				timestamps.push_back(ts);
				fIdx++;
			}
		}
		
		double cost_ = edgenessCalc(pc, timestamps);
		std::cout <<"variance cost:" << cost << std::endl;
		cost_ = round(cost_ * 10) / 10;
		std::cout <<"edgecost:" << cost_ << std::endl;
		cost =cost + 1/cost_;//scale
	}
	
	std::cout << "total cost:" << cost << std::endl;


	

	return cost;
}


double fusionscan::varianceCalc(std::vector<Eigen::Vector4d> pointcloud) {
	//point cloud

	double cost = 0;
	double wsum = 0;
#pragma omp parallel for reduction(+:cost),reduction(+:wsum) 
	for (int i = 0; i < pointcloud.size();i++) {
		double colord[3] = { 0,0,0 };
		double colordev[3] = { 0,0,0 };

		double w = onePointsColorAndVarianceCalc(pointcloud.at(i),i,pointcloud.size(),colord,colordev);
		double colordist = colordev[0] + colordev[1] + colordev[2];
		if (colordist < 0)colordist = 0;
		double errd = colordist;

		double colorNorm = colord[0] * colord[0]+ colord[1] * colord[1] + colord[2] * colord[2] + 1;//: offset for avoiding 0-div
//		colorNorm /= 255.0 * 255.0;
		w = w / sqrt(colorNorm);

		double addcost = w * /*lossFunction*/(errd);
		// addcost with color weight
//		addcost = pointcloud.at(i)(3) * addcost;
/*		if (distmode == 1) {
			addcost = sqrt(ptdist.at(i))*addcost;//far point: contribution is same to near point?
		}
		else */
		if (distmode == -1) {
			addcost = sqrt(1/ptdist.at(i))*addcost;
		}

		cost += addcost;
		wsum += w;
	}
	if (wsum == 0)return 100.0;
	double cost_ = cost / wsum;
	return cost_;
}

double fusionscan::edgenessCalc(std::vector<Eigen::Vector4d> pointcloud, std::vector<float> ts) {
	//point cloud

	double cost = 0;
	double wsum = 0;
#pragma omp parallel for reduction(+:cost),reduction(+:wsum) 
	for (int i = 0; i < pointcloud.size(); i++) {
		double edgecost;
		double w = onePointsEdgenessCalc(pointcloud.at(i),ts.at(i), i, pointcloud.size(), &edgecost);
		cost += edgecost;
		wsum += w;
	}
	double cost_ = cost / pointcloud.size();
	return cost_;
}

double fusionscan::onePointsColorAndVarianceCalc(Eigen::Vector4d pt,long long ptidx,long long pc_size,double* colord,double* colordev) {

	int cnt = 0;
	double weight, W = 0;
	double ret[3], dret[8], depret[8];
	double iave[3] = { 0,0,0 };

	for (int i = 0; i < usedIdx.size(); i++) {
		int idx = usedIdx.at(i);
		if (posibility != NULL && posibility[ptidx + i * pc_size]==0)continue;
		//camera pose
		Eigen::Matrix4d campos = deformedMotion.at(idx).inverse();
		Eigen::Matrix3d R = campos.block(0, 0, 3, 3);
		Eigen::Vector3d t = campos.block(0, 3, 3, 1);
		double ix, iy;
		Eigen::Vector3d t_p = R * pt.segment(0, 3) + t;

		cam->projection(t_p, ix, iy);

		if (ix < 0 || iy < 0 || ix >= imgs.at(idx).cols || iy >= imgs.at(idx).rows)continue;
		getColorSubPixel_diff_mask(imgs.at(idx), mask.at(idx), cv::Point2d(ix, iy), ret, dret, weight);

		double preW = W;
		if (posibility != NULL)weight = weight * (posibility[ptidx + i * pc_size] / 255.0);
		W += weight;
		if (weight != 0) {
			for (int j = 0; j < 3; j++) {
				double rgbretd = ret[j] / 256.0;
				double preave = iave[j];
				iave[j] = (preW*iave[j] + weight * rgbretd) / (W);
				colordev[j] = (preW * (colordev[j] + preave * preave) + weight * (rgbretd)*(rgbretd)) / (W)-iave[j] * iave[j];
			}
		}
	}
	colord[0] = iave[0] * 256;
	colord[1] = iave[1] * 256;
	colord[2] = iave[2] * 256;
	return W;
};

double fusionscan::onePointsEdgenessCalc(Eigen::Vector4d pt,float ts, long long ptidx, long long pc_size, double* colord) {

	int cnt = 0;
	double weight, W = 0;
	double ret[1], dret[8], depret[8];
	double iave[3] = { 0,0,0 };
	double edgecost = 0;
	for (int i = 0; i < usedIdx.size(); i++) {
		int idx = usedIdx.at(i);
		if (fabs((ts + sync_offset) - image_timestamp.at(idx)) > 0.1)continue;
		//if (posibility != NULL && posibility[ptidx + i * pc_size] == 0)continue;
		//camera pose
		Eigen::Matrix4d campos = deformedMotion.at(idx).inverse();
		Eigen::Matrix3d R = campos.block(0, 0, 3, 3);
		Eigen::Vector3d t = campos.block(0, 3, 3, 1);
		double ix, iy;
		Eigen::Vector3d t_p = R * pt.segment(0, 3) + t;

		cam->projection(t_p, ix, iy);

		if (ix < 0 || iy < 0 || ix >= imgs.at(idx).cols || iy >= imgs.at(idx).rows)continue;
		getGraySubPixel(edgeMat.at(idx), mask.at(idx), cv::Point2d(ix, iy), ret, weight);

		double preW = W;
		//if (posibility != NULL)weight = weight * (posibility[ptidx + i * pc_size] / 255.0);
		W += weight;
		edgecost = /*weight * */pt(3) * ret[0];
	}
	if (W == 0) {
		colord[0] = 0;
		return 0;
	}
	colord[0] = edgecost/W;
	return W;
};

void fusionscan::setSpecular(cv::Mat img, cv::Mat& outimg) {

	//specular detection (mask on just a bright area)

	uchar lut[256];
	double gm = 1.0 / 5.0;
	for (int i = 0; i < 256; i++) {
		lut[i] = pow(1.0*i / 255, gm) * 255;
	}
	cv::Mat lutmat = cv::Mat(cv::Size(256, 1), CV_8U, lut);
	cv::Mat img_color = img;
	cv::Mat gray_img;
	cv::cvtColor(img_color, gray_img, cv::COLOR_BGR2GRAY);
	//
	cv::threshold(gray_img, gray_img, 240, 255,  cv::THRESH_TOZERO);//get high value point
	cv::Mat blurredimg, gray_img_gamma;
	cv::GaussianBlur(gray_img, blurredimg, cv::Size(51, 51), 50);
	gray_img = cv::max(gray_img, blurredimg);
	gray_img = ~gray_img;
	gray_img.convertTo(gray_img, CV_32F, 1.0 / 255);
	gray_img.copyTo(outimg);


}

void fusionscan::setMask(nlohmann::json baseConf,double c2f) {
	cv::Mat baseMask;
	if (!baseConf["mask"].is_null()) {
		if (!baseConf["mask"]["baseFile"].is_null()) {
			baseMask = cv::imread(baseConf["mask"]["baseFile"].get<std::string>(),cv::ImreadModes::IMREAD_GRAYSCALE);
			baseMask.convertTo(baseMask, CV_32FC1, 1 / 255.0);
		}
		else {
			baseMask = cv::Mat(imgs.at(0).size(), CV_32FC1, 1.0);
		}
	
	}
	else {
		baseMask = cv::Mat(imgs.at(0).size(), CV_32FC1, 1.0);
	}
	if (c2f < 1.0 && c2f>0.0) {
		int w = baseMask.size().width * c2f;
		int h = baseMask.size().height * c2f;
		if (w % 2 == 1)w++;
		if (h % 2 == 1)h++;
		cv::resize(baseMask, baseMask, cv::Size(w,h));
	}

	for (int i = 0; i < usedIdx.size(); i++) {
		int idx = usedIdx.at(i);
		cv::Mat maskspec;
		setSpecular(imgs.at(idx), maskspec);

		cv::Mat masklay = maskspec & baseMask;
		masklay.copyTo(mask.at(idx));

		~maskspec;


	}

	if (!baseConf["mask"]["out"].is_null()) {
		cv::Mat outmask;
		mask.at(0).convertTo(outmask, CV_8UC1, 255.0);
		cv::imwrite(baseConf["mask"]["out"].get<std::string>(), outmask);
	}
}

void fusionscan::setEdge(nlohmann::json baseConf) {
	//edge detection and inverse test
	for (int i = 0; i < usedIdx.size(); i++) {
		int idx = usedIdx.at(i);
		cv::Mat edge, imggray;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		cv::cvtColor(imgs.at(idx), imggray, cv::COLOR_RGB2GRAY);
		//cv::Canny(imggray, edge, 150, 150);
		//cv::Laplacian(imggray, edge, CV_32FC1); edge = cv::abs(edge);
		imggray.convertTo(imggray, CV_32FC1, 1.0);
		edgeDetection(imggray, edge);
		cv::Mat invedge;
		edge.copyTo(invedge);
		//			invedge.convertTo(invedge, CV_32FC1, 1.0);
		if (true) {
			cv::Mat dilated, diff, odil;
			invedge.copyTo(odil);
			double gamma = 0.96;
			double gm = gamma;
			for (int i = 0; i < 50; i++) {
				cv::dilate(invedge, dilated, kernel);
				cv::max(gamma*dilated, invedge, invedge);

			}
			cv::Mat invedge_ = (1 / 3.0) * edge + (2 / 3.0) * invedge;
			//invedge_.convertTo(invedge, CV_8UC1, 1.0);
			//cv::imshow("nyan", invedge);
			//cv::waitKey();
			invedge_.copyTo(edgeMat[idx]);
		}
		else{
			cv::GaussianBlur(edge, edge, cv::Size(25, 25), 5);
			edge.copyTo(edgeMat[idx]);
		}

	}

	if (!baseConf["edgecheck"].is_null()) {
		cv::Mat o;
		edgeMat[0].convertTo(o,CV_8UC1,1.0);
		cv::imwrite(baseConf["edgecheck"].get<std::string>(),o);
	}

	std::ifstream edgeifs(baseConf["edge"].get<std::string>());
	nlohmann::json edgejson;
	edgeifs >> edgejson;
	edgeifs.close();
	std::ifstream ifs(edgejson["edgedata"].get<std::string>());
	ptnum_edge = edgejson["edgepoint_num"].get<long long>();

	allpt_edge = (float*)malloc(sizeof(float) * ptnum_edge * 5);
	ifs.read((char*)allpt_edge, sizeof(float) * ptnum_edge * 5);
	ifs.close();

	double stTS = image_timestamp.at(0);
	double endTS = image_timestamp.at(image_timestamp.size() - 2);
	usedEdgePtIdx.clear();
	{
		long long fIdx = edgeFindIdx(stTS - sync_offset);
		while (true) {
			double ts = allpt_edge[fIdx * 5 + 4];
			if (ts + sync_offset > endTS) {
				break;
			}
			usedEdgePtIdx.push_back(fIdx);
			fIdx++;
		}
	}


	bEdge = true;
}

void fusionscan::pointColorWeightComputation() {
	{
		ptweight.clear();
		ptdist.clear();
		std::vector<Eigen::Vector4d> pc;
		std::vector<double> timestamps;
		{
			int downsamp = 15;
			for (int i = 0; i < usedPtIdx.size(); i++) {
				long long fIdx = usedPtIdx.at(i);
				Eigen::Vector3d v(allpt[fIdx * 5 + 0], allpt[fIdx * 5 + 1], allpt[fIdx * 5 + 2]);
//				if (v.norm() < 0.5)continue;
				ptdist.push_back(v.norm());
				double ts = allpt[fIdx * 5 + 4];

				Eigen::Matrix4d motion = timeMotion(ts + sync_offset);

				Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
				Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

				v = Rot * (extR * v + extt) + trans;
				Eigen::Vector4d vt(v(0), v(1), v(2), allpt[fIdx * 5 + 3]);
				pc.push_back(vt);
				timestamps.push_back(ts + sync_offset);
				fIdx++;
			}
		}
		std::vector<uchar> color;
		//std::vector<Eigen::Matrix4d> tf_cams, std::vector<uchar>& color, std::vector<double> timestamps, std::vector<double> imtimestamp, camera*cam, double thres

		colorMapping(pc, color, timestamps);
		Eigen::Vector3d whitevec(1, 1, 1);
		whitevec.normalize();
		for (int i = 0; i < color.size() / 3; i++) {
			if (color[i * 3] == 0 && color[i * 3 + 1] == 0 && color[i * 3 + 2] == 0) {
				ptweight.push_back(0);
			}
			else {
				Eigen::Vector3d colorvec(color[i * 3], color[i * 3 + 1], color[i * 3 + 2]);
				colorvec.normalize();
				double colweight = (whitevec.cross(colorvec)).norm()+0.05;
				ptweight.push_back(colweight);

			}

		
		}
	
	}

}

void fusionscan::visiblePosibilityCalc(std::string plypath) {
	std::cout << "matrix size:" << usedPtIdx.size() *usedIdx.size() << std::endl;
	//depth map
	//set up pointcloud
	std::ifstream ply(plypath);
	cvl_toolkit::plyObject po;
	std::vector<float> color;

	std::vector<float> vtx;
	std::vector<float> rfl;
	if (ply.is_open()) {
		ply.close();
		po.readPlyFile(plypath);
		float *rflptr = po.getReflectancePointer();
		for (int idx = 0; idx < po.getVertexNumber();idx++) {
			color.push_back(rflptr[idx]); color.push_back(rflptr[idx]); color.push_back(rflptr[idx]);
		}
	}
	else {

		//reconstruction
		{
			for (int i = 0; i < usedPtIdx.size(); i++) {
				long long fIdx = usedPtIdx.at(i);
				Eigen::Vector3d v(allpt[fIdx * 5 + 0], allpt[fIdx * 5 + 1], allpt[fIdx * 5 + 2]);
				double ts = allpt[fIdx * 5 + 4];
				Eigen::Matrix4d motion = timeMotion(ts + sync_offset);//make with initial motion

				Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
				Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

				v = Rot * (extR * v + extt) + trans;
				Eigen::Vector4d vt(v(0), v(1), v(2), allpt[fIdx * 5 + 3]);
				vtx.push_back(v(0));
				vtx.push_back(v(1));
				vtx.push_back(v(2));

				color.push_back(allpt[fIdx * 5 + 3]);
				color.push_back(allpt[fIdx * 5 + 3]);
				color.push_back(allpt[fIdx * 5 + 3]);
				rfl.push_back(allpt[fIdx * 5 + 3]);
			}
		}

		po.setVertecesPointer(vtx.data(), vtx.size() / 3);
		po.setReflectancePointer(rfl.data(), rfl.size());
		genMesh(po);

		po.writePlyFile(plypath);
	}


	pr->setDataRGB_(po.getVertecesPointer(), po.getFaces(),color.data(),po.getVertexNumber(),po.getFaceNumber());
	int viewHeight;
	int viewWidth;
	PanoramaRenderer::getViewSize(viewWidth, viewHeight);
	posibility = (unsigned char*)malloc(sizeof(unsigned char) * usedPtIdx.size() *usedIdx.size());
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));//todo, check bettersize
	double depret[8];
	for (int i = 0; i < usedIdx.size(); i++) {
		int idx = usedIdx.at(i);
		//rendering
		Eigen::Matrix4d m = cameramotion.at(idx).inverse();	//make with initial motion
		pr->viewClear();
		pr->setView();
		pr->rendering(m);
		pr->memoryCopy();

		float* dat = pr->getDepthData();
		cv::Mat depthimage = cv::Mat(cv::Size(viewWidth, viewHeight), CV_32FC1);

		memcpy(depthimage.data, dat, sizeof(float) * depthimage.size().width * depthimage.size().height);
		cv::flip(depthimage, depthimage, 0);		
		cv::erode(depthimage, depthimage, kernel);//fill holes;
		double ts_image = cammot_timestamp.at(idx);
		//cv::Mat depthshow;
		//depthimage.convertTo(depthshow,CV_8UC1,1.0);
		//cv::imshow("neko",depthimage);
		//cv::waitKey();
		pr->clearImage();
		//posibitity computation
		for (int j = 0; j < usedPtIdx.size(); j++) {
			long long fIdx = usedPtIdx.at(j);
			Eigen::Vector3d v(allpt[fIdx * 5 + 0], allpt[fIdx * 5 + 1], allpt[fIdx * 5 + 2]);
			double ts = allpt[fIdx * 5 + 4];

			if (timeThresh > 0 && fabs(ts + sync_offset - ts_image) > timeThresh) {
				posibility[i*usedPtIdx.size() + j] = (unsigned char)(0);
				continue;
			}
			Eigen::Matrix4d motion = timeMotion(ts + sync_offset);

			Eigen::Matrix3d Rot = motion.block(0, 0, 3, 3);
			Eigen::Vector3d trans = motion.block(0, 3, 3, 1);

			v = Rot * (extR * v + extt) + trans;

			Eigen::Matrix3d Rot_c = m.block(0, 0, 3, 3);
			Eigen::Vector3d trans_c = m.block(0, 3, 3, 1);

			v = Rot_c * v + trans_c;


			double ix, iy;
			cam->projection(v, ix, iy);
			
			double w = 1.0;

			getGraySubPixel(depthimage, cv::Point2f(ix, iy), depret);
			double dpth = depret[0] * depret[1] + depret[2] * depret[3] +
				depret[4] * depret[5] + depret[6] * depret[7];
			double d = v.norm();

			w = depthRateCompute(dpth*20.0, d);//depth far clip
			/*if (j % 1000 == 0) {
				std::cout << ix << "," << iy << std::endl;
				std::cout << w << std::endl;
			}*/
			posibility[i*usedPtIdx.size() + j] = (unsigned char)(w*255);
		
		}
	}


};


void fusionscan::deformation(Eigen::Vector4d q,Eigen::Vector3d t,double scale) {

	double twsum = 0,qwsum=0;
	deformedMotion.clear();
	for (int i = 0; i < cameramotion.size(); i++) {
		Eigen::Matrix4d m = cameramotion.at(i);
		m.block(0, 3, 3, 1) = scale * m.block(0, 3, 3, 1);
		if (i > 0) {
			twsum += motionTranslationWeight.at(i - 1);
			qwsum += motionRotationWeight.at(i - 1);
			Eigen::Matrix4d mdiff;
			Eigen::Vector4d qdiff = qwsum * q;//interpolation between q and (0,0,0,1) with weight qwsum
			qdiff(3) += (1 - qwsum);
			qdiff.normalize();

			Eigen::Matrix3d qmat = cvl_toolkit::conversion::q2dcm(qdiff);
			Eigen::Vector3d tvec = twsum * t;
			mdiff.block(0, 0, 3, 3) = qmat;
			mdiff.block(0, 3, 3, 1) = tvec;
			mdiff.block(3, 0, 1, 4) << 0, 0, 0, 1;
			m = m * mdiff;
		}
		deformedMotion.push_back(m);
	}
};

void fusionscan::optimization_test() {
	//column_vector stpt = {0,0};
	//dlib::find_min_using_approximate_derivatives(
	//	dlib::lbfgs_search_strategy(10), 
	//	dlib::objective_delta_stop_strategy(1e-7),
	//	costComputation, stpt, -1);

}

void fusionscan::colorMapping(std::vector<Eigen::Vector4d> pc, std::vector<uchar>& color, std::vector<double> timestamps) {

		int imageIdx = 0;
		int imageWidth = imgs.at(0).size().width, imageHeight = imgs.at(0).size().height;
		double rgbret[3];

		for (int pidx = 0; pidx < pc.size(); pidx++) {
			double colord[3] = { 0,0,0 };
			int cnt = 0;
			double weight, W = 0;
			double ret[3];
			double iave[3] = { 0,0,0 };
			double colordev[3] = { 0,0,0 };
			double ts_point = timestamps.at(pidx);

			onePointsColorAndVarianceCalc(pc.at(pidx), pidx, pc.size(), colord, colordev);
			


			color.push_back(colord[0]);
			color.push_back(colord[1]);
			color.push_back(colord[2]);
		}
	

};

void fusionscan::outputCalib(const column_vector& m, std::string ofile){
	nlohmann::json extjson, extc2l, extl2c;

	std::vector<double> val;
	Eigen::Vector4d extqd(m(6), m(7), m(8), 1);
	extqd.normalize();
	Eigen::Matrix3d extr = cvl_toolkit::conversion::q2dcm(extqd);
	extr = extR * extr;
	Eigen::Vector3d exttd(m(9), m(10), m(11));
	exttd = extt + exttd;

	std::vector<double> extdat, extt;
	extdat.push_back(extr(0, 0));		extdat.push_back(extr(0, 1));		extdat.push_back(extr(0, 2));
	extdat.push_back(extr(1, 0));		extdat.push_back(extr(1, 1));		extdat.push_back(extr(1, 2));
	extdat.push_back(extr(2, 0));		extdat.push_back(extr(2, 1));		extdat.push_back(extr(2, 2));
	extt.push_back(exttd(0)); extt.push_back(exttd(1)); extt.push_back(exttd(2));
	extc2l["rotation"] = extdat;
	extc2l["translation"] = extt;
	extjson["camera2lidar"] = extc2l;

	Eigen::Matrix4d extParam;
	extParam.block(0, 0, 3, 3) = extr;
	extParam.block(0, 3, 3, 1) = exttd;
	extParam.block(3, 0, 1, 4) << 0, 0, 0, 1;

	Eigen::Matrix4d extinv = extParam.inverse();
	extr = extinv.block(0, 0, 3, 3);
	extdat.clear();
	extt.clear();
	extdat.push_back(extr(0, 0));		extdat.push_back(extr(0, 1));		extdat.push_back(extr(0, 2));
	extdat.push_back(extr(1, 0));		extdat.push_back(extr(1, 1));		extdat.push_back(extr(1, 2));
	extdat.push_back(extr(2, 0));		extdat.push_back(extr(2, 1));		extdat.push_back(extr(2, 2));
	extt.push_back(extinv(0, 3)); extt.push_back(extinv(1, 3)); extt.push_back(extinv(2, 3));

	extl2c["rotation"] = extdat;
	extl2c["translation"] = extt;
	extjson["lidar2camera"] = extl2c;

	for (int i = 0; i < 13; i++)val.push_back(m(i));
	extjson["optparams"] = val;


	std::ofstream ofs(ofile);
	

	ofs << std::setw(4) << extjson << std::endl;
	return;
}

