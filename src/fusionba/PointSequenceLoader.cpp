//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <fusionba\PointSequenceLoader.h>



void PointSequenceLoader::loadPointStream(std::string filePath,long long pnum)
{
	pointNum = pnum;
	if (ifs!=NULL)ifs->close();
	ifs = new std::ifstream(filePath, std::ios::binary);
	//ifs->read((char*)&pointNum, sizeof(__int64));
	std::cout << pointNum << std::endl;
	fcnt = 0;
	baseFilePath = filePath;
	bStream = true;
}

bool PointSequenceLoader::getNextPointData(float* dat) {
	size_t readed;
	//dat = (float*)malloc(sizeof(float) * 5);
	readed = ifs->read((char*)dat, sizeof(float) * 5).gcount();
	if (readed == 0) {
		fcnt++;
		std::stringstream filePathTmp;filePathTmp << baseFilePath << "-" << fcnt << ".dat";
		cout << filePathTmp.str() << endl;
		ifs->close();
		if (!PathFileExistsA(filePathTmp.str().c_str()))return false;//last point;
		ifs->open(filePathTmp.str(), ios::binary);
		//if(!ifs_stream->)return false;
		readed = ifs->read((char*)dat, sizeof(float) * 5).gcount();
	}
	dat[4] += ts;
	return true;
}

bool PointSequenceLoader::backPointData() {//cannot call twice
	size_t readed;
	
	if (ifs->eof()) {
		ifs->clear();
		ifs->seekg(sizeof(float) * -5, ios_base::end);
	}
	else {
		ifs->seekg(sizeof(float) * -5, ios_base::cur);	
	}
	return true;
}

void PointSequenceLoader::writePlyReflectance(string filePath, double* motionTime, _6dof* motion, Matrix4d& calib, int firstFrame, int lastFrame) {
	PointSequenceLoader::writePlyReflectance(filePath, motionTime, motion, calib, firstFrame, lastFrame, 0);

}

__int64 PointSequenceLoader::seekByTime(double t) {
	
	__int64 startIdx = 0,endIdx=pointNum-1,centerIdx=startIdx;
	float dat[5];
	
	while (endIdx-startIdx>1) {
		centerIdx = (endIdx+startIdx) / 2;
		seekByPointIdx(centerIdx);
		getNextPointData(dat);
		double tc = dat[4];
		if (tc > t) {
			endIdx = centerIdx;
		};
		if (tc <= t) {
			startIdx = centerIdx;
		};
	}
	centerIdx = (endIdx + startIdx) / 2;
	seekByPointIdx(centerIdx);

	return centerIdx;
}


bool PointSequenceLoader::seekByPointIdx(__int64 idx) {
	if (idx >= pointNum)return false;

	if (ifs != NULL)ifs->close();
	ifs = new std::ifstream(baseFilePath, std::ios::binary);
	ifs->seekg(0, fstream::end);
	size_t fsize=(size_t)ifs->tellg();
	fsize = (fsize) / (sizeof(float)*5);
	__int64 idx_ = idx;
	bool bFirst = true;;
	fcnt = 0;
	while (idx_>=fsize) {
		idx_ -= fsize;
		bFirst = false;
		fcnt++;
		std::stringstream filePathTmp;filePathTmp << baseFilePath << "-" << fcnt << ".dat";
		ifs->close();
		if (!PathFileExistsA(filePathTmp.str().c_str()))return false;//last point;
		ifs->open(filePathTmp.str(), ios::binary);
		ifs->seekg(0, fstream::end);
		fsize = (size_t)ifs->tellg();
		fsize = (fsize ) / (sizeof(float) * 5);
	}
	ifs->clear();
	if (bFirst) {
		ifs->seekg(idx_ * sizeof(float) * 5, fstream::beg);
	}
	else {
		ifs->seekg(idx_ * sizeof(float) * 5 , fstream::beg);
	}

	//ifs->read((char*)&pointNum, sizeof(__int64));




	return true;
}

void PointSequenceLoader::writePlyReflectance(string filePath, double* motionTime, _6dof* motion, Matrix4d& calib, int firstFrame, int lastFrame, int skip,double refThres) {
	
	
		ifs->close();
		loadPointStream(baseFilePath,pointNum);
		//ifs_stream->open(baseFilePath+".dat",ios::binary);
		ofstream ofs(filePath, ios::out | ios::binary);

		int time;
		int i = 0;
		int lineNumber = 0;
		Matrix4d linePose;
		int idx = firstFrame;
		double st, et;
		vector<float> v;
		vector<float> rf;
		vector<float> motp;
		int ds = skip + 1;
		Matrix4d invcalib = calib.inverse();
		float dat[5];
		this->seekByTime(motionTime[idx] - 1 / 32.0);
		while (this->getNextPointData(dat)) {
			if (i%ds != 0) { i++;continue; }
			i++;
			double t = dat[4];
			if (t<motionTime[idx] - 1 / 32.0) { continue; }
			while (idx<lastFrame && (et = motionTime[idx + 1])<t) {
				//cout<<motion.at(idx).x<<","<<motion.at(idx).y<<","<<motion.at(idx).z<<","<<motion.at(idx).rx<<","<<motion.at(idx).ry<<","<<motion.at(idx).rz<<endl;
				linePose = _6dof2m(motion[idx]);
				Matrix4d mot_s = invcalib*linePose*calib;
				motp.push_back(mot_s(0,3));
				motp.push_back(mot_s(1, 3));
				motp.push_back(mot_s(2, 3));
				idx++;
			}
			st = motionTime[idx];
			Vector4d point;

			if (idx == lastFrame&&motionTime[idx] + 1 / 32.0<t) { break; }

			point << dat[0], dat[1], dat[2], 1;
			double d = dat[0] * dat[0] + dat[1] * dat[1] + dat[2] * dat[2];
			if (d<0.04 || d>10000.0) { continue; }//range cut


			_6dof dof;
			if (t<st)dof = motion[idx];
			else if (idx<lastFrame) {
				_6dof dofa = motion[idx];
				_6dof dofb = motion[idx + 1];
				double w = 1 - (t - st) / (et - st);
				dof = w*dofa + (1 - w)*dofb;
			}
			else {
				dof = motion[idx];
			}
			Matrix4d Rx, Ry, Rz;
			double sinv = sin(dof.rx), cosv = cos(dof.rx);
			Rx << 1, 0, 0, 0,
				0, cosv, -sinv, 0,
				0, sinv, cosv, 0,
				0, 0, 0, 1;
			sinv = sin(dof.ry);cosv = cos(dof.ry);
			Ry << cosv, 0, sinv, 0,
				0, 1, 0, 0,
				-sinv, 0, cosv, 0,
				0, 0, 0, 1;
			sinv = sin(dof.rz);cosv = cos(dof.rz);
			Rz << cosv, -sinv, 0, 0,
				sinv, cosv, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1;
			linePose = (Rx*Ry*Rz);
			linePose(0, 3) = dof.x;
			linePose(1, 3) = dof.y;
			linePose(2, 3) = dof.z;

			float fa[5];

			point = invcalib*linePose*calib*point;
			if (dat[3]>refThres) {
				if (point(0, 0) != point(0, 0) || point(1, 0) != point(1, 0) || point(2, 0) != point(2, 0)) {

					continue;
				}

				v.push_back(point(0, 0));
				v.push_back(point(1, 0));
				v.push_back(point(2, 0));
				rf.push_back(dat[3]);
			}
		}

		int motiondata = 0;




		ofs << "ply" << endl;
		ofs << "format binary_little_endian 1.0" << endl;
		ofs << "element vertex " << rf.size() + motp.size() / 3 << endl;
		ofs << "property float x" << endl;
		ofs << "property float y" << endl;
		ofs << "property float z" << endl;
		ofs << "property float confidence" << endl;
//		ofs << "property float intensity" << endl;
		ofs << "property uchar red" << endl;
		ofs << "property uchar green" << endl;
		ofs << "property uchar blue" << endl;
		ofs << "property uchar alpha" << endl;
		ofs << "element face 0" << endl;
		ofs << "property list uchar int vertex_index" << endl;
		ofs << "end_header" << endl;

		for (int i = 0;i<rf.size();i++) {
			float fa[4] = { v.at(i * 3),v.at(i * 3 + 1),v.at(i * 3 + 2),1.0 };
			ofs.write((char *)fa, sizeof(float) * 4);
			unsigned char rgba[4];
			rgba[0] = rgba[1] = rgba[2] = (unsigned char)(rf.at(i)*256);
			rgba[3] = 255;
			ofs.write((char *)rgba, sizeof(unsigned char) * 4);
		}
		for (int i = 0;i < motp.size()/3;i++) {
			float fa[4] = { motp.at(i * 3),motp.at(i * 3 + 1),motp.at(i * 3 + 2),1.0 };
			ofs.write((char *)fa, sizeof(float) * 4);
			unsigned char rgba[4];
			rgba[0] = rgba[1]=0;
			rgba[2] = 255;
			rgba[3] = 255;
			ofs.write((char *)rgba, sizeof(unsigned char) * 4);
		}
		ofs.close();
};

void PointSequenceLoader::writePlyReflectance(string filePath, string filePathPreview, double* motionTime, _6dof* motion, Matrix4d& calib, int firstFrame, int lastFrame, int skip) {


	ifs->close();
	loadPointStream(baseFilePath, pointNum);
	//ifs_stream->open(baseFilePath+".dat",ios::binary);
	

	int time;
	int i = 0;
	int lineNumber = 0;
	Matrix4d linePose;
	int idx = firstFrame;
	double st, et;
	vector<float> v;
	vector<float> rf;
	int ds = skip + 1;
	Matrix4d invcalib = calib.inverse();
	float dat[5];

	this->seekByTime(motionTime[idx] - 1 / 32.0);

	while (this->getNextPointData(dat)) {
		if (i%ds != 0) { i++;continue; }
		i++;
		double t = dat[4];
		if (t<motionTime[idx] - 1 / 32.0) { continue; }
		while (idx<lastFrame && (et = motionTime[idx + 1])<t) {
			//cout<<motion.at(idx).x<<","<<motion.at(idx).y<<","<<motion.at(idx).z<<","<<motion.at(idx).rx<<","<<motion.at(idx).ry<<","<<motion.at(idx).rz<<endl;
			idx++;
		}
		st = motionTime[idx];
		Vector4d point;

		if (idx == lastFrame && motionTime[idx] + 1 / 32.0<t) { break; }

		point << dat[0], dat[1], dat[2], 1;
		double d = dat[0] * dat[0] + dat[1] * dat[1] + dat[2] * dat[2];
		if (d<0.04 || d>10000.0) { continue; }//range cut


		_6dof dof;
		if (t<st)dof = motion[idx];
		else if (idx<lastFrame) {
			_6dof dofa = motion[idx];
			_6dof dofb = motion[idx + 1];
			double w = 1 - (t - st) / (et - st);
			dof = w * dofa + (1 - w)*dofb;
		}
		else {
			dof = motion[idx];
		}
		Matrix4d Rx, Ry, Rz;
		double sinv = sin(dof.rx), cosv = cos(dof.rx);
		Rx << 1, 0, 0, 0,
			0, cosv, -sinv, 0,
			0, sinv, cosv, 0,
			0, 0, 0, 1;
		sinv = sin(dof.ry);cosv = cos(dof.ry);
		Ry << cosv, 0, sinv, 0,
			0, 1, 0, 0,
			-sinv, 0, cosv, 0,
			0, 0, 0, 1;
		sinv = sin(dof.rz);cosv = cos(dof.rz);
		Rz << cosv, -sinv, 0, 0,
			sinv, cosv, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		linePose = (Rx*Ry*Rz);
		linePose(0, 3) = dof.x;
		linePose(1, 3) = dof.y;
		linePose(2, 3) = dof.z;

		float fa[5];

		point = invcalib * linePose*calib*point;
		if (dat[3]>0.05) {
			if (point(0, 0) != point(0, 0) || point(1, 0) != point(1, 0) || point(2, 0) != point(2, 0)) {

				continue;
			}

			v.push_back(point(0, 0));
			v.push_back(point(1, 0));
			v.push_back(point(2, 0));
			rf.push_back(dat[3]);
		}

	}


	ofstream ofs(filePath, ios::out | ios::binary);
	ofs << "ply" << endl;
	ofs << "format binary_little_endian 1.0" << endl;
	ofs << "element vertex " << rf.size() << endl;
	ofs << "property float x" << endl;
	ofs << "property float y" << endl;
	ofs << "property float z" << endl;
	ofs << "property float confidence" << endl;
	ofs << "property float intensity" << endl;
	ofs << "element face 0" << endl;
	ofs << "property list uchar int vertex_index" << endl;
	ofs << "end_header" << endl;

	for (int i = 0;i<rf.size();i++) {
		float fa[5] = { v.at(i * 3),v.at(i * 3 + 1),v.at(i * 3 + 2),1.0,rf.at(i) };
		ofs.write((char *)fa, sizeof(float) * 5);
	}
	ofs.close();

	ofs.open(filePathPreview, ios::out | ios::binary);
	ofs << "ply" << endl;
	ofs << "format binary_little_endian 1.0" << endl;
	ofs << "element vertex " << rf.size()/100 << endl;
	ofs << "property float x" << endl;
	ofs << "property float y" << endl;
	ofs << "property float z" << endl;
	ofs << "property float confidence" << endl;
	ofs << "property float intensity" << endl;
	ofs << "element face 0" << endl;
	ofs << "property list uchar int vertex_index" << endl;
	ofs << "end_header" << endl;

	for (int i = 0;i<rf.size();i+=100) {
		float fa[5] = { v.at(i * 3),v.at(i * 3 + 1),v.at(i * 3 + 2),1.0,rf.at(i) };
		ofs.write((char *)fa, sizeof(float) * 5);
	}
	ofs.close();
};



void PointSequenceLoader::getPCReflectance(double* motionTime, _6dof* motion, Matrix4d& calib, int firstFrame, int lastFrame, int skip, vector<float>& v,vector<float>& rf) {
	ifs->close();
	loadPointStream(baseFilePath, pointNum);
	
	int time;
	int i = 0;
	int lineNumber = 0;
	Matrix4d linePose;
	int idx = firstFrame;
	double st, et;
	int ds = skip + 1;
	Matrix4d invcalib = calib.inverse();
	float dat[5];

	this->seekByTime(motionTime[idx] - 1 / 32.0);

	while (this->getNextPointData(dat)) {
		if (i%ds != 0) { i++;continue; }
		i++;
		double t = dat[4];
		if (t<motionTime[idx] - 1 / 32.0) { continue; }
		while (idx<lastFrame && (et = motionTime[idx + 1])<t) {
			idx++;
		}
		st = motionTime[idx];
		Vector4d point;

		if (idx == lastFrame && motionTime[idx] + 1 / 32.0<t) { break; }

		point << dat[0], dat[1], dat[2], 1;
		double d = dat[0] * dat[0] + dat[1] * dat[1] + dat[2] * dat[2];
		if (d<0.5 || d>400.0) { continue; }//range cut


		_6dof dof;
		if (t<st)dof = motion[idx];
		else if (idx<lastFrame) {
			_6dof dofa = motion[idx];
			_6dof dofb = motion[idx + 1];
			double w = 1 - (t - st) / (et - st);
			dof = w * dofa + (1 - w)*dofb;
		}
		else {
			dof = motion[idx];
		}
		Matrix4d Rx, Ry, Rz;
		double sinv = sin(dof.rx), cosv = cos(dof.rx);
		Rx << 1, 0, 0, 0,
			0, cosv, -sinv, 0,
			0, sinv, cosv, 0,
			0, 0, 0, 1;
		sinv = sin(dof.ry);cosv = cos(dof.ry);
		Ry << cosv, 0, sinv, 0,
			0, 1, 0, 0,
			-sinv, 0, cosv, 0,
			0, 0, 0, 1;
		sinv = sin(dof.rz);cosv = cos(dof.rz);
		Rz << cosv, -sinv, 0, 0,
			sinv, cosv, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		linePose = (Rx*Ry*Rz);
		linePose(0, 3) = dof.x;
		linePose(1, 3) = dof.y;
		linePose(2, 3) = dof.z;

		float fa[5];

		point = invcalib * linePose*calib*point;
		if (dat[3]>0.05) {
			if (point(0, 0) != point(0, 0) || point(1, 0) != point(1, 0) || point(2, 0) != point(2, 0)) {
				continue;
			}

			v.push_back(point(0, 0));
			v.push_back(point(1, 0));
			v.push_back(point(2, 0));
			rf.push_back(dat[3]);
		}
	}
}

#if USE_OPENCV
void PointSequenceLoader::getPCColor(double* motionTime, _6dof* motion, Matrix4d& calib, int firstFrame, int lastFrame, vector<string>imgFileNameList, vector<float>& v, vector<uchar>& rgba, int skip) {

	ifs->close();
	loadPointStream(baseFilePath);

	int time;
	int i = 0;
	int lineNumber = 0;
	Matrix4d linePose;
	int idx = firstFrame;
	cv::Mat imgFrame = cv::imread(imgFileNameList.at(idx));
	double st, et;
	int ds = skip + 1;
	Matrix4d invcalib = calib.inverse();
	float dat[5];

	this->seekByTime(motionTime[idx] - 1 / 32.0);

	while (this->getNextPointData(dat)) {
		if (i%ds != 0) { i++;continue; }
		i++;
		double t = dat[4];
		if (t < motionTime[idx] - 1 / 32.0) { continue; }
		while (idx < lastFrame && (et = motionTime[idx + 1]) < t) {
			idx++;
			imgFrame = cv::imread(imgFileNameList.at(idx));
		}
		st = motionTime[idx];
		Vector4d point;

		if (idx == lastFrame && motionTime[idx] + 1 / 32.0 < t) { break; }

		point << dat[0], dat[1], dat[2], 1;
		double d = dat[0] * dat[0] + dat[1] * dat[1] + dat[2] * dat[2];
		if (d<0.5 || d>400.0) { continue; }//range cut


		_6dof dof;
		if (t < st)dof = motion[idx];
		else if (idx < lastFrame) {
			_6dof dofa = motion[idx];
			_6dof dofb = motion[idx + 1];
			double w = 1 - (t - st) / (et - st);
			dof = w * dofa + (1 - w)*dofb;
		}
		else {
			dof = motion[idx];
		}
		Matrix4d baseMat = _6dof2m(motion[idx]);
		Matrix4d Rx, Ry, Rz;
		double sinv = sin(dof.rx), cosv = cos(dof.rx);
		Rx << 1, 0, 0, 0,
			0, cosv, -sinv, 0,
			0, sinv, cosv, 0,
			0, 0, 0, 1;
		sinv = sin(dof.ry);cosv = cos(dof.ry);
		Ry << cosv, 0, sinv, 0,
			0, 1, 0, 0,
			-sinv, 0, cosv, 0,
			0, 0, 0, 1;
		sinv = sin(dof.rz);cosv = cos(dof.rz);
		Rz << cosv, -sinv, 0, 0,
			sinv, cosv, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		linePose = (Rx*Ry*Rz);
		linePose(0, 3) = dof.x;
		linePose(1, 3) = dof.y;
		linePose(2, 3) = dof.z;

		float fa[5];
		Vector4d cameraCenterPoint = baseMat.inverse()*linePose*calib*point;
		point = invcalib * linePose*calib*point;
		if (dat[3] > 0.05) {
			if (point(0, 0) != point(0, 0) || point(1, 0) != point(1, 0) || point(2, 0) != point(2, 0)) {

				continue;
			}
			double ret[3];
			double ix, iy;
			v.push_back(point(0, 0));
			v.push_back(point(1, 0));
			v.push_back(point(2, 0));
			omniTrans(cameraCenterPoint(0), cameraCenterPoint(1), cameraCenterPoint(2), iy, ix, imgFrame.size().height);

			getColorSubPixel(imgFrame, cv::Point2f(ix, iy), ret);

			rgba.push_back(ret[0]);
			rgba.push_back(ret[1]);
			rgba.push_back(ret[2]);
			rgba.push_back(255);
		}

	}

};
#endif