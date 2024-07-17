//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/function.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>

#include <shlwapi.h>
#pragma comment(lib, "shlwapi.lib")
#include <shobjidl.h>
#include <shlguid.h>
#include <thread>

#include <Eigen\Core>
#include <Eigen\Dense>

#include <fusionba/utility.h>
#include <fusionba/PointSequenceLoader.h>

#include <nlohmann/json.hpp>

using namespace Eigen;

#ifndef SENSORFUSION
#define SENSORFUSION

class SensorFusion {
public:
	

	struct Status {
	public:
		int cIdx;
		std::vector<cv::Point2f> cor2d;
		std::vector<Vector3d> cor3d;
		std::vector<float> pointTimestamp;
		cv::Mat lastImage;
		cv::Mat mask;
		bool bMask = false;
		_6dof camPose;
		void* dataptr;
	};

	struct Inputs {
	public:
		PointSequenceLoader* psl;
		std::vector<string> imgFileNameList;
		double* imTimeStamp;
		Matrix4d extCalib;
	};


	Inputs setFolder(std::string workFolder);
	Inputs setFolder(std::string workFolder, std::string extCparaFile);

	void setPointCloud(nlohmann::json config);
	void setImage(nlohmann::json scanconf);
	void setInitCalib(nlohmann::json baseConf);

	Status initialize(Status init);
	Status update();
	void setCpara(string extFile) {
		inputs.extCalib = readCPara(extFile).inverse();
	}


	void profileRotation(double degangle) {
		Matrix4d profRot;
		double horizAngle = degangle / 180 * M_PI;
		profRot << cos(horizAngle), sin(horizAngle), 0, 0,
			-sin(horizAngle), cos(horizAngle), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		inputs.extCalib= inputs.extCalib * profRot;
	};
	void imrectRotation() {
		Vector3d axz, axr, ep;axz << 0, 0, 1;
		ep << 1, 0, 0;
		axr = axz.cross(ep);
		double angle = -asin(axr.norm());
		if (axz.dot(ep) < 0)angle = M_PI - angle;

		axr = axr.normalized();



		Matrix3d raxis;
		raxis << 0, -axr(2), axr(1),
			axr(2), 0, -axr(0),
			-axr(1), axr(0), 0;

		Matrix3d R0g = Matrix3d::Identity() + sin(angle)*raxis + (1 - cos(angle))*raxis*raxis;

		Matrix4d R0g4x4;R0g4x4.block(0, 0, 3, 3) = R0g;
		inputs.extCalib = R0g4x4* inputs.extCalib;
	};
	void setUpdateFunc(boost::function<void(Status&, Inputs&)>& func) { updateFunc = func; };
	void setInitializeFunc(boost::function<void( Status&, Inputs&)>& func) { initializeFunc = func; };
	bool setMask(cv::Mat maskimg);
	Inputs getInput() { return inputs; };
	Status getStatus() {
		return stat;
	};

	string imageBase;


protected:
	//status
	Status stat;
	boost::function<void(Status&, Inputs&)> updateFunc = NULL;
	boost::function<void(Status&, Inputs&)> initializeFunc = NULL;

	//input
	Inputs inputs;
	//string getShortcut(std::string shortcutLink) {
	//		TCHAR path[MAX_PATH], file_path[MAX_PATH];   //      �V���[�g�J�b�g�̃����N��
	//		char* imageLiseBaseChar;
	//		mbstowcs(file_path, shortcutLink.c_str(), (sizeof file_path) / 2);
	//		
	//		
	//		//CoInitialize(NULL);

	//		{
	//		
	//			IShellLink  *m_pShellLink = NULL;
	//			IPersistFile    *m_pPersistFile = NULL;
	//			//      IShellLink�I�u�W�F�N�g���쐬
	//			if (CoCreateInstance(CLSID_ShellLink, NULL, CLSCTX_INPROC_SERVER, IID_IShellLink, (LPVOID *)&m_pShellLink) == S_OK) {
	//				//      IPersistFile�C���^�[�t�F�[�X�̍쐬
	//				if (m_pShellLink->QueryInterface(IID_IPersistFile, (LPVOID *)&m_pPersistFile) == S_OK) {
	//					//      �V���[�g�J�b�g�t�@�C���̓ǂݍ��݁@�����̃V���[�g�J�b�g����������ꍇ�͈ȉ�6�s���J��Ԃ�
	//					if (m_pPersistFile->Load(file_path, STGM_READ) == S_OK) {
	//						if (m_pShellLink->Resolve(NULL, 0) == NOERROR) {      //      �����N��Ƀt�@�C�����Ȃ��ꍇ�̓G���[���b�Z�[�W��\������
	//							m_pShellLink->GetPath(path, sizeof(path) / sizeof(TCHAR), NULL, SLGP_UNCPRIORITY);        //      �����N��擾
	////							MessageBox(0, pass, _TEXT("�����N��"), MB_OK);
	//						}
	//					}
	//					else {
	//						//						MessageBox(0, file_pass, _TEXT("�V���[�g�J�b�g�t�@�C����������܂���B"), MB_OK);
	//					}
	//					m_pShellLink->Release();     //      IShellLink�I�u�W�F�N�g�̉��
	//					m_pPersistFile->Release();   //      IPersistFile�C���^�[�t�F�[�X�̉��
	//				}
	//			}
	//	
	//		}

	//		std::thread th_short;

	//		wcstombs(imageLiseBaseChar, path, sizeof(path));
	//		return string(imageLiseBaseChar);
	//}
	
};

#endif