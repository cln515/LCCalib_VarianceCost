//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include "BA_funcs.h"
#include "SensorFusionBA.h"
#include "nlohmann/json.hpp"
//#include <time.h> 
//
//#if defined(WIN32) || defined(WIN64)
//// Windows 32-bit and 64-bit
//#include <Windows.h>
//#include <windows.system.h>
//#include <imagehlp.h>
//#pragma comment(lib, "imagehlp.lib")
//
//#elif defined(MAC_OSX)
//// Mac OSX
//
//#else
//// Linux and all others
//// Using GCC 4 where hiding attributes is possible
//#include <sys/stat.h>
//
//#endif
//
//
//std::string getTimeStamp() {
//	time_t rawtime;
//	struct tm * timeinfo;
//	char buffer[80];
//
//	time(&rawtime);
//	timeinfo = localtime(&rawtime);
//
//	strftime(buffer, 80, "%y%m%d_%H%M%S", timeinfo);
//	std::string ts(buffer);
//	return ts;
//}
//
//bool makeFolder(std::string folderPath) {
//	std::cout << "create " + folderPath << std::endl;
//#if defined(WIN32) || defined(WIN64)
//	if (MakeSureDirectoryPathExists(folderPath.c_str())) {
//		std::cout << "succeeded." << std::endl;
//		return true;
//	}
//	else {
//		std::cout << "failed" << std::endl;
//		return false;
//	}
//#elif defined(__unix__)
//
//	std::string folderPath_make = folderPath.substr(0, folderPath.find_last_of('/'));
//	//	std::cout<<folderPath_make;
//	std::string cmd = "mkdir -p " + folderPath_make;
//	system(cmd.c_str());
//#endif
//}

int main(int argc, char* argv[]) {

	std::ifstream ifs(argv[1]);
	nlohmann::json config;
	ifs >> config;
	ifs.close();

	std::string outputDir = config["output"].get<std::string>();
	std::string timeStamp = getTimeStamp();
	makeFolder(outputDir + "\\config" + timeStamp + ".json");
	std::ofstream ofsj(outputDir + "\\config" + timeStamp + ".json");
	ofsj << std::setw(4) << config << std::endl;
	ofsj.close();

	SensorFusionBA* sfba = new SensorFusionBA();

	ifs.open(config["scan"].get<std::string>());
	nlohmann::json scanconf;
	ifs >> scanconf;
	ifs.close();
	
	sfba->setImage(scanconf);
	sfba->setPointCloud(scanconf);
	sfba->setInitCalib(config);
	//if (!config["time_offset"].is_null()) {
	//	sync_offset = config["sync_offset"].get<double>();
	//	sfba->getInput().psl->setTimeOffset(config["time_offset"].get<double>());
	//}
	int firstFrame = 0;// scanconf["first_frame"].get<int>();
	int lastFrame = sfba->getInput().imgFileNameList.size()-1;// scanconf["end_frame"].get<int>();

	bool mok = sfba->LoadMotion(config["initial_motion"].get<std::string>());
	if (!mok) {
		cout << "loading motion: false" << endl;
		return -1;
	}
	int ds = 1;


	if (lastFrame >= sfba->framePos.size())lastFrame = sfba->framePos.size() - 1;
	std::string asciiout = outputDir + "\\ascii" + timeStamp + ".csv";
	std::string binout = outputDir + "\\motion.json";
	std::string logout = outputDir + "\\log_" + timeStamp + ".txt";
	makeFolder(asciiout);
	makeFolder(binout);
	ofstream logout_(logout);


	if (argc >= 3) {
		sfba->getInput().psl->writePlyReflectance(config["output"].get<std::string>() + "\\" + "model_output.ply", sfba->getInput().imTimeStamp, sfba->framePos.data(),
			sfba->getInput().extCalib, firstFrame, lastFrame, config["skip"].get<int>(), 0.04);//config["skip"].get<int>()
		return 0;
	};

	cv::Mat mask = cv::imread(config["mask"].get<std::string>(), cv::ImreadModes::IMREAD_GRAYSCALE);
	if (mask.cols > 0) {
		cv::resize(mask,mask, cv::Size(2048, 1024));
		sfba->setMask(mask);
	}

	{
		nlohmann::json baparam = config["ba_parameter"].get<nlohmann::json>();
		int num;
		if (baparam["divide"].is_null()) {
			num = 1;
		}
		else {
			num = baparam["divide"].get<int>();
		}
		if (baparam["frame_downsample"].is_null()) {
			ds = 40;
		}
		else {
			ds = baparam["frame_downsample"].get<int>();
		}
		if (!baparam["frame_skip"].is_null()) {
			sfba->fitv = baparam["frame_skip"].get<int>();
		}
		if (!baparam["detect_rad"].is_null()) {
			sfba->dcpara = baparam["detect_rad"].get<double>();
		}
		sfba->bfixTrans = true;
		vector<double> accumlatedLength;
		double sum = 0;
		accumlatedLength.push_back(sum);
		for (int i = firstFrame; i < lastFrame - 1; i++) {
			Matrix4d m1 = _6dof2m(sfba->framePos.at(i));
			Matrix4d m2 = _6dof2m(sfba->framePos.at(i + 1));
			Matrix4d mdiff = m1.inverse()*m2;
			double difft = mdiff.block(0, 3, 3, 1).norm();
			sum += difft;
			accumlatedLength.push_back(sum);
		}

		double intervaldist = 2 * sum / (num + 1);
		double overlapl = intervaldist / 2;
		int aboutfnum = ds;
		double interval_framedist = intervaldist / aboutfnum;
		//		if (num > 1)intervaldist += overlapl;

		//		double interval = (end - start - batch) / (double)(num - 1);
		cout << "start:" << firstFrame << "  end:" << lastFrame << "  length:" << sum << "  num:" << num << "  interval/frame interval:" << intervaldist << "m/" << interval_framedist << "m" << endl;
		logout_ << "start:" << firstFrame << "  end:" << lastFrame << "  num:" << num << "  interval/frame interval:" << intervaldist << "m/" << interval_framedist << "m" << endl;
		int startframe = firstFrame;
		int cnt = 0;
		int precnt = 0;
		sfba->getInput().psl->writePlyReflectance(config["output"].get<std::string>() + "\\" + "model_o.ply", sfba->getInput().imTimeStamp, sfba->framePos.data(),
			sfba->getInput().extCalib, firstFrame, lastFrame, config["skip"].get<int>(), 0.00);

		for (int i = -1; i < num; i++) {

			int init = cnt;
			int ovcnt = 0;
			vector<int> sampledFrame;
			sampledFrame.push_back(cnt + firstFrame);
			double dcnt = 1;
			double offset = accumlatedLength.at(cnt);
			int overlap = -1;
			{
				while (cnt < accumlatedLength.size() - 1 && accumlatedLength.at(cnt) <= (intervaldist*(i + 2) / (2))) {
					std::cout << accumlatedLength.at(cnt) << std::endl;
					if (accumlatedLength.at(cnt) > interval_framedist * dcnt + offset) {
						sampledFrame.push_back(cnt + firstFrame);
						while (accumlatedLength.at(cnt) > interval_framedist * dcnt + offset)dcnt++;
					}
					if (accumlatedLength.at(cnt) > ((intervaldist*(i + 2) / (2)) - overlapl)) {
						ovcnt++;
					}
					if (overlap == -1 && cnt - init >= precnt) { overlap = sampledFrame.size() - 2; }
					cnt++;
				}
			}

			int frameNum = cnt - init;

			//sfba.BatchBA_TL(start + i * interval, FN_VAL, 2);
//			int overlap = precnt ;//i == 0 ? 0 : ovcnt;
			cout << "batch " << i << ", start:" << startframe << "  end:" << startframe + frameNum << "  overlap:" << overlap << endl;
			logout_ << "batch " << i << ", start:" << startframe << "  end:" << startframe + frameNum << "  overlap:" << overlap << endl;
			for (int j = 0; j < sampledFrame.size(); j++) {
				std::cout << sampledFrame.at(j) << std::endl;
			}

			std::string log = sfba->BatchBA_Global_seq(sampledFrame, overlap);
			startframe = startframe + frameNum - ovcnt;
			cnt = startframe - firstFrame;
			precnt = ovcnt;
			stringstream ss;
			ss << i;
			ofstream usedpointda(outputDir + "\\logdata" + ss.str() + ".txt");
			usedpointda << log << endl;
			usedpointda.close();
			sfba->WriteMotionAscii(outputDir + "\\" + ss.str() + ".csv");
		}
	}

	sfba->WriteMotionAscii(asciiout);
	sfba->WriteMotion(binout);
	sfba->getInput().psl->writePlyReflectance(config["output"].get<std::string>() + "\\" + "model_ba.ply", sfba->getInput().imTimeStamp, sfba->framePos.data(),
		sfba->getInput().extCalib, firstFrame, lastFrame, config["skip"].get<int>(), 0.00);
	std::ofstream ofsj2(outputDir + "\\config_out" + timeStamp + ".json");
	config["motion"] = binout;
	ofsj << std::setw(4) << config << std::endl;
	ofsj2.close();

	return 0;
}