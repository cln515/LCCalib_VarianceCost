//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <iostream>
#include <fstream>
#include <nlohmann\json.hpp>
#include <direct_calib/fusion.h>


class fusionOptimizeFunction
{
	public:
	fusionOptimizeFunction(fusionscan* f_) {
		f = f_;
	}
	double operator() (const column_vector& arg) const
	{
		// Return the mean squared error between the target vector and the input vector.
		return f->costComputation(arg);
	}

private:
	fusionscan* f;
};

template <typename T> T getConfig(nlohmann::json j, std::string key,T default_val) {
	if (j[key].is_null()) {
		return default_val;
	}
	else {
		return j[key].get<T>();
	}
}


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

int main(int argc, char* argv[]) {
	std::cout << "Kitty on your lap!!" << std::endl;

	std::ifstream ifs(argv[1]);
	nlohmann::json config;
	ifs >> config;
	ifs.close();

	fusionscan fs;
	fs.setscan(config["scan"].get<std::string>());
	fs.setInitialMotion(config["initial_motion"].get<std::string>());
	std::string outfolder = config["outfolder"].get<std::string>();
	std::string outCalFile = outfolder + "\\out_calib.json";
	makeFolder(outCalFile);

	int setImageNum = getConfig<int>(config, "setImageNum", 30);
	int setPointDSNum = getConfig<int>(config, "setPointDSNum", 5);
	int setOutImg = getConfig<int>(config, "outputImageNum", 3);
	double timeThreshold = getConfig<double>(config, "timeThresh", 5.0);

	double coarse2fine = getConfig<double>(config, "coarse2fine", 1.0);

	fs.setCamera(config, coarse2fine);
	fs.setImage(setImageNum, coarse2fine);
	fs.setPoints(setPointDSNum);
	fs.setInitCalib(config);
	fs.setTimeThreshold(timeThreshold);
	if (!config["mesh"].is_null()) {//occlusion handring
		fs.visiblePosibilityCalc(config["mesh"].get<std::string>());
	}	
	if (!config["edge"].is_null()) {
		fs.setEdge(config);
	}
	fs.setMask(config, coarse2fine);


	//	fs.deformation(Eigen::Vector4d(0, 0, 0, 1), Eigen::Vector3d(0, 0, 1));
	column_vector starting_point(13);
	starting_point = 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1;

	if (!config["optparams"].is_null()) {
		for (int i = 0; i < 13; i++)starting_point(i) = config["optparams"].get<std::vector<double>>()[i];
	}

	std::string outPath = outfolder + "\\vis_init";
	fs.visualize(starting_point, outPath, setOutImg);

	if(true) {//experimental
		fs.pointColorWeightComputation();	
	}


	if (config["strategy"].is_null()) {
		dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-5),
			fusionOptimizeFunction(&fs), starting_point, -1);
		std::cout << "test_function solution:\n" << starting_point << std::endl;
	} else{
		std::vector<std::vector<int>> optstrategy = config["strategy"].get<std::vector<std::vector<int>>>();

		int optitr = config["strategy_itr"].get <int>();		

		//std::ofstream plot("plot.csv");

		//plot << "qx,qy,qz,x,y,z" << std::endl;
		for (int i = 0; i < optitr; i++) {
			for (int j = 0; j < optstrategy.size(); j++) {
				fs.optvalue = optstrategy.at(j);
				fs.distModeCheck();
				fs.m_ = starting_point;
				
				//for (int i = -100; i < 100; i++) {
				//	for (int j = 6; j < 12; j++) {
				//		column_vector opt_point(fs.optvalue.size());
				//		opt_point(0) = 0;
				//		opt_point(1) = 0;
				//		opt_point(2) = 0;
				//		opt_point(3) = 0;
				//		opt_point(4) = 0;
				//		opt_point(5) = 0;
				//		opt_point(6) = 0;
				//		opt_point(7) = 0;
				//		opt_point(8) = 0;
				//		opt_point(9) = 0;
				//		opt_point(10) = 0;
				//		opt_point(11) = 0;
				//		opt_point(j) = i * 0.0005;
				//		if (j < 9) {
				//			fs.directModeSet(1);
				//		}
				//		else {
				//			fs.directModeSet(-1);
				//		}

				//		plot << fs.costComputation(opt_point) << ",";
				//	}
				//	plot << std::endl;
				//}

				column_vector opt_point(fs.optvalue.size());
				for (int k = 0; k < fs.optvalue.size(); k++) {
					opt_point(k) = starting_point(fs.optvalue[k]);
				}

				dlib::find_min_using_approximate_derivatives(dlib::cg_search_strategy(),
					dlib::objective_delta_stop_strategy(1e-4),
					fusionOptimizeFunction(&fs), opt_point, -1,1e-7);
				for (int k = 0; k < fs.optvalue.size(); k++) {
					
					starting_point(fs.optvalue[k]) = opt_point(k);
				}
				std::cout << "test_function solution:\n" << starting_point << std::endl;

			}
		}
	
	}

	 outPath = outfolder + "\\vis_opt";
	fs.visualize(starting_point,outPath,setOutImg);
	fs.outputCalib(starting_point,outCalFile);

	//fs.costComputation();

	return 0;
}
