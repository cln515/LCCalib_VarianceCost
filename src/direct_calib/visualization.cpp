//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#include <direct_calib/visualization.h>


void colorMapping(std::vector<Eigen::Vector4d> pc, std::vector<cv::Mat> imgs, std::vector<cv::Mat> masks,std::vector<int> usedImageIdx, 
	std::vector<Eigen::Matrix4d> tf_cams, std::vector<uchar>& color, std::vector<double> timestamps, std::vector<double> imtimestamp,camera*cam, double thres,unsigned char* posibility) {
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

		for(int uidx=0; uidx < usedImageIdx.size(); uidx++){
//		for (int i = 0; i < imgs.size(); i++) {
			int i = usedImageIdx.at(uidx);
			double ts_image = imtimestamp[i];
			if (thres > 0 && fabs(ts_point - ts_image) > thres)continue;
			Eigen::Matrix4d tf_cam = tf_cams.at(i).inverse();
			Eigen::Matrix3d R = tf_cam.block(0, 0, 3, 3);
			Eigen::Vector3d t = tf_cam.block(0, 3, 3, 1);
			double ix, iy;
			Eigen::Vector3d t_p = R * pc.at(pidx).segment(0, 3) + t;

			cam->projection(t_p, ix, iy);

			double d = t_p.norm();
			//getColorSubPixel(imgs.at(i).at(idx), cv::Point2d(ix, iy), rgbret);
			if (ix < 0 || iy < 0 || ix >= imgs.at(i).cols || iy >= imgs.at(i).rows)continue;
			getColorSubPixel_mask(imgs.at(i), masks.at(i), cv::Point2d(ix, iy), ret, weight);

			//if ((rgbret[0] == rgbret[1]) && (rgbret[0] == rgbret[2]) && (rgbret[2] == 0))continue;
			double preW = W;
			if(posibility!=NULL)weight = weight * (posibility[pidx + uidx * pc.size()] / 255.0);

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

		color.push_back(colord[0]);
		color.push_back(colord[1]);
		color.push_back(colord[2]);
	}
}

void imageProjection(std::string imageFileName ,cv::Mat projImage,std::vector<Eigen::Vector4d> points, std::vector<uchar> color,Eigen::Matrix4d camera_pose,
	int stPtIdx,int endPtIdx,camera* cam) {

	Eigen::Matrix4d campos_inv = camera_pose.inverse();
	Eigen::Matrix3d R = campos_inv.block(0, 0, 3, 3);
	Eigen::Vector3d t = campos_inv.block(0, 3, 3, 1);
	cv::Mat projimage_;
	projImage.copyTo(projimage_);

	for (long long pidx = 0; pidx < points.size(); pidx++) {
		
		if (points.at(pidx)(3) == 1)continue;
		if (color.at(pidx * 3) == 0 && color.at(pidx * 3 + 1) == 0 && color.at(pidx * 3 + 2) == 0)continue;

		double ix, iy;
		Eigen::Vector3d t_p = R * points.at(pidx).segment(0, 3) + t;
		double nrm = t_p.norm();
		double val = 1.0;

		cam->projection(t_p,ix,iy);
		cv::circle(projimage_, cv::Point2f(ix - 0.5, iy - 0.5), 0, cv::Scalar(color.at(pidx * 3), color.at(pidx * 3 + 1), color.at(pidx * 3 + 2)));
	
	}
	cv::imwrite(imageFileName, projimage_);
}


void edgeDetection(cv::Mat in,cv::Mat& out) {
	out.create(in.size(), in.type());
	float* outptr = (float*)out.data;
	float* inptr = (float*)in.data;
	int area = out.size().area();
	for (int i = 0; i < area; i++) {
		double edgev = 0;
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				int ox = i % out.cols;
				int oy = i / out.cols;
				if (ox + x < 0 || ox + x >= out.cols)continue;
				if (oy + y < 0 || oy + y >= out.rows)continue;
				int idx = (ox + x) + (oy+y)*out.cols;
				if (edgev < fabs(inptr[i] - inptr[idx])) {
					edgev = fabs(inptr[i] - inptr[idx]);
				};
			}
		}
		outptr[i] = edgev;
	}
}

//masked color

void getColorSubPixel_mask(cv::Mat image, cv::Mat mask, cv::Point2d p, double *ret, double& weight) {
	int xl = (int)(p.x - 0.5);
	int yt = (int)(p.y - 0.5);
	int xr = xl + 1;
	int yb = yt + 1;

	if (xl < 0)xl = 0;
	if (yt < 0)yt = 0;
	if (xr >= image.cols)xr = image.cols - 1;
	if (yb >= image.rows)yb = image.rows - 1;
	uchar* imArray = image.data;//3 channel

	float* maskArray = (float*)mask.data;//1 channel, float
	double dx = (p.x - 0.5) - xl;
	double dy = (p.y - 0.5) - yt;

	double rgb[3];

	double weights[4];
	weights[0] = maskArray[(xl + yt * mask.cols)];
	weights[1] = maskArray[(xr + yt * mask.cols)];
	weights[2] = maskArray[(xl + yb * mask.cols)];
	weights[3] = maskArray[(xr + yb * mask.cols)];
	weight = ((1 - dx)*(1 - dy)*weights[0] +
		(dx)*(1 - dy) *weights[1] +
		(1 - dx)*(dy)*weights[2] +
		(dx)*(dy)*weights[3]);

	if (weight == 0) {
		memset(ret, 0, sizeof(double) * 3);
		return;
	}


	//weights[0] /= weight;
	//weights[1] /= weight;
	//weights[2] /= weight;
	//weights[3] /= weight;

	rgb[0] = weights[0] * (1 - dx)*(1 - dy)*imArray[(xl + yt * image.cols) * 3]
		+ weights[1] * (dx)*(1 - dy) * imArray[(xr + yt * image.cols) * 3]
		+ weights[2] * (1 - dx)*(dy)* imArray[(xl + yb * image.cols) * 3]
		+ weights[3] * (dx)*(dy)* imArray[(xr + yb * image.cols) * 3];
	rgb[1] = weights[0] * (1 - dx)*(1 - dy)*imArray[(xl + yt * image.cols) * 3 + 1]
		+ weights[1] * (dx)*(1 - dy) * imArray[(xr + yt * image.cols) * 3 + 1]
		+ weights[2] * (1 - dx)*(dy)* imArray[(xl + yb * image.cols) * 3 + 1]
		+ weights[3] * (dx)*(dy)* imArray[(xr + yb * image.cols) * 3 + 1];
	rgb[2] = weights[0] * (1 - dx)*(1 - dy)*imArray[(xl + yt * image.cols) * 3 + 2]
		+ weights[1] * (dx)*(1 - dy) * imArray[(xr + yt * image.cols) * 3 + 2]
		+ weights[2] * (1 - dx)*(dy)* imArray[(xl + yb * image.cols) * 3 + 2]
		+ weights[3] * (dx)*(dy)* imArray[(xr + yb * image.cols) * 3 + 2];


	ret[0] = rgb[0] / weight;
	ret[1] = rgb[1] / weight;
	ret[2] = rgb[2] / weight;
}

void getColorSubPixel_diff_mask(cv::Mat image, cv::Mat mask, cv::Point2d p, double *ret, double * dret, double& weight) {
	int xl = (int)(p.x - 0.5);
	int yt = (int)(p.y - 0.5);
	int xr = xl + 1;
	int yb = yt + 1;

	if (xl < 0)xl = 0;
	if (yt < 0)yt = 0;
	if (xr >= image.cols)xr = image.cols - 1;
	if (yb >= image.rows)yb = image.rows - 1;
	uchar* imArray = image.data;//3 channel
	float* maskArray = (float*)mask.data;//1 channel, float
	double dx = (p.x - 0.5) - xl;
	double dy = (p.y - 0.5) - yt;


	double rgb[3];
	double rgbdiff[6];

	double weights[4];
	weights[0] = maskArray[(xl + yt * mask.cols)];
	weights[1] = maskArray[(xr + yt * mask.cols)];
	weights[2] = maskArray[(xl + yb * mask.cols)];
	weights[3] = maskArray[(xr + yb * mask.cols)];
	weight = ((1 - dx)*(1 - dy)*weights[0] +
		(dx)*(1 - dy) *weights[1] +
		(1 - dx)*(dy)*weights[2] +
		(dx)*(dy)*weights[3]);

	if (weight == 0) {
		memset(ret, 0, sizeof(double) * 3);
		memset(dret, 0, sizeof(double) * 8);
		return;
	}
	//weights[0] /= weight;
	//weights[1] /= weight;
	//weights[2] /= weight;
	//weights[3] /= weight;

	rgb[0] = weights[0] * (1 - dx)*(1 - dy)*imArray[(xl + yt * image.cols) * 3]
		+ weights[1] * (dx)*(1 - dy) * imArray[(xr + yt * image.cols) * 3]
		+ weights[2] * (1 - dx)*(dy)* imArray[(xl + yb * image.cols) * 3]
		+ weights[3] * (dx)*(dy)* imArray[(xr + yb * image.cols) * 3];
	rgb[1] = weights[0] * (1 - dx)*(1 - dy)*imArray[(xl + yt * image.cols) * 3 + 1]
		+ weights[1] * (dx)*(1 - dy) * imArray[(xr + yt * image.cols) * 3 + 1]
		+ weights[2] * (1 - dx)*(dy)* imArray[(xl + yb * image.cols) * 3 + 1]
		+ weights[3] * (dx)*(dy)* imArray[(xr + yb * image.cols) * 3 + 1];
	rgb[2] = weights[0] * (1 - dx)*(1 - dy)*imArray[(xl + yt * image.cols) * 3 + 2]
		+ weights[1] * (dx)*(1 - dy) * imArray[(xr + yt * image.cols) * 3 + 2]
		+ weights[2] * (1 - dx)*(dy)* imArray[(xl + yb * image.cols) * 3 + 2]
		+ weights[3] * (dx)*(dy)* imArray[(xr + yb * image.cols) * 3 + 2];
	//dx
	rgbdiff[0] = -weights[0] * (1 - dy)*imArray[(xl + yt * image.cols) * 3]
		+ weights[1] * (1 - dy) * imArray[(xr + yt * image.cols) * 3]
		+ -weights[2] * (dy)* imArray[(xl + yb * image.cols) * 3]
		+ weights[3] * (dy)* imArray[(xr + yb * image.cols) * 3];
	rgbdiff[1] = -weights[0] * (1 - dy)*imArray[(xl + yt * image.cols) * 3 + 1]
		+ weights[1] * (1 - dy) * imArray[(xr + yt * image.cols) * 3 + 1]
		+ -weights[2] * (dy)* imArray[(xl + yb * image.cols) * 3 + 1]
		+ weights[3] * (dy)* imArray[(xr + yb * image.cols) * 3 + 1];
	rgbdiff[2] = -weights[0] * (1 - dy)*imArray[(xl + yt * image.cols) * 3 + 2]
		+ weights[1] * (1 - dy) * imArray[(xr + yt * image.cols) * 3 + 2]
		+ -weights[2] * (dy)* imArray[(xl + yb * image.cols) * 3 + 2]
		+ weights[3] * (dy)* imArray[(xr + yb * image.cols) * 3 + 2];
	rgbdiff[3] = -weights[0] * (1 - dx)*imArray[(xl + yt * image.cols) * 3]
		+ -weights[1] * (dx)* imArray[(xr + yt * image.cols) * 3]
		+ weights[2] * (1 - dx)* imArray[(xl + yb * image.cols) * 3]
		+ weights[3] * (dx)* imArray[(xr + yb * image.cols) * 3];
	rgbdiff[4] = -weights[0] * (1 - dx)*imArray[(xl + yt * image.cols) * 3 + 1]
		+ -weights[1] * (dx)* imArray[(xr + yt * image.cols) * 3 + 1]
		+ weights[2] * (1 - dx)* imArray[(xl + yb * image.cols) * 3 + 1]
		+ weights[3] * (dx)* imArray[(xr + yb * image.cols) * 3 + 1];
	rgbdiff[5] = -weights[0] * (1 - dx)*imArray[(xl + yt * image.cols) * 3 + 2]
		+ -weights[1] * (dx)* imArray[(xr + yt * image.cols) * 3 + 2]
		+ weights[2] * (1 - dx)* imArray[(xl + yb * image.cols) * 3 + 2]
		+ weights[3] * (dx)* imArray[(xr + yb * image.cols) * 3 + 2];

	double wdiff[2];
	wdiff[0] = -weights[0] * (1 - dy)
		+ weights[1] * (1 - dy)
		+ -weights[2] * (dy)
		+weights[3] * (dy);

	wdiff[1] = -weights[0] * (1 - dx)
		+ -weights[1] * (dx)
		+weights[2] * (1 - dx)
		+ weights[3] * (dx);

	ret[0] = rgb[0] / weight;
	ret[1] = rgb[1] / weight;
	ret[2] = rgb[2] / weight;
	double w2 = weight * weight;
	dret[0] = (rgbdiff[0] * weight - rgb[0] * wdiff[0]) / w2;
	dret[1] = (rgbdiff[1] * weight - rgb[1] * wdiff[0]) / w2;
	dret[2] = (rgbdiff[2] * weight - rgb[2] * wdiff[0]) / w2;
	dret[3] = (rgbdiff[3] * weight - rgb[0] * wdiff[1]) / w2;
	dret[4] = (rgbdiff[4] * weight - rgb[1] * wdiff[1]) / w2;;
	dret[5] = (rgbdiff[5] * weight - rgb[2] * wdiff[1]) / w2;
	dret[6] = wdiff[0];
	dret[7] = wdiff[1];
}

void getGraySubPixel(cv::Mat image, cv::Point2f p, double *ret) {
	int xl = (int)(p.x - 0.5);
	int yt = (int)(p.y - 0.5);
	int xr = xl + 1;
	int yb = yt + 1;

	if (xl < 0)xl = 0;
	if (yt < 0)yt = 0;
	if (xr >= image.cols)xr = image.cols - 1;
	if (yb >= image.rows)yb = image.rows - 1;
	float* imArray = (float*)image.data;
	double dx = (p.x - 0.5) - xl;
	double dy = (p.y - 0.5) - yt;

	ret[0] = (1 - dx)*(1 - dy);
	ret[1] = imArray[xl + yt * image.cols];
	ret[2] = (dx)*(1 - dy);
	ret[3] = imArray[xr + yt * image.cols];
	ret[4] = (1 - dx)*(dy);
	ret[5] = imArray[xl + yb * image.cols];
	ret[6] = (dx)*(dy);
	ret[7] = imArray[xr + yb * image.cols];
}


void getGraySubPixel(cv::Mat image, cv::Mat mask, cv::Point2f p, double *ret, double& weight) {
	int xl = (int)(p.x - 0.5);
	int yt = (int)(p.y - 0.5);
	int xr = xl + 1;
	int yb = yt + 1;

	if (xl < 0)xl = 0;
	if (yt < 0)yt = 0;
	if (xr >= image.cols)xr = image.cols - 1;
	if (yb >= image.rows)yb = image.rows - 1;
	float* imArray = (float*)image.data;
	double dx = (p.x - 0.5) - xl;
	double dy = (p.y - 0.5) - yt;

	float* maskArray = (float*)mask.data;//1 channel, float

	double weights[4];
	weights[0] = maskArray[(xl + yt * mask.cols)];
	weights[1] = maskArray[(xr + yt * mask.cols)];
	weights[2] = maskArray[(xl + yb * mask.cols)];
	weights[3] = maskArray[(xr + yb * mask.cols)];
	weight = ((1 - dx)*(1 - dy)*weights[0] +
		(dx)*(1 - dy) *weights[1] +
		(1 - dx)*(dy)*weights[2] +
		(dx)*(dy)*weights[3]);

	if (weight == 0) {
		memset(ret, 0, sizeof(double) * 1);
		return;
	}

	ret[0] = weights[0]*(1 - dx)*(1 - dy) * imArray[xl + yt * image.cols]
		+ weights[1] * (dx)*(1 - dy) * imArray[xr + yt * image.cols]
	    + weights[2] * (1 - dx)*(dy) * imArray[xl + yb * image.cols];
		+ weights[3] * (dx)*(dy) * imArray[xr + yb * image.cols];
}