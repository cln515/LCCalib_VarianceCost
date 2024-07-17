//Copyright © 2024 Ryoichi Ishikawa All right reserved.
#define _USE_MATH_DEFINES
#include <fstream>
#include <iostream>
#include <string>
#include "fusionba/utility.h"
#include <vector>

#include "Eigen/Core"
#include "Eigen/Eigen"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

#ifndef BASICPLY
#define BASICPLY

typedef map<int, int> IDXMAP;

class BasicPly{
private:
	typedef struct {
		unsigned char	nindex_;
		unsigned int			index0_, index1_, index2_;
	}ply_index;


protected:
	vector<POINT_3D> dataPtsVector;

	Matrix4d GlobalPose;
	Vector3d g;
	float* verteces;
	float* norm;
	float* reflectance;
	unsigned int* faces;
	unsigned char* rgba;//{r,g,b,a,r,g,b,a,...}
#if defined(_WIN32) || defined(_WIN64)
	__int64 facenum;
	__int64 vertexnum;
#elif defined(__unix__)
	long facenum;
	long vertexnum;
#endif
	
	double alpha;
	bool isANN;
	bool bRead;

	bool bG = false;
	bool bC = false;
public:

	enum PROP {
		PROP_XF,
		PROP_YF,
		PROP_ZF,
		PROP_XD,
		PROP_YD,
		PROP_ZD,
		PROP_INTENSITY,
		PROP_R,
		PROP_G,
		PROP_B,
		PROP_A,
		PROP_OTHER_D,
		PROP_OTHER_F,
		PROP_OTHER_UC
	};
	BasicPly(){
		vertexnum=0;
		facenum=0;
		bRead=false;
		norm = NULL;
		GlobalPose = Matrix4d::Identity();
	}

	bool readPlyFile(vector<string> fileName,int dataNum);
	bool readPlyFileRGB(vector<string> fileName,int dataNum);
	bool readPlyFile_(string fileName);

	bool isReflectance() { return bG; }
	bool isColor() { return bC; }

	void panoramaTexture(unsigned char* rgbArray,int width,int height,Matrix4d& transMat);

	int getVertexNumber(){return vertexnum;};
	int getFaceNumber(){return facenum;};
	void release();
	float* getVertecesPointer(){return verteces;};
	void setVertecesPointer(float* vertices,int vtnum){verteces=vertices;vertexnum=vtnum;};
	void setRgbaPointer(unsigned char* rgba_, int vtnum) { rgba = rgba_; vertexnum = vtnum; bC = true; };
	void setFacePointer(unsigned int* faces_, int fcnum) { faces = faces_;facenum = fcnum; };
	float* getNormPointer(){return norm;};
	float* getReflectancePointer(){return reflectance;};
	void setReflectancePointer(float* reflectance_, int vtnum) { reflectance = reflectance_; vertexnum = vtnum; bG = true; };
	unsigned int* getFaces(){return faces;};
	unsigned char* getRgbaPointer(){return rgba;};
	Vector3d getCentroid(){return g;};
	void computeNorm();
	void removeNoMeshPoints();
	void removeZeroPoints();
	void transform(Matrix4d m);

	void writePlyFile(string fileName);
	void writePlyFileAuto(string fileName);
//	void writePlyFileForMeshlab(string fileName);
	void writePlyFileRGB(string fileName);
	void writePlyFileRGBForMeshlab(string fileName);

	void convertFAROColoredPtx2Ply(string inFileName, double threshold);
	void convertFAROColoredPtx2SpherePly(string inFileName);

	void testOutput(){
		ofstream ofs("check_lm_model.ply",ios::out|ios::binary);
	ofs<<"ply"<<endl;
	ofs<<"format binary_little_endian 1.0"<<endl;
	ofs<<"element vertex "<<vertexnum<<endl;
	ofs<<"property float x"<<endl;
	ofs<<"property float y"<<endl;
	ofs<<"property float z"<<endl;
	ofs<<"property float confidence"<<endl;
	ofs<<"property float intensity"<<endl;
	ofs<<"element face 0"<<endl;
	ofs<<"property list uchar int vertex_index"<<endl;
	ofs<<"end_header"<<endl;
	int time;
	int i;
	for(i=0;i<vertexnum;i++){
	
		float fa[5];
		fa[0]=verteces[i*3];
		fa[1]=verteces[i*3+1];
		fa[2]=verteces[i*3+2];
		fa[3]=1.0f;
		fa[4]=reflectance[i];
		for(time=0;time<5;time++){
			ofs.write((char *) &fa[time],sizeof(float));	
		}
	}
	ofs.close();
	}
};

#endif // !REFDATA
