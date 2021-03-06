#pragma once

#include <cv.h>
#include <highgui.h>
#include <cc_v5.h>
#include <pa_file/pa_file.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>

#include <windows.h>
#include "tinyxml/tinyxml.h"

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

#pragma comment(lib, "libcaffe5.0.lib")

using namespace std;
using namespace cv;
using namespace cc;

enum NMS_TYPE{
	MIN,
	UNION,
};

struct XMLInfo
{
	float xmin, ymin, xmax, ymax;
	string slabel;
	vector<Point2f> GetBoxPoint(){

		vector<Point2f> vecPointOut;
		Point2f tmpPoint;
		//左上
		tmpPoint.x = xmin;
		tmpPoint.y = ymin;
		vecPointOut.push_back(tmpPoint);

		//右上
		tmpPoint.x = xmax;
		tmpPoint.y = ymin;
		vecPointOut.push_back(tmpPoint);

		//左下
		tmpPoint.x = xmin;
		tmpPoint.y = ymax;
		vecPointOut.push_back(tmpPoint);

		//右下
		tmpPoint.x = xmax;
		tmpPoint.y = ymax;
		vecPointOut.push_back(tmpPoint);
		return vecPointOut;
	}
};

struct configure{
	string datapath;
	string xmlpath;
	int num_classes;
};

struct BBox{
	float xmin, ymin, xmax, ymax, score;
	int label;
	int fx, fy, a;
	int batch_id;
	float w() const{
		float tw = xmax - xmin;
		return tw + 1;
	}

	int ind(int fw, int fh){
		return fx + fy * fw + a * fw * fh;
	}

	float h() const{
		float th = ymax - ymin;
		return th + 1;
	}

	float area() const{ return w()*h(); }
	Rect box(){ return Rect(xmin, ymin, w(), h()); }
};

#define config (getConfigure())

configure& getConfigure();
int randr(int mi, int mx);

vector<vector<float>> bbox_overlaps(vector<BBox> gt_boxes, vector<BBox> boxes);

Mat bbox_transform(Mat ex_rois, Mat gt_rois);

map<string, string> parseParamStr(const char* str);
int getParamInt(map<string, string>& p, const string& key, int default_ = 0);

vector<BBox> generateAnchors(int base_size = 16, const vector<float>& ratios = { 0.5, 1, 2 }, const vector<float>& scales = { pow(2.0f, 3.0f), pow(2.0f, 4.0f), pow(2.0f, 5.0f) });

Mat makeShifts(int feat_stride, int width, int height);

Mat bbox_transform_inv(Mat boxes, Mat deltas);

Mat clip_boxes(Mat boxes, int height, int width);

vector<BBox> nms(vector<BBox>& objs, float nmsThreshold, vector<int>& keepinds = vector<int>(), NMS_TYPE type = UNION);

void caffe_set(int count, float val, float* ptr);

Scalar getColor(int label);

void augmentation_flip(Mat& img_src, vector<XMLInfo>& item_pts);

struct ConfigTrain{
	//anchor target layer
	int IMS_PER_BATCH = 1;
	const int SCALES = 600;
	const bool RPN_CLOBBER_POSITIVES = false;
	const float RPN_FG_FRACTION = 0.5;
	int RPN_BATCHSIZE = 256;
	const float RPN_NEGATIVE_OVERLAP = 0.3;//
	float RPN_POSITIVE_OVERLAP = 0.7;
	vector<float> RPN_BBOX_INSIDE_WEIGHTS; // inside 权重
	const float RPN_POSITIVE_WEIGHT = -1.0;

	//roi data layer
	const float MIN_SIZE = 600;
	const float MAX_SIZE = 1000;
	const float FG_FRACTION = 0.25;
	const int BATCH_SIZE = 128;

	const float FG_THRESH = 0.5;
	const float BG_THRESH_HI = 0.5;
	float BG_THRESH_LO = 0.0;
	const float BBOX_THRESH = 0.5;

	bool HAS_RPN = true;
	bool BBOX_REG = true;
	const bool RPN_BBOX_REG = true;

	//proposal
	const int RPN_PRE_NMS_TOP_N = 12000;
	const int RPN_POST_NMS_TOP_N = 2000;
	const float RPN_NMS_THRESH = 0.7;
	const int RPN_MIN_SIZE = 16;

	int SNAPSHOT_ITERS = 2000;
	string SNAPSHOT_PREFIX = "20190704/vgg16_faster_rcnn";
	string PARAM_SNAPSHOT_PREFIX = "20190704/param_vgg16_faster_rcnn";


	bool BBOX_NORMALIZE_TARGETS_PRECOMPUTED = false;
	const bool BBOX_NORMALIZE_TARGETS = true;

	vector<float> BBOX_NORMALIZE_MEANS;
	vector<float> BBOX_NORMALIZE_STDS;
	vector<float> BBOX_INSIDE_WEIGHTS;
	
	ConfigTrain() {
		BBOX_NORMALIZE_MEANS = { 0.0f, 0.0f, 0.0f, 0.0f };
		BBOX_NORMALIZE_STDS = { 0.1f, 0.1f, 0.2f, 0.2f };
		BBOX_INSIDE_WEIGHTS = { 1.0f, 1.0f, 1.0f, 1.0f };
		RPN_BBOX_INSIDE_WEIGHTS = { 1.0f, 1.0f, 1.0f, 1.0f };
	}
};

struct ConfigTest{
	const int RPN_PRE_NMS_TOP_N = 6000;
	const int RPN_POST_NMS_TOP_N = 300;
	const float RPN_NMS_THRESH = 0.7;
	const int RPN_MIN_SIZE = 16;
	const int SCALES = 600;
	const int MAX_SIZE = 1000;

	bool HAS_RPN = true;
	bool BBOX_REG = true;

	float NMS = 0.3f;
};

static struct Config{
	float EPS = 1e-14;
	Scalar PIXEL_MEANS = Scalar(102.9801, 115.9465, 122.7717);
	ConfigTrain TRAIN;
	ConfigTest TEST;
}cfg;

extern map<string, vector<BBox>> rpn_generate_bbox;
extern map<string, int> g_labelmap;
extern vector<string> labelmap;