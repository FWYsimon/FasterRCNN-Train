#pragma once

#include <cv.h>
#include <highgui.h>
#include <cc_utils.h>
#include <cc.h>
#include <pa_file/pa_file.h>
#include <math.h>

#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))

using namespace std;
using namespace cv;

struct configure{
	string datapath;
};

struct BBox{
	float xmin, ymin, xmax, ymax, score;
	int label;
	int fx, fy, a;
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

map<string, vector<BBox>> rpn_generate_bbox;

vector<vector<float>> bbox_overlaps(vector<BBox> gt_boxes, vector<BBox> boxes);

Mat bbox_transform(Mat ex_rois, Mat gt_rois);

struct ConfigTrain{
	//anchor target layer
	const int IMS_PER_BATCH = 1;
	const int SCALES = 600;
	const int MAX_SIZE = 1000;
	const bool RPN_CLOBBER_POSITIVES = false;
	const float RPN_FG_FRACTION = 0.5;
	const int RPN_BATCHSIZE = 256;
	const float RPN_NEGATIVE_OVERLAP = 0.3;//
	const float RPN_POSITIVE_OVERLAP = 0.7;
	const float RPN_BBOX_INSIDE_WEIGHTS = 1.0; // inside ШЈжи
	const float RPN_POSITIVE_WEIGHT = -1.0;

	//roi data layer
	const float MIN_SIZE = 600;
	const float MAX_SIZE = 1000;
	const float FG_FRACTION = 0.25;
	const int BATCH_SIZE = 128;

	const float FG_THRESH = 0.5;
	const float BG_THRESH_HI = 0.5;
	const float BG_THRESH_LO = 0.1;
	const float BBOX_THRESH = 0.5;

	const bool BBOX_REG = true;
	const bool RPN_BBOX_REG = true;

	const bool BBOX_NORMALIZE_TARGETS_PRECOMPUTED = false;
	const bool BBOX_NORMALIZE_TARGETS = true;

	const vector<float> BBOX_NORMALIZE_MEANS = { 0.0, 0.0, 0.0, 0.0 };
	const vector<float> BBOX_NORMALIZE_STDS = { 0.1, 0.1, 0.2, 0.2 };
	const vector<float> BBOX_INSIDE_WEIGHTS = { 1.0, 1.0, 1.0, 1.0 };
	
};

struct Config{
	float EPS = 1e-14;
	ConfigTrain TRAIN;
}cfg;