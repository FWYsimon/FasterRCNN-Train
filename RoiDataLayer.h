#pragma once

#include "common.h"
#include "readxml.h"

struct roi{
	vector<vector<float>> gt_overlaps;
	vector<XMLInfo> xmlInfo;
	vector<BBox> boxes;
	vector<int> max_overlaps;
	vector<int> max_classes;
	Mat bbox_targets;
};

class RoiDataLayer : public BaseLayer {
public:
	SETUP_LAYERFUNC(RoiDataLayer);

	void prepareData();
	void loadBatch(Blob** top, int numTop);
	//setup()
	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);

	//loadBatch()

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);
	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);

private:
	

private:
	vector<string> _vecImageName;
	readxml _readXmlClass;

	//map<string, vector<XMLInfo>> _mapData;
	map<string, vector<vector<float>>> _map_gt_overlaps;

	map<string, roi> _map_data;

	int _num_classes;
	int _cursor = 0;

	Scalar means;

	const char* classname;
	//= {
	//	"__background__",
	//	"aeroplane", "bicycle", "bird", "boat",
	//	"bottle", "bus", "car", "cat", "chair",
	//	"cow", "diningtable", "dog", "horse",
	//	"motorbike", "person", "pottedplant",
	//	"sheep", "sofa", "train", "tvmonitor"
	//};

	vector<string> _vecClassName;
};