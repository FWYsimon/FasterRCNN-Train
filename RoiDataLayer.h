#pragma once

#include "common.h"
#include "readxml.h"

struct roi{
	vector<vector<float>> gt_overlaps;
	vector<XMLInfo> xmlInfo;
	vector<BBox> boxes;
	vector<float> max_overlaps;
	vector<float> max_classes;
	Mat bbox_targets;
};

class RoiDataLayer : public DataLayer {
public:
	SETUP_LAYERFUNC(RoiDataLayer);

	virtual int getBatchCacheSize(){ return 3; }
	virtual ~RoiDataLayer();

	void prepareData();

	//setup()
	virtual void setup(const char* name, const char* type, const char* param_str,
		int phase, Blob** bottom, int numBottom, Blob** top, int numTop);

	//loadBatch()
	virtual void loadBatch(Blob** top, int numTop);


private:
	

private:
	vector<string> _vecImageName;
	readxml _readXmlClass;

	//map<string, vector<XMLInfo>> _mapData;
	map<string, vector<vector<float>>> _map_gt_overlaps;

	map<string, roi> _map_data;

	

	bool _is_RPN;
	bool _is_BBox_Reg;
	int _num_classes;
	int _cursor = 0;

	Scalar means(102.9801, 115.9465, 122.7717);

	const char* classname[] = {
		"__background__",
		"aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair",
		"cow", "diningtable", "dog", "horse",
		"motorbike", "person", "pottedplant",
		"sheep", "sofa", "train", "tvmonitor"
	};

	map<string, int> g_labelmap = {
		{ "background", 0 },
		{ "aeroplane", 1 },
		{ "bicycle", 2 },
		{ "bird", 3 },
		{ "boat", 4 },
		{ "bottle", 5 },
		{ "bus", 6 },
		{ "car", 7 },
		{ "cat", 8 },
		{ "chair", 9 },
		{ "cow", 10 },
		{ "diningtable", 11 },
		{ "dog", 12 },
		{ "horse", 13 },
		{ "motorbike", 14 },
		{ "person", 15 },
		{ "pottedplant", 16 },
		{ "sheep", 17 },
		{ "sofa", 18 },
		{ "train", 19 },
		{ "tvmonitor", 20 }
	};

	vector<string> _vecClassName;
};