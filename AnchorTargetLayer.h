#pragma once

#include "common.h"

class AnchorTargetLayer : public BaseLayer {
public:
	SETUP_LAYERFUNC(AnchorTargetLayer);

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);

	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);

	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);

private:
	vector<BBox> _anchors;
	int _num_anchors;
	int _feat_stride;
	int _allowed_border;

private:
	
	Mat unmap(Mat data, int total_anchors, vector<int> inds_inside, int fill);
};