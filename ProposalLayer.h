#pragma once

#include "common.h"

class ProposalLayer : public BaseLayer {
public:
	SETUP_LAYERFUNC(ProposalLayer);

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop);

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop);

	virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);

	virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop);

private:
	int _feat_stride;

	vector<BBox> _anchors;
	int _num_anchors;
	int phase;
};