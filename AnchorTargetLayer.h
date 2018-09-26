#include "common.h"

class AnchorTargetLayer : public AbstractCustomLayer {
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
	vector<BBox> generate_anchors(int base_size = 16, const vector<float>& ratios = { 0.5, 1, 2 }, const vector<float>& scales = { pow(2.0f, 3.0f), pow(2.0f, 4.0f), pow(2.0f, 5.0f) });

	Mat makeShifts(int feat_stride, int width, int height);
	
	Mat unmap(Mat data, int total_anchors, vector<int> inds_inside, int fill);
};