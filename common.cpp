#include "common.h"
#include "readxml.h"

configure& getConfigure(){
	static configure cfg;
	return cfg;
}

int randr(int mi, int mx){
	if (mi > mx) std::swap(mi, mx);
	int r = mx - mi + 1;
	return rand() % r + mi;
}

vector<vector<float>> bbox_overlaps(vector<BBox> gt_boxes, vector<BBox> boxes) {
	unsigned int N = boxes.size();
	unsigned int K = gt_boxes.size();
	vector<vector<float>> overlaps(N, vector<float>(K, 0));
	float iw, ih, box_area;
	float ua;
	unsigned int n, k;
	for (int k = 0; k < K; k++) {
		box_area = (gt_boxes[k].xmax - gt_boxes[k].xmin + 1) * (gt_boxes[k].ymax - gt_boxes[k].ymin + 1);
		for (int n = 0; n < N; n++) {
			iw = (min(boxes[n].xmax, gt_boxes[k].xmax) - max(boxes[n].xmin, gt_boxes[k].xmin) + 1);
			if (iw > 0) {
				ih = (min(boxes[n].ymax, gt_boxes[k].ymax) - max(boxes[n].ymin, gt_boxes[k].ymin) + 1);
				if (ih > 0) {
					ua = (boxes[n].xmax - boxes[n].xmin + 1) * (boxes[n].ymax - boxes[n].ymin + 1) + box_area - iw * ih;
					overlaps[n][k] = iw * ih / ua;
				}

			}
		}
	}
	return overlaps;
}

Mat bbox_transform(Mat ex_rois, Mat gt_rois) {
	Mat cp_ex_rois = ex_rois.clone();
	Mat cp_gt_rois = gt_rois.clone();
	Mat ex_widths = cp_ex_rois.col(2) - cp_ex_rois.col(0) + 1.0;
	Mat ex_heights = cp_ex_rois.col(3) - cp_ex_rois.col(1) + 1.0;
	Mat ex_ctr_x = cp_ex_rois.col(0) + 0.5 * ex_widths;
	Mat ex_ctr_y = cp_ex_rois.col(1) + 0.5 * ex_heights;

	Mat gt_widths = cp_gt_rois.col(2) - cp_gt_rois.col(0) + 1.0;
	Mat gt_heights = cp_gt_rois.col(3) - cp_gt_rois.col(1) + 1.0;
	Mat gt_ctr_x = cp_gt_rois.col(0) + 0.5 * gt_widths;
	Mat gt_ctr_y = cp_gt_rois.col(1) + 0.5 * gt_heights;

	Mat targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths;
	Mat targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights;
	Mat targets_dw, targets_dh;
	cv::log(gt_widths / ex_widths, targets_dw);
	cv::log(gt_heights / ex_heights, targets_dh);

	Mat targets(4, cp_gt_rois.rows, CV_32F);
	targets.row(0) = targets_dx.t();
	targets.row(1) = targets_dy.t();
	targets.row(2) = targets_dw.t();
	targets.row(3) = targets_dh.t();

	return targets.t();
}

map<string, string> parseParamStr(const char* str){
	map<string, string> o;
	if (str){
		char* prev = 0;
		char* p = (char*)str;
		int stage = 0;
		string name, value;

		while (*p){
			while (*p){ if (*p != ' ') break; p++; }
			prev = p;

			while (*p){ if (*p == ' ' || *p == ':') break; p++; }
			if (*p) name = string(prev, p);

			while (*p){ if (*p != ' ' && *p != ':' || *p == '\'') break; p++; }
			bool has_yh = *p == '\'';
			if (has_yh) p++;
			prev = p;

			while (*p){ if (has_yh && *p == '\'' || !has_yh && (*p == ' ' || *p == ';')) break; p++; }
			if (p != prev){
				value = string(prev, p);
				o[name] = value;

				p++;
				while (*p){ if (*p != ' ' && *p != ';' && *p != '\'') break; p++; }
			}
		}
	}
	return o;
}

int getParamInt(map<string, string>& p, const string& key, int default_){
	if (p.find(key) == p.end())
		return default_;
	return atoi(p[key].c_str());
}

void caffe_set(int count, float val, float* ptr){
	for (int i = 0; i < count; ++i)
		ptr[i] = val;
}

vector<BBox> generateAnchors(int base_size, const vector<float>& ratios, const vector<float>& scales){

	BBox base_anchor;
	base_anchor.xmin = 0;
	base_anchor.ymin = 0;
	base_anchor.xmax = base_size - 1;
	base_anchor.ymax = base_size - 1;

	vector<BBox> out, ratio_anchors;
	float w = base_anchor.w();
	float h = base_anchor.h();
	float x_ctr = base_anchor.xmin + (w - 1) * 0.5;
	float y_ctr = base_anchor.ymin + (h - 1) * 0.5;
	float size = w * h;
	vector<float> size_ratios(ratios.size());
	for (int i = 0; i < size_ratios.size(); ++i){
		size_ratios[i] = size / ratios[i];

		float ws = round(sqrt(size_ratios[i]));
		float hs = round(ws * ratios[i]);
		BBox box;
		box.xmin = x_ctr - 0.5 * (ws - 1);
		box.ymin = y_ctr - 0.5 * (hs - 1);
		box.xmax = x_ctr + 0.5 * (ws - 1);
		box.ymax = y_ctr + 0.5 * (hs - 1);
		ratio_anchors.emplace_back(box);
	}

	//scale_enum
	for (int j = 0; j < ratio_anchors.size(); ++j){
		for (int i = 0; i < scales.size(); ++i){
			BBox anchor = ratio_anchors[j];
			float w = anchor.w();
			float h = anchor.h();
			float x_ctr = anchor.xmin + (w - 1)*0.5;
			float y_ctr = anchor.ymin + (h - 1) * 0.5;
			float ws = w * scales[i];
			float hs = h * scales[i];

			BBox box;
			box.xmin = x_ctr - 0.5 * (ws - 1);
			box.ymin = y_ctr - 0.5 * (hs - 1);
			box.xmax = x_ctr + 0.5 * (ws - 1);
			box.ymax = y_ctr + 0.5 * (hs - 1);
			out.emplace_back(box);
		}
	}

	return out;
}

Mat makeShifts(int feat_stride, int width, int height){
	Mat m = Mat::zeros(width * height, 4, CV_32F);
	for (int i = 0; i < m.rows; ++i){
		m.at<float>(i, 0) = (i % width) * feat_stride;
		m.at<float>(i, 1) = (i / width) * feat_stride;
	}
	m.col(0).copyTo(m.col(2));
	m.col(1).copyTo(m.col(3));
	return m;
}

Mat bbox_transform_inv(Mat boxes, Mat deltas){

	Mat cpdeltas = deltas.clone();
	Mat boxescopy = boxes.clone();
	Mat widths = boxescopy.col(2) - boxescopy.col(0) + 1.0;
	Mat heights = boxescopy.col(3) - boxescopy.col(1) + 1.0;
	Mat ctr_x = boxescopy.col(0) + 0.5 * widths;
	Mat ctr_y = boxescopy.col(1) + 0.5 * heights;

	Mat dx = cpdeltas.col(0);
	Mat dy = cpdeltas.col(1);
	Mat dw = cpdeltas.col(2);
	Mat dh = cpdeltas.col(3);

	Mat pred_ctr_x = dx.mul(widths) + ctr_x;
	Mat pred_ctr_y = dy.mul(heights) + ctr_y;
	cv::exp(dw, dw);
	cv::exp(dh, dh);

	Mat pred_w = dw.mul(widths);
	Mat pred_h = dh.mul(heights);

	Mat predboxes(boxescopy.rows, 4, CV_32F);
	predboxes.col(0) = pred_ctr_x - 0.5 * pred_w;
	predboxes.col(1) = pred_ctr_y - 0.5 * pred_h;
	predboxes.col(2) = pred_ctr_x + 0.5 * pred_w;
	predboxes.col(3) = pred_ctr_y + 0.5 * pred_h;
	return predboxes;
}

Mat clip_boxes(Mat boxes, int height, int width){

	Mat out = boxes.clone();
	out.col(0) = (cv::max)((cv::min)(boxes.col(0), width - 1), 0);
	out.col(1) = (cv::max)((cv::min)(boxes.col(1), height - 1), 0);
	out.col(2) = (cv::max)((cv::min)(boxes.col(2), width - 1), 0);
	out.col(3) = (cv::max)((cv::min)(boxes.col(3), height - 1), 0);

	return out;
}

float IoU2(const BBox& a, const BBox& b, NMS_TYPE type){
	float xmax = max(a.xmin, b.xmin);
	float ymax = max(a.ymin, b.ymin);
	float xmin = min(a.xmax, b.xmax);
	float ymin = min(a.ymax, b.ymax);
	//Union

	float uw = max(xmin - xmax + 1, 0);
	float uh = max(ymin - ymax + 1, 0);
	float inter = uw * uh;

	if (type == UNION)
		return inter / min(a.area(), b.area());
	else
		return inter / (a.area() + b.area() - inter);
}

vector<BBox> nms(vector<BBox>& objs, float nmsThreshold, vector<int>& keepinds, NMS_TYPE type){
	std::sort(objs.begin(), objs.end(), [](const BBox& a, const BBox& b){
		return a.score > b.score;
	});

	keepinds.clear();
	vector<BBox> out;
	vector<int> flags(objs.size(), 0);
	for (int i = 0; i < objs.size(); ++i){
		if (flags[i] == 1) continue;

		out.push_back(objs[i]);
		keepinds.push_back(i);

		flags[i] = 1;
		for (int j = i + 1; j < objs.size(); ++j){
			if (flags[j] == 0){
				float iouUnion = IoU2(objs[i], objs[j], type);
				if (iouUnion >= nmsThreshold)
					flags[j] = 1;
			}
		}
	}
	return out;
}

map<string, vector<BBox>> rpn_generate_bbox;

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