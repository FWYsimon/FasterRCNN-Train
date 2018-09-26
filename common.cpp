#include "common.h"

configure& getConfigure(){
	static configure cfg;
	return cfg;
}

int randr(int mi, int mx){
	if (mi > mx) std::swap(mi, mx);
	int r = mx - mi + 1;
	return rand() % r + mi;
}

vector<vector<float>> bbox_overlaps(vector<XMLInfo> gt_boxes, vector<BBox> boxes) {
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
	Mat targets_dw = cv::log(gt_widths / ex_widths);
	Mat targets_dh = cv::log(gt_heights / ex_heights);

	Mat targets(4, cp_gt_rois.cols, CV_32F);
	targets.row(0) = targets_dx;
	targets.row(1) = targets_dy;
	targets.row(2) = targets_dw;
	targets.row(3) = targets_dh;

	return targets.t();
}