#include "ProposalTargetLayer.h"

void ProposalTargetLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	map<string, string> param = parseParamStr(param_str);

	this->_num_classes = getParamInt(param, "num_classes");

	Blob* rois = top[0];
	Blob* labels = top[1];
	Blob* bbox_targets = top[2];
	Blob* bbox_inside_weights = top[3];
	Blob* bbox_outside_weights = top[4];

	rois->reshape(1, 5);
	labels->reshape(1, 1);
	bbox_targets->reshape(1, _num_classes * 4);
	bbox_inside_weights->reshape(1, _num_classes * 4);
	bbox_outside_weights->reshape(1, _num_classes * 4);
}

void ProposalTargetLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* all_rois_blob = bottom[0];
	Blob* gt_boxes_blob = bottom[1];

	float* all_rois_ptr = all_rois_blob->mutable_cpu_data();
	float* gt_boxes_ptr = gt_boxes_blob->mutable_cpu_data();

	int num_images = 1;
	int rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images;
	int fg_rois_per_image = cvRound(cfg.TRAIN.FG_FRACTION * rois_per_image);

	vector<BBox> all_rois_boxes;
	for (int i = 0; i < all_rois_blob->num(); ++i) {
		BBox bbox;
		bbox.xmin = all_rois_ptr[1];
		bbox.ymin = all_rois_ptr[2];
		bbox.xmax = all_rois_ptr[3];
		bbox.ymax = all_rois_ptr[4];
		bbox.batch_id = all_rois_ptr[0];

		all_rois_boxes.push_back(bbox);
		all_rois_ptr += 5;
	}

	vector<BBox> gt_boxes;
	for (int i = 0; i < gt_boxes_blob->num(); ++i) {
		BBox bbox;
		bbox.xmin = gt_boxes_ptr[0];
		bbox.ymin = gt_boxes_ptr[1];
		bbox.xmax = gt_boxes_ptr[2];
		bbox.ymax = gt_boxes_ptr[3];
		bbox.label = gt_boxes_ptr[4];
		bbox.batch_id = 0;

		all_rois_boxes.push_back(bbox);
		gt_boxes.push_back(bbox);
		gt_boxes_ptr += 5;
	}

	vector<vector<float>> overlaps = bbox_overlaps(all_rois_boxes, gt_boxes);
	vector<int> gt_assignment;
	vector<float> max_overlaps;
	vector<int> labels;
	
	vector<int> fg_inds;
	vector<int> bg_inds;
	for (int i = 0; i < overlaps.size(); ++i) {
		vector<float>::iterator biggest = max_element(overlaps[i].begin(), overlaps[i].end());
		int index = distance(begin(overlaps[i]), biggest);
		
		gt_assignment.push_back(index);
		max_overlaps.push_back(*biggest);
		labels.push_back(gt_boxes[index].label);

		if (*biggest >= cfg.TRAIN.FG_THRESH)
			fg_inds.push_back(i);

		if (*biggest < cfg.TRAIN.BG_THRESH_HI && *biggest >= cfg.TRAIN.BG_THRESH_LO)
			bg_inds.push_back(i);
	}

	int fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size());
	
	if (fg_inds.size() >= 0) {
		random_shuffle(fg_inds.begin(), fg_inds.end());
		if (fg_rois_per_this_image < fg_inds.size())
			fg_inds.erase(fg_inds.begin() + fg_rois_per_this_image, fg_inds.end());
	}

	int bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
	bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size());
	
	if (bg_inds.size() >= 0) {
		random_shuffle(bg_inds.begin(), bg_inds.end());
		if (bg_rois_per_this_image < bg_inds.size())
			bg_inds.erase(bg_inds.begin() + bg_rois_per_this_image, bg_inds.end());
	}

	vector<int> keep_inds = fg_inds;
	keep_inds.insert(keep_inds.end(), bg_inds.begin(), bg_inds.end());

	vector<int> keep_labels;
	vector<BBox> rois;
	vector<BBox> keep_gt_boxes;
	for (int i = 0; i < keep_inds.size(); ++i) {
		keep_labels.push_back(labels[keep_inds[i]]);
		rois.push_back(all_rois_boxes[keep_inds[i]]);
		keep_gt_boxes.push_back(gt_boxes[gt_assignment[keep_inds[i]]]);
	}

	for (int i = fg_inds.size(); i < keep_inds.size(); ++i)
		keep_labels[i] = 0;

	Mat rois_mat;
	Mat keep_gt_boxes_mat;
	for (int i = 0; i < rois.size(); ++i) {
		vector<float> vec_roi = { rois[i].xmin, rois[i].ymin, rois[i].xmax, rois[i].ymax };
		vector<float> vec_gt_box = { keep_gt_boxes[i].xmin, keep_gt_boxes[i].ymin, keep_gt_boxes[i].xmax, keep_gt_boxes[i].ymax };
		rois_mat.push_back(Mat(1, vec_roi.size(), CV_32F, vec_roi.data()));
		keep_gt_boxes_mat.push_back(Mat(1, vec_gt_box.size(), CV_32F, vec_gt_box.data()));
	}

	Mat targets = bbox_transform(rois_mat, keep_gt_boxes_mat);
	if (cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
		for (int i = 0; i < targets.cols; ++i) {
			targets.col(i) -= cfg.TRAIN.BBOX_NORMALIZE_MEANS[i];
			targets.col(i) /= cfg.TRAIN.BBOX_NORMALIZE_STDS[i];
		}
	}

	int num_samples = rois.size();
	Blob* rois_blob = top[0];
	Blob* labels_blob = top[1];
	Blob* bbox_targets = top[2];
	Blob* bbox_inside_weights = top[3];
	Blob* bbox_outside_weights = top[4];

	rois_blob->reshape(num_samples, 5, 1, 1);
	labels_blob->reshape(num_samples, 1, 1, 1);
	bbox_targets->reshape(num_samples, 84, 1, 1);
	bbox_inside_weights->reshape(num_samples, 84, 1, 1);
	bbox_outside_weights->reshape(num_samples, 84, 1, 1);

	caffe_set(rois_blob->count(0), 0.0f, rois_blob->mutable_cpu_data());
	caffe_set(labels_blob->count(0), 0.0f, labels_blob->mutable_cpu_data());
	caffe_set(bbox_targets->count(0), 0.0f, bbox_targets->mutable_cpu_data());
	caffe_set(bbox_inside_weights->count(0), 0.0f, bbox_inside_weights->mutable_cpu_data());
	caffe_set(bbox_outside_weights->count(0), 0.0f, bbox_outside_weights->mutable_cpu_data());

	for (int i = 0; i < num_samples; ++i) {
		float* rois_ptr = rois_blob->mutable_cpu_data() + i * rois_blob->count(1);
		float* labels_ptr = labels_blob->mutable_cpu_data() + i * labels_blob->count(1);
		float* bbox_targets_ptr = bbox_targets->mutable_cpu_data() + i * bbox_targets->count(1);
		float* bbox_inside_weights_ptr = bbox_inside_weights->mutable_cpu_data() + i * bbox_inside_weights->count(1);
		float* bbox_outside_weights_ptr = bbox_outside_weights->mutable_cpu_data() + i * bbox_outside_weights->count(1);

		rois_ptr[0] = rois[i].batch_id;
		rois_ptr[1] = rois[i].xmin;
		rois_ptr[2] = rois[i].ymin;
		rois_ptr[3] = rois[i].xmax;
		rois_ptr[4] = rois[i].ymax;
		labels_ptr[0] = keep_labels[i];

		if (keep_labels[i] != 0) {
			bbox_targets_ptr[keep_labels[i] * 4 + 0] = targets.at<float>(i, 0);
			bbox_targets_ptr[keep_labels[i] * 4 + 1] = targets.at<float>(i, 1);
			bbox_targets_ptr[keep_labels[i] * 4 + 2] = targets.at<float>(i, 2);
			bbox_targets_ptr[keep_labels[i] * 4 + 3] = targets.at<float>(i, 3);

			bbox_inside_weights_ptr[keep_labels[i] * 4 + 0] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[0];
			bbox_inside_weights_ptr[keep_labels[i] * 4 + 1] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[1];
			bbox_inside_weights_ptr[keep_labels[i] * 4 + 2] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[2];
			bbox_inside_weights_ptr[keep_labels[i] * 4 + 3] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[3];

			bbox_outside_weights_ptr[keep_labels[i] * 4 + 0] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[0];
			bbox_outside_weights_ptr[keep_labels[i] * 4 + 1] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[1];
			bbox_outside_weights_ptr[keep_labels[i] * 4 + 2] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[2];
			bbox_outside_weights_ptr[keep_labels[i] * 4 + 3] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[3];
		}
	}
}

void ProposalTargetLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}

void ProposalTargetLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

}