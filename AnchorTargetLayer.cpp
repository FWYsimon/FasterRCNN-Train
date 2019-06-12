#include "AnchorTargetLayer.h"

void AnchorTargetLayer::setup(const char* name, const char* type, const char* param_str, 
	int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	map<string, string> params = parseParamStr(param_str);

	_anchors = generateAnchors(16, { 0.5, 1, 2 }, { 8, 16, 32 });
	_num_anchors = _anchors.size();
	_feat_stride = getParamInt(params, "feat_stride");
	_allowed_border = 0;

	Blob* data = bottom[0];
	int height = data->height();
	int width = data->width();

	Blob* rpn_labels = top[0];
	Blob* rpn_bbox_targets = top[1];
	Blob* rpn_bbox_inside_weights = top[2];
	Blob* rpn_bbox_outside_weights = top[3];
	
	rpn_labels->reshape(1, 1, _num_anchors * height, width);
	rpn_bbox_targets->reshape(1, _num_anchors * 4, height, width);
	rpn_bbox_inside_weights->reshapeLike(rpn_bbox_targets);
	rpn_bbox_outside_weights->reshapeLike(rpn_bbox_targets);
}

void AnchorTargetLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* rpn_cls_score = bottom[0];
	Blob* gt_boxes = bottom[1];
	Blob* im_info = bottom[2];

	int height = rpn_cls_score->height();
	int width = rpn_cls_score->width();

	Mat shifts = makeShifts(_feat_stride, width, height);

	int A = _num_anchors;
	int K = shifts.rows;

	Mat anchors(A * K, 4, CV_32F);
	for (int i = 0; i < anchors.rows; ++i){
		//int k = i % K;
		//int a = i / K;
		int k = i / A;
		int a = i % A;
		//_anchors[k]
		anchors.at<float>(i, 0) = _anchors[a].xmin + shifts.at<float>(k, 0);
		anchors.at<float>(i, 1) = _anchors[a].ymin + shifts.at<float>(k, 1);
		anchors.at<float>(i, 2) = _anchors[a].xmax + shifts.at<float>(k, 2);
		anchors.at<float>(i, 3) = _anchors[a].ymax + shifts.at<float>(k, 3);
	}

	int total_anchors = K * A;

	int imh = im_info->mutable_cpu_data()[0];
	int imw = im_info->mutable_cpu_data()[1];
	float im_scale = im_info->mutable_cpu_data()[2];

	
	vector<BBox> keep_anchors;
	vector<int> keep_inds;
	for (int i = 0; i < anchors.rows; i++) {
		if ((anchors.at<float>(i, 0) >= -_allowed_border)
			&& (anchors.at<float>(i, 1) >= -_allowed_border)
			&& (anchors.at<float>(i, 2) < imw + _allowed_border)
			&& (anchors.at<float>(i, 3) < imh + _allowed_border))
		{
			BBox bbox;
			bbox.xmin = anchors.at<float>(i, 0);
			bbox.ymin = anchors.at<float>(i, 1);
			bbox.xmax = anchors.at<float>(i, 2);
			bbox.ymax = anchors.at<float>(i, 3);

			keep_anchors.push_back(bbox);
			keep_inds.push_back(i);
		}
	}

	Mat keep_anchors_mat(keep_anchors.size(), 4, CV_32F);
	for (int i = 0; i < keep_anchors.size(); i++) {
		keep_anchors_mat.at<float>(i, 0) = keep_anchors[i].xmin;
		keep_anchors_mat.at<float>(i, 1) = keep_anchors[i].ymin;
		keep_anchors_mat.at<float>(i, 2) = keep_anchors[i].xmax;
		keep_anchors_mat.at<float>(i, 3) = keep_anchors[i].ymax;
	}

	vector<int> labels(keep_inds.size(), -1);
	
	Mat gt_boxes_mat(gt_boxes->num(), gt_boxes->channel(), CV_32F, gt_boxes->mutable_cpu_data());
	vector<BBox> vec_gt_boxes;
	for (int i = 0; i < gt_boxes_mat.rows; i++) {
		BBox bbox;
		bbox.xmin = gt_boxes_mat.at<float>(i, 0);
		bbox.ymin = gt_boxes_mat.at<float>(i, 1);
		bbox.xmax = gt_boxes_mat.at<float>(i, 2);
		bbox.ymax = gt_boxes_mat.at<float>(i, 3);
		vec_gt_boxes.push_back(bbox);
	}
	vector<vector<float>> overlaps = bbox_overlaps(vec_gt_boxes, keep_anchors);

	vector<int> argmax_overlaps;
	vector<float> max_overlaps;
	for (int i = 0; i < overlaps.size(); i++) {
		vector<float>::iterator biggest = max_element(begin(overlaps[i]), end(overlaps[i]));
		argmax_overlaps.push_back((int)distance(begin(overlaps[i]), biggest));
		max_overlaps.push_back(*biggest);
	}
	
	vector<int> gt_argmax_overlaps;
	vector<float> gt_max_overlaps;
	for (int i = 0; i < overlaps[0].size(); i++) {
		float max_num = -FLT_MAX;
		int max_index = 0;
		for (int j = 0; j < overlaps.size(); j++) {
			float num = overlaps[j][i];
			if (num > max_num) {
				max_num = num;
				max_index = j;
			}
		}
		gt_max_overlaps.push_back(max_num);
		gt_argmax_overlaps.push_back(max_index);
	}

	gt_argmax_overlaps.clear();
	for (int i = 0; i < overlaps.size(); ++i) {
		for (int j = 0; j < overlaps[i].size(); ++j) {
			if (gt_max_overlaps[j] == overlaps[i][j]) {
				gt_argmax_overlaps.push_back(i);
			}
		}
	}

	if (!cfg.TRAIN.RPN_CLOBBER_POSITIVES) {
		for (int i = 0; i < max_overlaps.size(); i++)
			if (max_overlaps[i] < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
				labels[i] = 0;
	}

	for (int i = 0; i < gt_argmax_overlaps.size(); i++)
		labels[gt_argmax_overlaps[i]] = 1;

	for (int i = 0; i < max_overlaps.size(); i++)
		if (max_overlaps[i] >= cfg.TRAIN.RPN_POSITIVE_OVERLAP)
			labels[i] = 1;

	if (cfg.TRAIN.RPN_CLOBBER_POSITIVES)
		for (int i = 0; i < max_overlaps.size(); i++)
			if (max_overlaps[i] < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
				labels[i] = 0;

	int num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE);
	vector<int> fg_inds;
	for (int i = 0; i < labels.size(); i++)
		if (labels[i] == 1)
			fg_inds.push_back(i);

	if (fg_inds.size() > num_fg) {
		random_shuffle(fg_inds.begin(), fg_inds.end());
		for (int i = num_fg; i < fg_inds.size(); ++i)
			labels[fg_inds[i]] = -1;
		//fg_inds.erase(fg_inds.end() - num, fg_inds.end());
	}

	int num_bg = cfg.TRAIN.RPN_BATCHSIZE - (num_fg < fg_inds.size() ? num_fg : fg_inds.size());
	vector<int> bg_inds;
	for (int i = 0; i < labels.size(); i++)
		if (labels[i] == 0)
			bg_inds.push_back(i);

	if (bg_inds.size() > num_bg) {
		random_shuffle(bg_inds.begin(), bg_inds.end());
		for (int i = num_bg; i < bg_inds.size(); ++i)
			labels[bg_inds[i]] = -1;
	}

	//vector<vector<float>> bbox_targets(keep_inds.size(), vector<float>(4, 0));
	Mat gt_boxes_max_mat(argmax_overlaps.size(), 4, CV_32F);
	for (int i = 0; i < argmax_overlaps.size(); i++) {
		gt_boxes_max_mat.at<float>(i, 0) = gt_boxes_mat.at<float>(argmax_overlaps[i], 0);
		gt_boxes_max_mat.at<float>(i, 1) = gt_boxes_mat.at<float>(argmax_overlaps[i], 1);
		gt_boxes_max_mat.at<float>(i, 2) = gt_boxes_mat.at<float>(argmax_overlaps[i], 2);
		gt_boxes_max_mat.at<float>(i, 3) = gt_boxes_mat.at<float>(argmax_overlaps[i], 3);
	}
		

	Mat bbox_targets = bbox_transform(keep_anchors_mat, gt_boxes_max_mat);

	Mat bbox_inside_weights(keep_anchors.size(), 4, CV_32F, Scalar(0));
	int num_examples = 0;
	int num_positive = 0;
	for (int i = 0; i < bbox_inside_weights.rows; i++) {
		if (labels[i] == 1) {
			bbox_inside_weights.at<float>(i, 0) = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0];
			bbox_inside_weights.at<float>(i, 1) = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[1];
			bbox_inside_weights.at<float>(i, 2) = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[2];
			bbox_inside_weights.at<float>(i, 3) = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[3];
			num_positive++;
		}
		if (labels[i] >= 0)
			num_examples++;
	}

	//Mat bbox_outside_weights(keep_anchors.size(), 4, CV_32F, Scalar(0));
	Mat positive_weights(1, 4, CV_32F, Scalar(1));
	Mat negative_weights(1, 4, CV_32F, Scalar(1));
	if (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0) {
		positive_weights /= num_examples;
		negative_weights /= num_examples;
	}
	else {
		positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / num_positive;
		negative_weights = (1 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / (num_examples - num_positive);
	}

	Mat bbox_outside_weights(keep_anchors.size(), 4, CV_32F, Scalar(0));
	for (int i = 0; i < bbox_outside_weights.rows; i++) {
		if (labels[i] == 1)
			bbox_outside_weights.row(i) = positive_weights.row(0);
		else if (labels[i] == 0)
			bbox_outside_weights.row(i) = negative_weights.row(0);
	}

	Mat labels_mat(keep_inds.size(), 1, CV_32F);
	for (int i = 0; i < labels_mat.rows; i++)
		labels_mat.at<float>(i) = labels[i];

	labels_mat = unmap(labels_mat, total_anchors, keep_inds, -1);
	bbox_targets = unmap(bbox_targets, total_anchors, keep_inds, 0); //bbox_target (9*14*14, 4) 每36个数变为一行 为图的第一个像素的9个anchor的4个偏移值
	bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, keep_inds, 0);
	bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, keep_inds, 0);

	bbox_targets = bbox_targets.reshape(0, height * width);
	bbox_inside_weights = bbox_inside_weights.reshape(0, height * width);
	bbox_outside_weights = bbox_outside_weights.reshape(0, height * width);

	Blob* label_blob = top[0];
	Blob* bbox_targets_blob = top[1];
	Blob* bbox_targets_inside_blob = top[2];
	Blob* bbox_targets_outside_blob = top[3];
	label_blob->reshape(1, 1, A * height, width);
	bbox_targets_blob->reshape(1, A * 4, height, width);
	bbox_targets_inside_blob->reshapeLike(bbox_targets_blob);
	bbox_targets_outside_blob->reshapeLike(bbox_targets_blob);

	float* label_blob_ptr = label_blob->mutable_cpu_data();
	float* bbox_targets_blob_ptr = bbox_targets_blob->mutable_cpu_data();
	float* bbox_targets_inside_blob_ptr = bbox_targets_inside_blob->mutable_cpu_data();
	float* bbox_targets_outside_blob_ptr = bbox_targets_outside_blob->mutable_cpu_data();

	for (int i = 0; i < labels_mat.rows; ++i)
		label_blob_ptr[i] = labels_mat.at<float>(i);

	int feature_area = bbox_targets_blob->height() * bbox_targets_blob->width();
	for (int i = 0; i < bbox_targets_blob->channel(); ++i) {
		for (int j = 0; j < feature_area; ++j) {
			bbox_targets_blob_ptr[j] = bbox_targets.at<float>(j, i);
			bbox_targets_inside_blob_ptr[j] = bbox_inside_weights.at<float>(j, i);
			bbox_targets_outside_blob_ptr[j] = bbox_outside_weights.at<float>(j, i);
		}
		bbox_targets_blob_ptr += feature_area;
		bbox_targets_inside_blob_ptr += feature_area;
		bbox_targets_outside_blob_ptr += feature_area;
		//int n = i * 4;
		//bbox_targets_blob_ptr[n] = bbox_targets.at<float>(i, 0);
		//bbox_targets_blob_ptr[n + 1] = bbox_targets.at<float>(i, 1);
		//bbox_targets_blob_ptr[n + 2] = bbox_targets.at<float>(i, 2);
		//bbox_targets_blob_ptr[n + 3] = bbox_targets.at<float>(i, 3);
		//bbox_targets_inside_blob_ptr[n] = bbox_inside_weights.at<float>(i, 0);
		//bbox_targets_inside_blob_ptr[n + 1] = bbox_inside_weights.at<float>(i, 1);
		//bbox_targets_inside_blob_ptr[n + 2] = bbox_inside_weights.at<float>(i, 2);
		//bbox_targets_inside_blob_ptr[n + 3] = bbox_inside_weights.at<float>(i, 3);
		//bbox_targets_outside_blob_ptr[n] = bbox_outside_weights.at<float>(i, 0);
		//bbox_targets_outside_blob_ptr[n + 1] = bbox_outside_weights.at<float>(i, 1);
		//bbox_targets_outside_blob_ptr[n + 2] = bbox_outside_weights.at<float>(i, 2);
		//bbox_targets_outside_blob_ptr[n + 3] = bbox_outside_weights.at<float>(i, 3);
	}
}

void AnchorTargetLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}

void AnchorTargetLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

}

Mat AnchorTargetLayer::unmap(Mat data, int total_anchors, vector<int> inds_inside, int fill) {
	Mat result(total_anchors, data.cols, CV_32F, Scalar(float(fill)));
	for (int i = 0; i < inds_inside.size(); ++i) {
		for (int j = 0; j < data.cols; ++j) {
			result.at<float>(inds_inside[i], j) = data.at<float>(i, j);
		}
	}
	return result;
}