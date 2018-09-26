#include "AnchorTargetLayer.h"

vector<BBox> AnchorTargetLayer::generateAnchors(int base_size = 16, const vector<float>& ratios = { 0.5, 1, 2 }, const vector<float>& scales = { pow(2.0f, 3.0f), pow(2.0f, 4.0f), pow(2.0f, 5.0f) }){

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

Mat AnchorTargetLayer::makeShifts(int feat_stride, int width, int height){
	Mat m = Mat::zeros(width * height, 4, CV_32F);
	for (int i = 0; i < m.rows; ++i){
		m.at<float>(i, 0) = (i % width) * feat_stride;
		m.at<float>(i, 1) = (i / width) * feat_stride;
	}
	m.col(0).copyTo(m.col(2));
	m.col(1).copyTo(m.col(3));
	return m;
}

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
	
	rpn_labels->Reshape(1, 1, _num_anchors * height, width);
	rpn_bbox_targets->Reshape(1, _num_anchors * 4, height, width);
	rpn_bbox_inside_weights->ReshapeLike(rpn_bbox_targets);
	rpn_bbox_outside_weights->ReshapeLike(rpn_bbox_targets);
}

void AnchorTargetLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	Blob* data = bottom[0];
	Blob* gt_boxes = bottom[1];
	Blob* im_info = bottom[2];

	int height = data->height();
	int width = data->width();

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
			BBox tmp;
			tmp.xmin = anchors.at<float>(i, 0);
			tmp.ymin = anchors.at<float>(i, 1);
			tmp.xmax = anchors.at<float>(i, 2);
			tmp.ymax = anchors.at<float>(i, 3);
			
			

			keep_anchors.push_back(tmp);
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
		BBox tmp;
		tmp.xmin = gt_boxes_mat.at<float>(i, 0);
		tmp.ymin = gt_boxes_mat.at<float>(i, 1);
		tmp.xmax = gt_boxes_mat.at<float>(i, 2);
		tmp.ymax = gt_boxes_mat.at<float>(i, 3);
		vec_gt_boxes.push_back(tmp);
	}
	vector<vector<float>> overlaps = bbox_overlaps(keep_anchors, vec_gt_boxes);

	vector<int> argmax_overlaps;
	vector<float> max_overlaps;
	for (int i = 0; i < overlaps.size(); i++) {
		vector<float> biggest = max_element(begin(overlaps[i]), end(overlaps[i]));
		argmax_overlaps.push_back(distance(begin(overlaps[i]), biggest));
		max_overlaps.push_back(*biggest);
	}
	
	vector<int> gt_argmax_overlaps;
	vector<float> gt_max_overlaps;
	for (int i = 0; i < overlaps[0].size(); i++) {
		int max_num = -1;
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

	if (!cfg.TRAIN.RPN_CLOBBER_POSITIVES) {
		for (int i = 0; i < max_overlaps.size(); i++)
			if (max_overlaps[i] < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
				labels[i] = 0;
	}

	for (int i = 0; i < gt_argmax_overlaps.size(); i++)
		labels[gt_argmax_overlaps] = 1;

	for (int i = 0; i < max_overlaps.size(); i++)
		if (max_overlaps[i] >= cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
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
		int num = fg_inds.size() - num_fg
		random_shuffle(fg_inds.begin(), fg_inds.end());
		for (int i = fg_inds.size(); i >= num_fg; i--)
			labels[fg_inds[i]] = -1;
		fg_inds.erase(fg_inds.end() - num, fg_inds.end());
	}

	int num_bg = cfg.TRAIN.RPN_BATCHSIZE - (num_fg < fg_inds.size() : num_fg, fg_inds.size());
	vector<int> bg_inds;
	for (int i = 0; i < labels.size(); i++)
		if (labels[i] == 0)
			bg_inds.push_back(i);

	if (bg_inds.size() > num_bg) {
		random_shuffle(bg_inds.begin(), bg_inds.end());
		for (int i = 0; i < bg_inds.size() - num_bg; i++)
			labels[bg_inds[i]] = -1;
	}

	//vector<vector<float>> bbox_targets(keep_inds.size(), vector<float>(4, 0));
	Mat gt_boxes_mat_tmp(argmax_overlaps.size(), 4, CV_32F);
	for (int i = 0; i < argmax_overlaps.size(); i++)
		gt_boxes_mat_tmp.at<float>(i) = gt_boxes_mat.at<float>(argmax_overlaps[i]);

	Mat bbox_targets = bbox_transform(keep_anchors_mat, gt_boxes_mat_tmp);

	Mat bbox_inside_weights(keep_anchors.size(), 4, CV_32F, Scalar(0));
	int num_examples = 0;
	int num_positive = 0;
	for (int i = 0; i < bbox_inside_weights.rows; i++) {
		if (labels[i] == 1) {
			bbox_inside_weights.at<float>(i, 0) = 1.0;
			bbox_inside_weights.at<float>(i, 1) = 1.0;
			bbox_inside_weights.at<float>(i, 2) = 1.0;
			bbox_inside_weights.at<float>(i, 3) = 1.0;
			num_positive++;
		}
		if (labels[i] >= 0)
			num_examples++;
	}

	Mat bbox_outside_weights(keep_anchors.size(), 4, CV_32F, Scalar(0));
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
			bbox_outside_weights.at<float>(i) = positive_weights;
		else if (labels[i] == 0)
			bbox_outside_weights.at<float>(i) = negative_weights;
	}

	Mat labels_mat(keep_inds.size(), 1, CV_32F);
	for (int i = 0; i < labels_mat.rows; i++)
		labels_mat.at<float>(labels[i]);

	Mat labels_mat = unmap(labels_mat, total_anchors, keep_inds, -1);
	Mat bbox_targets = unmap(bbox_targets, total_anchors, keep_inds, 0);
	Mat bbox_inside_weights = unmap(bbox_inside_weights, total_anchors, keep_inds, 0);
	Mat bbox_outside_weights = unmap(bbox_outside_weights, total_anchors, keep_inds, 0);

	Blob* label_blob = top[0];
	Blob* bbox_targets_blob = top[1];
	Blob* bbox_targets_inside_blob = top[2];
	Blob* bbox_targets_outside_blob = top[3];
	label_blob->Reshape(1, 1, A * height, width);
	bbox_targets_blob->Reshape(1, A * 4, height, width);
	bbox_targets_inside_blob->ReshapeLike(bbox_targets);
	bbox_targets_outside_blob->ReshapeLike(bbox_targets);

	float* label_blob_ptr = label_blob->mutable_cpu_data();
	float* bbox_targets_blob_ptr = bbox_targets_blob->mutable_cpu_data();
	float* bbox_targets_inside_blob_ptr = bbox_targets_inside_blob->mutable_cpu_data();
	float* bbox_targets_outside_blob_ptr = bbox_targets_outside_blob->mutable_cpu_data();

	for (int i = 0; i < labels_mat.rows; i++)
		label_blob_ptr[i] = labels_mat.at<float>(i);

	for (int i = 0; i < label_blob->count(); i++) {
		int n = i * 4;
		bbox_targets_blob_ptr[n] = bbox_targets.at<float>(i, 0);
		bbox_targets_blob_ptr[n + 1] = bbox_targets.at<float>(i, 1);
		bbox_targets_blob_ptr[n + 2] = bbox_targets.at<float>(i, 2);
		bbox_targets_blob_ptr[n + 3] = bbox_targets.at<float>(i, 3);
		bbox_targets_inside_blob_ptr[n] = bbox_inside_weights.at<float>(i, 0);
		bbox_targets_inside_blob_ptr[n + 1] = bbox_inside_weights.at<float>(i, 1);
		bbox_targets_inside_blob_ptr[n + 2] = bbox_inside_weights.at<float>(i, 2);
		bbox_targets_inside_blob_ptr[n + 3] = bbox_inside_weights.at<float>(i, 3);
		bbox_targets_outside_blob_ptr[n] = bbox_outside_weights.at<float>(i, 0);
		bbox_targets_outside_blob_ptr[n + 1] = bbox_outside_weights.at<float>(i, 1);
		bbox_targets_outside_blob_ptr[n + 2] = bbox_outside_weights.at<float>(i, 2);
		bbox_targets_outside_blob_ptr[n + 3] = bbox_outside_weights.at<float>(i, 3);
	}
}

void AnchorTargetLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}

void AnchorTargetLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

}

Mat AnchorTargetLayer::unmap(Mat data, int total_anchors, vector<int> inds_inside, int fill) {
	Mat result(total_anchors, data.cols, CV_32F, Scalar(float(fill)));
	for (int i = 0; i < inds_inside.size(); i++) {
		result.at<float>(inds_inside[i]) = data.at<float>(i);
	}
	return result;
}