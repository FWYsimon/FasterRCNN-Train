#include "ProposalLayer.h"

void ProposalLayer::setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	map<string, string> param = parseParamStr(param_str);

	int feat_stride = getParamInt(param, "feat_stride");

	this->_feat_stride = feat_stride;
	this->_anchors = generateAnchors(16, { 0.5, 1, 2 }, { 8, 16, 32 });
	this->_num_anchors = _anchors.size();

	Blob* rois = top[0];
	rois->reshape(1, 5);
	
	if (numTop > 1)
		top[1]->reshape(1, 1, 1, 1);

	this->phase = phase;
}

void ProposalLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	int pre_nms_topN;
	int post_nms_topN;
	float nms_thresh;
	float min_size;
	
	if (phase == PhaseTrain) {
		pre_nms_topN = cfg.TRAIN.RPN_PRE_NMS_TOP_N;
		post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N;
		nms_thresh = cfg.TRAIN.RPN_NMS_THRESH;
		min_size = cfg.TRAIN.RPN_MIN_SIZE;
	}
	else {
		pre_nms_topN = cfg.TEST.RPN_PRE_NMS_TOP_N;
		post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N;
		nms_thresh = cfg.TEST.RPN_NMS_THRESH;
		min_size = cfg.TEST.RPN_MIN_SIZE;
	}

	Blob* rpn_cls_prob_reshape = bottom[0];
	Blob* rpn_bbox_pred = bottom[1];
	Blob* im_info = bottom[2];

	int offset = _num_anchors * rpn_cls_prob_reshape->height() * rpn_cls_prob_reshape->width();
	float* rpn_cls_ptr = rpn_cls_prob_reshape->mutable_cpu_data() + offset;
	shared_ptr<Blob> rpn_cls_blob = newBlobByShape(rpn_cls_prob_reshape->num(), _num_anchors, rpn_cls_prob_reshape->height(), rpn_cls_prob_reshape->width());
	rpn_cls_blob->set_cpu_data(rpn_cls_ptr);
	shared_ptr<Blob> rpn_cls_blob_trans = rpn_cls_blob->transpose(0, 2, 3, 1);
	
	Mat rpn_cls = Mat(rpn_cls_blob_trans->count(), 1, CV_32F, rpn_cls_blob_trans->mutable_cpu_data());

	int im_height = im_info->mutable_cpu_data()[0];
	int im_width = im_info->mutable_cpu_data()[1];
	float im_scale = im_info->mutable_cpu_data()[2];

	int batch_size = rpn_cls_prob_reshape->num();
	int height = rpn_cls_prob_reshape->height();
	int width = rpn_cls_prob_reshape->width();

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

	//vector<float> test = vector<float>(rpn_bbox_pred->mutable_cpu_data(), rpn_bbox_pred->mutable_cpu_data() + rpn_bbox_pred->count());

	// bbox deltas will be(1, 4 * A, H, W) format
	// transpose to(1, H, W, 4 * A)
	// reshape to(1 * H * W * A, 4) where rows are ordered by(h, w, a)
	shared_ptr<Blob> bbox_deltas_blob = newBlobByShape(rpn_bbox_pred->num(), rpn_bbox_pred->channel(), rpn_bbox_pred->height(), rpn_bbox_pred->width());
	bbox_deltas_blob->copyFrom(rpn_bbox_pred);
	shared_ptr<Blob> bbox_deltas_blob_trans = bbox_deltas_blob->transpose(0, 2, 3, 1);
	bbox_deltas_blob_trans->reshape(bbox_deltas_blob_trans->count() / 4, 4);

	Mat bbox_deltas = Mat(bbox_deltas_blob_trans->num(), bbox_deltas_blob_trans->channel(), CV_32F, bbox_deltas_blob_trans->mutable_cpu_data());

	Mat proposals = bbox_transform_inv(anchors, bbox_deltas);

	proposals = clip_boxes(proposals, im_height, im_width);

	// filter boxes
	vector<int> keep_inds;
	vector<BBox> keep_boxes;
	for (int i = 0; i < proposals.rows; ++i) {
		float xmin = proposals.at<float>(i, 0);
		float ymin = proposals.at<float>(i, 1);
		float xmax = proposals.at<float>(i, 2);
		float ymax = proposals.at<float>(i, 3);
		float ws = xmax - xmin + 1;
		float hs = ymax - ymin + 1;
		if (ws >= min_size * im_scale && hs >= min_size * im_scale) {
			BBox bbox;
			bbox.xmin = xmin;
			bbox.ymin = ymin;
			bbox.xmax = xmax;
			bbox.ymax = ymax;
			bbox.score = rpn_cls.at<float>(i, 0);

			keep_inds.push_back(i);
			keep_boxes.push_back(bbox);
		}
	}

	std::sort(keep_boxes.begin(), keep_boxes.end(), [](BBox& a, BBox& b){
		return a.score > b.score;
	});

	if (pre_nms_topN > 0 && keep_boxes.size() > pre_nms_topN)
		keep_boxes.erase(keep_boxes.begin() + pre_nms_topN, keep_boxes.end());

	keep_boxes = nms(keep_boxes, nms_thresh, keep_inds, MIN);

	if (post_nms_topN > 0 && keep_boxes.size() > post_nms_topN)
		keep_boxes.erase(keep_boxes.begin() + post_nms_topN, keep_boxes.end());

	top[0]->reshape(keep_boxes.size(), 5);
	float* rois_ptr = top[0]->mutable_cpu_data();
	for (int i = 0; i < keep_boxes.size(); ++i) {
		rois_ptr[0] = 0;
		rois_ptr[1] = keep_boxes[i].xmin;
		rois_ptr[2] = keep_boxes[i].ymin;
		rois_ptr[3] = keep_boxes[i].xmax;
		rois_ptr[4] = keep_boxes[i].ymax;
		rois_ptr += 5;
	}

	if (numTop > 1) {
		top[1]->reshape(keep_boxes.size(), 1);
		float* scores_ptr = top[1]->mutable_cpu_data();
		for (int i = 0; i < keep_boxes.size(); ++i) {
			scores_ptr[0] = keep_boxes[i].score;
			scores_ptr += 1;
		}
	}

	bbox_deltas_blob.reset();
	bbox_deltas_blob_trans.reset();
	rpn_cls_blob.reset();
	rpn_cls_blob_trans.reset();
}

void ProposalLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}

void ProposalLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

}