#include "RoiDataLayer.h"

void RoIDataLayer::setup(const char* name, const char* type, const char* param_str,
	int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	map<string, string> param = parseParamStr(param_str);

	//int batch_size = getParamInt(param, "batch_size");
	//const int width = getParamInt(param, "width");
	//const int height = getParamInt(param, "height");
	const int numClass = getParamInt(param, "num_classes");

	this->_num_classes = numClass;
	Blob* image = top[0];
	Blob* raw_image = top[3];
	image->reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.TRAIN.SCALES, cfg.TRAIN.MAX_SIZE);
	raw_image->reshape(cfg.TRAIN.IMS_PER_BATCH, 3, cfg.TRAIN.SCALES, cfg.TRAIN.MAX_SIZE);
	if (cfg.TRAIN.HAS_RPN) {
		//rpn网络batch_size只能为1
		CV_Assert(cfg.TRAIN.IMS_PER_BATCH == 1);
		Blob* im_info = top[1];
		Blob* gt_boxes = top[2];
		im_info->reshape(1, 3);
		gt_boxes->reshape(1, 4);
	}
	else {
		Blob* rois = top[1];
		Blob* labels = top[2];
		rois->reshape(1, 5);
		labels->reshape(1);
		if (cfg.TRAIN.BBOX_REG) {
			Blob* bbox_target = top[3];
			Blob* bbox_inside_weights = top[4];
			Blob* bbox_outside_weights = top[5];
			bbox_target->reshape(1, numClass * 4);
			bbox_inside_weights->reshape(1, numClass * 4);
			bbox_outside_weights->reshape(1, numClass * 4);
		}
	}

	prepareData();
}

void RoIDataLayer::prepareData() {
	PaVfiles vfs;
	paFindFilesShort(config.datapath.c_str(), vfs, "*.jpg", false, true, PaFindFileType_File);

	//_vecClassName.insert(_vecClassName.begin(), classname, classname + 21);
	vector<string> vecImagePath;
	vector<XMLInfo> vecXmlInfo;
	for (int i = 0; i < vfs.size(); i++) {
		string xmlfile = config.xmlpath + "/" + vfs[i].substr(0, vfs[i].length() - 4) + ".xml";

		vector<XMLInfo> tmpXmlInfo = _readXmlClass.GetXmlInfo(xmlfile, "");

		string tmpImagePath = config.datapath + "/" + vfs[i];
		//_mapData.insert(make_pair(tmpImagePath, tmpXmlInfo));
		_vecImageName.push_back(tmpImagePath);

		vector<BBox> gt_boxes;
		for (int m = 0; m < tmpXmlInfo.size(); m++) {
			BBox tmp;
			tmp.xmin = tmpXmlInfo[m].xmin;
			tmp.ymin = tmpXmlInfo[m].ymin;
			tmp.xmax = tmpXmlInfo[m].xmax;
			tmp.ymax = tmpXmlInfo[m].ymax;
			gt_boxes.push_back(tmp);
		}

		roi oneImageData;
		vector<vector<float>> tmp_gt_overlaps;
		if (cfg.TRAIN.HAS_RPN) {
			vector<string>::iterator it;
			for (int m = 0; m < tmpXmlInfo.size(); m++) {
				vector<float> tmp_gt_overlap(_num_classes, 0.0f);

				int classId = g_labelmap[tmpXmlInfo[m].slabel];
				tmp_gt_overlap[classId] = 1.0;
				tmp_gt_overlaps.push_back(tmp_gt_overlap);
			}
		}
		else {
			_map_data.clear();

			vector<vector<float>> gt_overlaps_matrix = bbox_overlaps(gt_boxes, rpn_generate_bbox[tmpImagePath]);
			vector<int> maxes;
			vector<int> max_classes;
			for (int m = 0; m < gt_overlaps_matrix.size(); m++) {
				vector<float> tmp_gt_overlap(_num_classes, 0.0f);
				vector<float>::iterator biggest = max_element(begin(gt_overlaps_matrix[m]), end(gt_overlaps_matrix[m]));
					
				int num_gt_box = distance(begin(gt_overlaps_matrix[m]), biggest);

				int classId = g_labelmap[tmpXmlInfo[num_gt_box].slabel];
					
				tmp_gt_overlap[classId] = *biggest;
				tmp_gt_overlaps.push_back(tmp_gt_overlap);
				max_classes.push_back(classId);
				maxes.push_back(*biggest);
			}
			oneImageData.max_classes = max_classes;
			oneImageData.max_overlaps = maxes;
			oneImageData.boxes = rpn_generate_bbox[tmpImagePath];

			vector<BBox> rois = rpn_generate_bbox[tmpImagePath];
			if (cfg.TRAIN.BBOX_REG) {
				Mat bbox_targets(rois.size(), 5, CV_32F, Scalar(0));
				vector<int> gt_inds;
				vector<BBox> gt_inds_boxes;
				vector<int> ex_inds;
				Mat ex_rois;
				for (int m = 0; m < maxes.size(); m++) {
					if (maxes[m] == 1) {
						gt_inds.push_back(m);
						gt_inds_boxes.push_back(rois[m]);
					}
							
					if (maxes[m] >= cfg.TRAIN.BBOX_THRESH) {
						ex_inds.push_back(m);
						vector<float> vec_rois = { rois[m].xmin, rois[m].ymin, rois[m].xmax, rois[m].ymax };
						ex_rois.push_back(Mat(1, vec_rois.size(), CV_32F, vec_rois.data()));
					}
				}

				if (gt_inds.size() != 0) {
					vector<vector<float>> ex_gt_overlaps = bbox_overlaps(ex_rois, gt_inds_boxes);

					//vector<float> gt_assignment;
					Mat gt_rois;
					for (int m = 0; m < ex_gt_overlaps.size(); m++) {
						vector<float>::iterator biggest = max_element(begin(ex_gt_overlaps[m]), end(ex_gt_overlaps[m]));

						int gt_assignment = distance(ex_gt_overlaps[m].begin(), biggest);
						//gt_assignment.push_back(distance(begin(ex_gt_overlaps[m], biggest)));
						BBox tmp = rois[gt_inds[gt_assignment]];
						vector<float> vec_tmp = { tmp.xmin, tmp.ymin, tmp.xmax, tmp.ymax };

						gt_rois.push_back(Mat(1, vec_tmp.size(), CV_32F, vec_tmp.data()));
					}

					Mat bbox_transform_mat = bbox_transform(ex_rois, gt_rois);

					for (int m = 0; m < ex_inds.size(); m++) {
						bbox_targets.at<float>(ex_inds[m], 0) = max_classes[ex_inds[m]];
						bbox_targets.at<float>(ex_inds[m], 1) = bbox_transform_mat.at<float>(m, 0);
						bbox_targets.at<float>(ex_inds[m], 2) = bbox_transform_mat.at<float>(m, 1);
						bbox_targets.at<float>(ex_inds[m], 3) = bbox_transform_mat.at<float>(m, 2);
						bbox_targets.at<float>(ex_inds[m], 4) = bbox_transform_mat.at<float>(m, 3);
					}
				}
					
				Mat means(_num_classes, 4, CV_32F);
				Mat stds(_num_classes, 4, CV_32F);
				if (cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED) {
					// Use fixed / precomputed "means" and "stds" instead of empirical values
					for (int m = 0; m < means.rows; m++) {
						means.at<float>(m, 0) = cfg.TRAIN.BBOX_NORMALIZE_MEANS[0];
						means.at<float>(m, 1) = cfg.TRAIN.BBOX_NORMALIZE_MEANS[1];
						means.at<float>(m, 2) = cfg.TRAIN.BBOX_NORMALIZE_MEANS[2];
						means.at<float>(m, 3) = cfg.TRAIN.BBOX_NORMALIZE_MEANS[3];
						stds.at<float>(m, 0) = cfg.TRAIN.BBOX_NORMALIZE_STDS[0];
						stds.at<float>(m, 1) = cfg.TRAIN.BBOX_NORMALIZE_STDS[1];
						stds.at<float>(m, 2) = cfg.TRAIN.BBOX_NORMALIZE_STDS[2];
						stds.at<float>(m, 3) = cfg.TRAIN.BBOX_NORMALIZE_STDS[3];
					}
				}
				else {
					// Compute values needed for means and stds
					// var(x) = E(x ^ 2) - E(x) ^ 2
					Mat class_counts(_num_classes, 1, CV_32F, Scalar(0));
					class_counts += Scalar(cfg.EPS);
					Mat sums(_num_classes, 4, CV_32F, Scalar(0));
					Mat squared_sums(_num_classes, 4, CV_32F, Scalar(0));

						
					for (int m = 0; m < _num_classes; m++) {
						vector<int> cls_inds;
						Mat cls_vals;
						for (int n = 0; n < bbox_targets.rows; n++) {
							if (bbox_targets.at<int>(n, 0) == m) {
								cls_inds.push_back(m);
								Mat tmp(1, 4, CV_32F);
								tmp.at<float>(0, 0) = bbox_targets.at<float>(n, 1);
								tmp.at<float>(0, 1) = bbox_targets.at<float>(n, 2);
								tmp.at<float>(0, 2) = bbox_targets.at<float>(n, 3);
								tmp.at<float>(0, 3) = bbox_targets.at<float>(n, 4);
								cls_vals.push_back(tmp.clone());
							}
						}
						if (cls_inds.size() > 0) {
							class_counts.at<float>(m, 0) += cls_inds.size();
							vector<float> tmp_sums;
							vector<float> tmp_pow_sums;
							for (int n = 0; n < cls_vals.cols; n++) {
								float tmp = 0;
								float pow_tmp = 0;
								for (int x = 0; x < cls_vals.rows; x++) {
									tmp += cls_vals.at<float>(x, n);
									pow_tmp += pow(cls_vals.at<float>(x, n), 2);
								}
								tmp_sums.push_back(tmp);
								tmp_pow_sums.push_back(pow_tmp);
							}
							for (int n = 0; n < sums.cols; n++) {
								sums.at<float>(m, n) = tmp_sums[n];
								squared_sums.at<float>(m, n) = tmp_pow_sums[n];
							}
						}

						for (int n = 0; n < means.cols; n++) {
							means.at<float>(m, n) = sums.at<float>(m, n) / class_counts.at<float>(m, n);
							stds.at<float>(m, n) = pow((squared_sums.at<float>(m, n) / class_counts.at<float>(m, n) - pow(means.at<float>(m, n), 2)), 0.5);
						}

					}

				}
				// Normalize targets
				if (cfg.TRAIN.BBOX_NORMALIZE_TARGETS) {
					for (int m = 0; m < _num_classes; m++) {
						for (int n = 0; n < bbox_targets.rows; n++) {
							if (bbox_targets.at<int>(n, 0) == m) {
								//cls_inds.push_back(m);
								bbox_targets.at<float>(n, 1) -= means.at<float>(m, 0);
								bbox_targets.at<float>(n, 2) -= means.at<float>(m, 1);
								bbox_targets.at<float>(n, 3) -= means.at<float>(m, 2);
								bbox_targets.at<float>(n, 4) -= means.at<float>(m, 3);
								bbox_targets.at<float>(n, 1) /= stds.at<float>(m, 0);
								bbox_targets.at<float>(n, 2) /= stds.at<float>(m, 1);
								bbox_targets.at<float>(n, 3) /= stds.at<float>(m, 2);
								bbox_targets.at<float>(n, 4) /= stds.at<float>(m, 3);
							}
						}
							
					}
				}

				oneImageData.bbox_targets = bbox_targets;
			}

		}
			
		oneImageData.gt_overlaps = tmp_gt_overlaps;
		oneImageData.xmlInfo = tmpXmlInfo;
		_map_data.insert(make_pair(tmpImagePath, oneImageData));
	}
	random_shuffle(_vecImageName.begin(), _vecImageName.end());
	
}

void RoIDataLayer::loadBatch(Blob** top, int numTop) {
	Blob* image = top[0];
	Blob* raw_image = top[3];
	int max_height = 0;
	int max_width = 0;

	int batch_size = image->num();

	vector<Mat> input_images;
	vector<Mat> raw_images;
	for (int i = 0; i < batch_size; i++) {
		string imageName = _vecImageName[_cursor];
		Mat im = imread(imageName);

		//随机翻转
		int aug = randr(0, 1);
		if (aug)
			augmentation_flip(im, _map_data[imageName].xmlInfo);

		//for (int i = 0; i < _map_data[imageName].xmlInfo.size(); ++i) {
		//	XMLInfo info = _map_data[imageName].xmlInfo[i];
		//	rectangle(im, Rect(info.xmin, info.ymin, info.xmax - info.xmin, info.ymax - info.ymin), Scalar(255, 255, 0), 1);
		//}

		float im_size_min = min(im.rows, im.cols);
		float im_size_max = max(im.rows, im.cols);
		float im_scale = cfg.TRAIN.MIN_SIZE / im_size_min;

		if (round(im_scale * im_size_max) > cfg.TRAIN.MAX_SIZE)
			im_scale = cfg.TRAIN.MAX_SIZE / im_size_max;

		resize(im, im, Size(), im_scale, im_scale, CV_INTER_LINEAR);

		max_height = max(max_height, im.rows);
		max_width = max(max_width, im.cols);

		im.convertTo(im, CV_32F);
		Mat raw_im;
		im.copyTo(raw_im);
		raw_images.push_back(raw_im);

		im -= cfg.PIXEL_MEANS;
		
		input_images.push_back(im);

		if (cfg.TRAIN.HAS_RPN) {
			Blob* im_info = top[1];
			Blob* gt_boxes = top[2];
			float* im_info_ptr = im_info->mutable_cpu_data();
			gt_boxes->reshape(_map_data[imageName].xmlInfo.size(), 5, 1, 1);
			Mat gt_boxes_mat(_map_data[imageName].xmlInfo.size(), 5, CV_32F, top[2]->mutable_cpu_data());
			for (int j = 0; j < _map_data[imageName].xmlInfo.size(); j++) {
				float* gt_boxes_ptr = gt_boxes_mat.ptr<float>(j);
				gt_boxes_ptr[0] = _map_data[imageName].xmlInfo[j].xmin * im_scale;
				gt_boxes_ptr[1] = _map_data[imageName].xmlInfo[j].ymin * im_scale;
				gt_boxes_ptr[2] = _map_data[imageName].xmlInfo[j].xmax * im_scale;
				gt_boxes_ptr[3] = _map_data[imageName].xmlInfo[j].ymax * im_scale;
				gt_boxes_ptr[4] = g_labelmap[_map_data[imageName].xmlInfo[j].slabel];
				
				//gt_boxes_ptr += 5;
			}

			im_info_ptr[0] = im.rows;
			im_info_ptr[1] = im.cols;
			im_info_ptr[2] = im_scale;
			//im_info_ptr += im_info->count();
		}
		else {
			Blob* rois_blob = top[1];
			Blob* labels_blob = top[2];
			Blob* bbox_targets_blob = top[3];
			Blob* bbox_inside_weights_blob = top[4];
			Blob* bbox_outside_weights_blob = top[5];
			int rois_per_image = cfg.TRAIN.BATCH_SIZE / batch_size;
			int fg_rois_per_image = cfg.TRAIN.FG_FRACTION * rois_per_image;

			vector<vector<float>> gt_overlaps = _map_data[imageName].gt_overlaps;
			vector<int> labels = _map_data[imageName].max_classes;
			vector<BBox> rois = _map_data[imageName].boxes;
			vector<int> max_overlaps = _map_data[imageName].max_overlaps;
			Mat bbox_targets_data = _map_data[imageName].bbox_targets;

			//vector<int>
			//选取前景的id
			vector<int> fg_inds;
			vector<int> bg_inds;
			for (int j = 0; j < gt_overlaps.size(); j++) {
				vector<float>::iterator biggest = max_element(gt_overlaps[j].begin(), gt_overlaps[j].end());
				//max_overlaps.push_back(*biggest);
				if (*biggest >= cfg.TRAIN.FG_THRESH)
					fg_inds.push_back(j);
				else if (*biggest >= cfg.TRAIN.BG_THRESH_LO && *biggest < cfg.TRAIN.BG_THRESH_HI)
					bg_inds.push_back(j);

			}
			int fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size());

			if (fg_inds.size() > 0)
				random_shuffle(fg_inds.begin(), fg_inds.end());

			int bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image;
			bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size());

			if (bg_inds.size() > 0)
				random_shuffle(bg_inds.begin(), bg_inds.end());

			vector<int> keep_inds(fg_rois_per_this_image + bg_rois_per_this_image);
			copy(fg_inds.begin(), fg_inds.begin() + fg_rois_per_this_image, keep_inds.begin());
			copy(bg_inds.begin(), bg_inds.begin() + bg_rois_per_this_image, keep_inds.begin() + fg_rois_per_this_image);

			vector<vector<float>> gt_overlaps_keeps;
			vector<int> labels_keeps;
			vector<BBox> rois_keeps;

			for (int j = 0; j < keep_inds.size(); j++) {
				labels_keeps.push_back(labels[keep_inds[j]]);
				gt_overlaps_keeps.push_back(gt_overlaps[keep_inds[j]]);
				rois_keeps.push_back(rois[keep_inds[j]]);
			}
			
			for (int j = 0; j < bg_rois_per_this_image; j++)
				labels[bg_inds[j]] = 0;

			rois_blob->reshape(1, gt_overlaps_keeps.size(), 5, 1);
			Mat rois_mat(gt_overlaps_keeps.size(), 5, CV_32F, top[1]->mutable_cpu_data());
			for (int j = 0; j < rois_keeps.size(); j++) {
				float* rois_ptr = rois_mat.ptr<float>(j);
				rois_keeps[j].xmax *= im_scale;
				rois_keeps[j].ymax *= im_scale;
				rois_keeps[j].xmin *= im_scale;
				rois_keeps[j].ymin *= im_scale;
				rois_ptr[0] = i;
				rois_ptr[1] = rois_keeps[j].xmin;
				rois_ptr[2] = rois_keeps[j].ymin;
				rois_ptr[3] = rois_keeps[j].xmax;
				rois_ptr[4] = rois_keeps[j].ymax;
			}

			labels_blob->reshape(1, gt_overlaps_keeps.size(), 1, 1);
			Mat labels_mat(gt_overlaps_keeps.size(), 1, CV_32F, top[2]->mutable_cpu_data());
			for (int j = 0; j < labels_keeps.size(); j++) {
				float* label_ptr = labels_mat.ptr<float>(j);
				label_ptr[0] = labels_keeps[j];
			}

			if (cfg.TRAIN.BBOX_REG) {
				vector<float> clss;
				vector<float> inds;
				for (int j = 0; j < keep_inds.size(); j++) {
					clss.push_back(bbox_targets_data.at<float>(j, 0));
					if (bbox_targets_data.at<float>(j, 0) > 0)
						inds.push_back(bbox_targets_data.at<float>(j, 0));
				}
					

				Mat bbox_targets(clss.size(), 4 * _num_classes, CV_32F, Scalar(0));
				Mat bbox_inside_weights(clss.size(), 4 * _num_classes, CV_32F, Scalar(0));
				Mat bbox_outside_weights(clss.size(), 4 * _num_classes, CV_32F, Scalar(0));
				for (int j = 0; j < inds.size(); j++) {
					int ind = inds[j];
					int cls = clss[ind];
					int start = 4 * cls;
					bbox_targets.at<float>(ind, start) = bbox_targets_data.at<float>(ind, 1);
					bbox_targets.at<float>(ind, start + 1) = bbox_targets_data.at<float>(ind, 2);
					bbox_targets.at<float>(ind, start + 2) = bbox_targets_data.at<float>(ind, 3);
					bbox_targets.at<float>(ind, start + 3) = bbox_targets_data.at<float>(ind, 4);
					bbox_inside_weights.at<float>(ind, start) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[0];
					bbox_inside_weights.at<float>(ind, start + 1) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[1];
					bbox_inside_weights.at<float>(ind, start + 2) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[2];
					bbox_inside_weights.at<float>(ind, start + 3) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[3];
					bbox_outside_weights.at<float>(ind, start) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[0];
					bbox_outside_weights.at<float>(ind, start + 1) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[1];
					bbox_outside_weights.at<float>(ind, start + 2) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[2];
					bbox_outside_weights.at<float>(ind, start + 3) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[3];
				}

				bbox_targets_blob->reshape(1, gt_overlaps_keeps.size(), 4 * _num_classes, 1);
				Mat bbox_targets_mat(gt_overlaps_keeps.size(), 4, CV_32F, top[3]->mutable_cpu_data());
				Mat bbox_inside_weights_mat(gt_overlaps_keeps.size(), 4, CV_32F, top[4]->mutable_cpu_data());
				Mat bbox_outside_weights_mat(gt_overlaps_keeps.size(), 4, CV_32F, top[5]->mutable_cpu_data());
				for (int j = 0; j < keep_inds.size(); j++) {
					float* bbox_targets_ptr = bbox_targets_mat.ptr<float>(j);
					float* bbox_inside_weights_ptr = bbox_inside_weights_mat.ptr<float>(j);
					float* bbox_outside_weights_ptr = bbox_outside_weights_mat.ptr<float>(j);
					for (int m = 0; m < _num_classes * 4; m++) {
						bbox_targets_ptr[m] = bbox_targets.at<float>(j, m);
						bbox_inside_weights_ptr[m] = bbox_inside_weights.at<float>(j, m);
						bbox_outside_weights_ptr[m] = bbox_outside_weights.at<float>(j, m);
					}
				}
			}
			
		}

		this->_cursor++;
		if (this->_cursor == this->_vecImageName.size()) {
			this->_cursor = 0;
			std::random_shuffle(_vecImageName.begin(), _vecImageName.end());
		}

	}

	image->reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max_height, max_width);
	raw_image->reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max_height, max_width);
	float* raw_image_ptr = raw_image->mutable_cpu_data();
	float* image_ptr = image->mutable_cpu_data();
	caffe_set(image->count(), 0.0f, image_ptr);
	caffe_set(raw_image->count(), 0.0f, raw_image_ptr);
	for (int i = 0; i < input_images.size(); ++i) {
		Mat im = input_images[i];
		Mat raw_im = raw_images[i];
		vector<Mat> ms;
		split(im, ms);
		vector<Mat> raw_ms;
		split(raw_im, raw_ms);
		// 假如是比最大尺寸小的图就放在左上角，其他地方置0
		for (int j = 0; j < ms.size(); ++j) {
			for (int h = 0; h < ms[j].rows; ++h) {
				for (int w = 0; w < ms[j].cols; ++w) {
					int index = h * image->width() + w;
					image_ptr[index] = ms[j].at<float>(h, w);
					raw_image_ptr[index] = raw_ms[j].at<float>(h, w);
				}
			}
			image_ptr += image->width() * image->height();
			raw_image_ptr += image->width() * image->height();
		}
	}

}


void RoIDataLayer::forward(Blob** bottom, int numBottom, Blob** top, int numTop) {
	loadBatch(top, numTop);
}

void RoIDataLayer::backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down) {

}

void RoIDataLayer::reshape(Blob** bottom, int numBottom, Blob** top, int numTop) {

}