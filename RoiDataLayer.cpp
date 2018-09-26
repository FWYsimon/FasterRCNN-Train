#include "RoiDataLayer.h"

#pragma comment(lib, "libcaffe.lib")

RoiDataLayer::~RoiDataLayer() {
	stopBatchLoader();
}

void RoiDataLayer::setup(const char* name, const char* type, const char* param_str, 
	int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {
	map<string, string> param = parseParamStr(param_str);

	int is_RPN = getParamInt(param, "is_PRN");
	int is_BBox_Reg = getParamInt(param, "is_BBox_Reg");
	int batch_size = getParamInt(param, "batch_size");
	const int width = getParamInt(param, "width");
	const int height = getParamInt(param, "height");
	const int numClass = getParamInt(param, "num_classes");


	this->_num_classes = numClass;
	this->_is_RPN = is_RPN;
	this->_is_BBox_Reg = is_BBox_Reg;
	Blob* image = top[0];
	image->Reshape(batch_size, 3, height, width);
	if (is_RPN) {
		//rpn网络batch_size只能为1
		CV_Assert(batch_size == 1);
		Blob* im_info = top[1];
		Blob* gt_boxes = top[2];
		im_info->Reshape(1, 3);
		gt_boxes->Reshape(1, 4);
	}
	else {
		Blob* rois = top[1];
		Blob* labels = top[2];
		rois->Reshape(1, 5);
		labels->Reshape(1);
		if (is_BBox_Reg) {
			Blob* bbox_target = top[3];
			Blob* bbox_inside_weights = top[4];
			Blob* bbox_outside_weights = top[5];
			bbox_target->Reshape(1, numClass * 4);
			bbox_inside_weights->Reshape(1, numClass * 4);
			bbox_outside_weights->Reshape(1, numClass * 4);
		}
		
	}

	prepareData();
	__super::setup(name, type, param_str, phase, bottom, numBottom, top, numTop);
}

void RoiDataLayer::prepareData() {
	PaVfiles vfsFloder;
	paFindFilesShort(config.datapath.c_str(), vfsFloder, "*", false, true, PaFindFileType_Directory);

	//_vecClassName.insert(_vecClassName.begin(), classname, classname + 21);
	vector<string> vecImagePath;
	vector<XMLInfo> vecXmlInfo;
	for (int i = 0; i < vfsFloder.size(); i++) {
		string label = vfsFloder[i];

		PaVfiles vfs;
		paFindFiles((config.datapath + "/" + vfsFloder[i]).c_str(), vfs, "*.xml");
		for (int j = 0; j < vfs.size(); j++) {
			vector<XMLInfo> tmpXmlInfo = _readXmlClass.GetXmlInfo(vfs[j], "");

			string tmpImagePath = vfs[j].substr(0, vfs[j].length() - 4) + ".jpg";
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
			if (_is_RPN) {
				vector<string>::iterator it;
				vector<vector<float>> tmp_gt_overlaps;
				for (int m = 0; m < tmpXmlInfo.size(); m++) {
					vector<float> tmp_gt_overlap(_num_classes, 0.0f);


					int classId = g_labelmap[tmpXmlInfo[m].slabel];
					tmp_gt_overlap[classId] = 1.0;
					tmp_gt_overlaps.push_back(tmp_gt_overlap);
				}
			}
			else {
				vector<vector<float>> gt_overlaps_matrix = bbox_overlaps(gt_boxes, rpn_generate_bbox[tmpImagePath]);
				vector<int> maxes;
				vector<int> max_classes;
				for (int m = 0; m < gt_overlaps_matrix.size(); m++) {
					vector<float> tmp_gt_overlap(_num_classes, 0.0f);
					vector<float>::iterator biggest = max_element(begin(gt_overlaps_matrix[m]), end(gt_overlaps_matrix[m]));
					
					int num_gt_box = distance(begin(gt_overlaps_matrix[m], biggest));

					int classId = g_labelmap[tmpXmlInfo[num_gt_box].slabel];
					
					tmp_gt_overlap[classId] = *biggest;
					tmp_gt_overlaps.push_back(tmp_gt_overlaps);
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
					vector<BBox> ex_rois;
					for (int m = 0; m < maxes.size(); m++) {
						if (maxes[m] == 1) {
							gt_inds.push_back(m);
							gt_inds_boxes.push_back(maxes[m]);
						}
							
						if (maxes[m] >= cfg.TRAIN.BBOX_THRESH) {
							ex_inds.push_back(m);
							ex_rois.push_back(maxes[m]);
						}
					}

					if (gt_inds.size() != 0) {
						vector<vector<float>> ex_gt_overlaps = bbox_overlaps(ex_rois, gt_inds_boxes);

						//vector<float> gt_assignment;
						vector<BBox> gt_rois;
						for (int m = 0; m < ex_gt_overlaps.size(); m++) {
							vector<float>::iterator biggest = max_element(begin(ex_gt_overlaps[m]), end(ex_gt_overlaps[m]));

							int gt_assignment = distance(begin(ex_gt_overlaps[m], biggest));
							//gt_assignment.push_back(distance(begin(ex_gt_overlaps[m], biggest)));
							gt_rois.push_back(rois[gt_inds[gt_assignment]]);
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
								class_counts.at<float>(m, 0) += Scalar(cls_inds.size());
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
			_map_data.insert(tmpImagePath, oneImageData);
		}

	}
	random_shuffle(_vecImageName.begin(), _vecImageName.end());
	
}

void RoiDataLayer::loadBatch(Blob** top, int numTop) {
	Blob* image = top[0];


	//
	//float* gt_boxes_ptr = gt_boxes->mutable_cpu_data();

	int batch_size = image->num();
	for (int i = 0; i < batch_size; i++) {
		string imageName = _vecImageName[_cursor];
		Mat im = imread(imageName);

		float im_size_min = min(im.rows, im.cols);
		float im_size_max = max(im.rows, im.cols);
		float im_scale = cfg.TRAIN.MIN_SIZE / im_size_min;

		if (cvRound(im_scale * im_size_max) > cfg.TRAIN.MAX_SIZE)
			im_scale = cfg.TRAIN.MAX_SIZE / im_size_max;

		resize(im, im, Size(), im_scale, im_scale, CV_INTER_LINEAR);

		im.convertTo(im, CV_32F);
		im -= means;
		image->setDataRGB(i, im);

		if (_is_RPN) {
			Blob* im_info = top[1];
			Blob* gt_boxes = top[2];
			float* im_info_ptr = im_info->mutable_cpu_data();
			gt_boxes->Reshape(1, 5, _map_data[imageName].xmlInfo.size(), 1);
			Mat blob(_map_data[imageName].xmlInfo.size(), 5, CV_32F, top[2]->mutable_cpu_data());
			for (int j = 0; j < _map_data[imageName].xmlInfo.size(); j++) {
				float* gt_boxes_ptr = blob.ptr<float>(j);
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
			this->_cursor++;
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
			vector<float> max_overlaps = _map_data[imageName].max_overlaps;
			Mat bbox_targets_data = _map_data[imageName].bbox_targets;

			//vector<int>
			//选取前景的id
			vector<int> fg_inds;
			vector<int> bg_inds;
			for (int j = 0; j < gt_overlaps.size(); j++) {
				vector<float>::iterator biggest = max_element(begin(gt_overlaps), end(gt_overlaps));
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
				labels_keeps.push_back(labels.begin() + keep_inds[j]);
				gt_overlaps_keeps.push_back(gt_overlaps.begin() + keep_inds[j]);
				rois_keeps.push_back(rois.begin() + keep_inds[j]);
			}
			
			for (int j = 0; j < bg_rois_per_this_image; j++)
				labels[bg_inds[j]] = 0;

			rois_blob->Reshape(1, gt_overlaps_keeps.size(), 5, 1);
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

			labels_blob->Reshape(1, gt_overlaps_keeps.size(), 1, 1);
			Mat labels_mat(gt_overlaps_keeps.size(), 1, CV_32F, top[2]->mutable_cpu_data());
			for (int j = 0; j < labels_keeps.size(); j++) {
				float* label_ptr = labels_mat.ptr<int>(j);
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
					bbox_targets.at<float>(ind, start) = bbox_targets_data[ind, 1];
					bbox_targets.at<float>(ind, start + 1) = bbox_targets_data[ind, 2];
					bbox_targets.at<float>(ind, start + 2) = bbox_targets_data[ind, 3];
					bbox_targets.at<float>(ind, start + 3) = bbox_targets_data[ind, 4];
					bbox_inside_weights.at<float>(ind, start) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[0];
					bbox_inside_weights.at<float>(ind, start + 1) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[1];
					bbox_inside_weights.at<float>(ind, start + 2) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[2];
					bbox_inside_weights.at<float>(ind, start + 3) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[3];
					bbox_outside_weights.at<float>(ind, start) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[0];
					bbox_outside_weights.at<float>(ind, start + 1) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[1];
					bbox_outside_weights.at<float>(ind, start + 2) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[2];
					bbox_outside_weights.at<float>(ind, start + 3) = cfg.TRAIN.BBOX_INSIDE_WEIGHTS[3];
				}

				bbox_targets_blob->Reshape(1, gt_overlaps_keeps.size(), 4 * _num_classes, 1);
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

		if (this->_cursor == this->_vecImageName.size())
		{
			this->_cursor = 0;
		}

	}
}


