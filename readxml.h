#pragma once
#include "common.h"


struct XMLInfo
{
	float xmin, ymin, xmax, ymax;
	string slabel;
	vector<Point2f> GetBoxPoint(){

		vector<Point2f> vecPointOut;
		Point2f tmpPoint;
		//左上
		tmpPoint.x = xmin;
		tmpPoint.y = ymin;
		vecPointOut.push_back(tmpPoint);

		//右上
		tmpPoint.x = xmax;
		tmpPoint.y = ymin;
		vecPointOut.push_back(tmpPoint);

		//左下
		tmpPoint.x = xmin;
		tmpPoint.y = ymax;
		vecPointOut.push_back(tmpPoint);

		//右下
		tmpPoint.x = xmax;
		tmpPoint.y = ymax;
		vecPointOut.push_back(tmpPoint);
		return vecPointOut;
	}
};

class readxml
{
public:
	readxml();
	~readxml();

	vector<XMLInfo> readxml::GetXmlInfo(string XMLFile, string objectName);
	bool ModifyNode_Text(std::string& XmlFile, std::string& strNodeName, const std::string& strText);
private:
	bool FiltObject(vector<TiXmlElement*> &NodeVector, string objectName);
	bool GetAllNodePointerByName(TiXmlElement* pRootEle, string strNodeName, vector<TiXmlElement*> &NodeVector);
	bool GetNodePointerByName(TiXmlElement* pRootEle, std::string &strNodeName, vector<TiXmlElement*> &NodeVector);
	void GetBoundingBox(vector<TiXmlElement*> NodeVector);
private:
	string savepath_,imagename_;
	vector<XMLInfo> vecXmlInfo_;
	string className_;
};

