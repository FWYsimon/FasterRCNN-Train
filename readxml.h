#pragma once
#include "common.h"




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

