#include "readxml.h"


readxml::readxml()
{
}


readxml::~readxml()
{
}



/**
* 通过根节点和节点名查找所有指定节点，结果放到节点数组NodeVector中
* @param pRootEle xml文件的根节点
* @param strNodeName 要查询的节点名
* @param NodeVector 查询到的节点指针数组
* @return 找到至少一个相应节点，返回true；否则false
*/
bool readxml::GetAllNodePointerByName(TiXmlElement* pRootEle, string strNodeName, vector<TiXmlElement*> &NodeVector)
{
	//如果NodeName等于根节点名，加入NodeVector数组
	
	if (strNodeName == pRootEle->Value())
	{
		NodeVector.push_back(pRootEle);//添加到数组末尾  
		//这里根据VOC Annotation的XML文件格式，认为相同节点名的节点不会有父子关系，所以所有相同节点名的节点都在同一级别上  
		//只要找到第一个，剩下的肯定在它的兄弟节点里面  
		for (TiXmlElement * pElement = pRootEle->NextSiblingElement(); pElement; pElement = pElement->NextSiblingElement())
		if (strNodeName == pElement->Value())
			NodeVector.push_back(pElement);
		return true;
	}
	TiXmlElement * pEle = pRootEle;
	for (pEle = pRootEle->FirstChildElement(); pEle; pEle = pEle->NextSiblingElement())
	{
		//递归处理子节点，获取节点指针  
		if (GetAllNodePointerByName(pEle, strNodeName, NodeVector))
			return true;
	}
	return false;//没找到  
}

/**
* 根据目标名过滤目标节点数组,删除所有目标名不是objectName的元素
* @param NodeVector 要操作的TiXmlElement元素指针数组
* @param objectName 指定的目标名，删除所有目标名不是objectName的元素
* @return 过滤后目标数组为空，返回false；否则返回true
*/
bool readxml::FiltObject(vector<TiXmlElement*> &NodeVector, string objectName)
{
	TiXmlElement * pEle = NULL;
	vector<TiXmlElement *>::iterator iter = NodeVector.begin();//数组的迭代器  
	for (; iter != NodeVector.end();)
	{
		pEle = *iter;//第i个元素  
		//若目标名不是objectName，删除此节点  
		if (objectName != pEle->FirstChildElement()->GetText())
		{
			//cout<<"删除的目标节点："<<pEle->FirstChildElement()->GetText() <<endl;  
			iter = NodeVector.erase(iter);//删除目标名不是objectName的，返回下一个元素的指针  
		}
		else
			iter++;
	}
	if (0 == NodeVector.size())//过滤后目标数组为空，说明不包含指定目标  
		return false;
	else
		return true;
}

/**
* @param NodeVector 目标节点数组
* @return xml的目标信息
*/
void readxml::GetBoundingBox( vector<TiXmlElement*> NodeVector)
{
	int xmin, ymin, xmax, ymax;//从目标节点中读出的包围盒参数  
	char fileName[256];//剪裁后的图片和其水平翻转图片的文件名  

	//遍历目标数组  
	vector<TiXmlElement *>::iterator iter = NodeVector.begin();//数组的迭代器  
	for (; iter != NodeVector.end(); iter++)
	{
		//遍历每个目标的子节点  
		TiXmlElement *pEle = (*iter)->FirstChildElement();//第i个元素的第一个孩子  
		int ncount = 0;
		XMLInfo tmpXmlInfo;
		for (; pEle; pEle = pEle->NextSiblingElement())
		{
		
			

			//找到包围盒"bndbox"节点  
			if (string("name")==pEle->Value())
			{
				string name = pEle->GetText();
				className_ = name;
			}

			if (string("bndbox") == pEle->Value())
			{
				ncount++;
				TiXmlElement * pCoord = pEle->FirstChildElement();//包围盒的第一个坐标值  
				//依次遍历包围盒的4个坐标值，放入整型变量中  
				for (; pCoord; pCoord = pCoord->NextSiblingElement())
				{
					if (string("xmin") == pCoord->Value())
						xmin = atoi(pCoord->GetText());//xmin  
					if (string("ymin") == pCoord->Value())
						ymin = atoi(pCoord->GetText());//ymin  
					if (string("xmax") == pCoord->Value())
						xmax = atoi(pCoord->GetText());//xmax  
					if (string("ymax") == pCoord->Value())
						ymax = atoi(pCoord->GetText());//ymax  
				}
				//cout<<"xmin:"<<xmin<<","<<"ymin:"<<ymin<<","<<"xmax:"<<xmax<<","<<"ymax:"<<ymax<<endl;;  
				//根据读取的包围盒坐标设置图像ROI  
				tmpXmlInfo.slabel = className_;
				tmpXmlInfo.xmin = xmin;
				tmpXmlInfo.ymin = ymin;
				tmpXmlInfo.xmax = xmax;
				tmpXmlInfo.ymax = ymax;
				vecXmlInfo_.push_back(tmpXmlInfo);
			}
		
		}
	}
}

/**
* 根据XML文件，从图像中剪裁出objectName目标
* @param XMLFile XML文件名
* @return 若图像中包含objectName目标，返回目标的类别及位置信息
*/
vector<XMLInfo> readxml::GetXmlInfo(string XMLFile, string objectName)
{
	vecXmlInfo_.clear();
	TiXmlDocument * pDoc = new TiXmlDocument();//创建XML文档  
	bool btrue = pDoc->LoadFile(XMLFile.c_str()); //装载XML文件  
	if (btrue==false)
		return vecXmlInfo_;

	vector<TiXmlElement*> nodeVector;//节点数组  

	//查找所有节点名是object的节点，即目标节点，结果放到节点数组nodeVector中  
	if (false == GetAllNodePointerByName(pDoc->RootElement(), "object", nodeVector))//未找到指定目标  
		return vecXmlInfo_;
	//cout<<"所有目标个数："<<nodeVector.size()<<endl;  

	//过滤节点数组，删除所有节点名不是objectName的节点  
	//if (false == FiltObject(nodeVector, objectName))//目标数组中没有指定目标  
	//	return vecXmlInfo_;
	//cout<<"过滤后的目标个数："<<nodeVector.size()<<endl;  

	//根据每个目标的BoundingBox，剪裁图像，保存为文件  
	GetBoundingBox(nodeVector);
	return vecXmlInfo_;
}


bool readxml::GetNodePointerByName(TiXmlElement* pRootEle, std::string &strNodeName, vector<TiXmlElement*> &NodeVector)
{
	// 假如等于根节点名，就退出  
	if (strNodeName == pRootEle->Value())
	{
		NodeVector.push_back(pRootEle);
		//Node = pRootEle;
		//return true;
	}
	TiXmlElement* pEle = pRootEle;
	for (pEle = pRootEle->FirstChildElement(); pEle; pEle = pEle->NextSiblingElement())
	{
		//递归处理子节点，获取节点指针  
		GetNodePointerByName(pEle, strNodeName, NodeVector);
		//if (GetNodePointerByName(pEle, strNodeName, NodeVector))
		//return true;
	}
	return false;
}

bool readxml::ModifyNode_Text(std::string& XmlFile, std::string& strNodeName, const std::string& strText)
{
	// 定义一个TiXmlDocument类指针
	TiXmlDocument *pDoc = new TiXmlDocument();
	if (NULL == pDoc)
	{
		return false;
	}
	pDoc->LoadFile(XmlFile.c_str());
	TiXmlElement *pRootEle = pDoc->RootElement();
	if (NULL == pRootEle)
	{
		return false;
	}

	TiXmlElement *pNode = NULL;
	vector<TiXmlElement*> nodeVector;//节点数组  
	GetNodePointerByName(pRootEle, strNodeName, nodeVector);
	for (int i = 0; i < nodeVector.size(); i++)
	{
		pNode = nodeVector[i];   //第i个元素  
		pNode->Clear();  // 首先清除所有文本
		// 然后插入文本，保存文件
		TiXmlText *pValue = new TiXmlText((char*)strText.c_str());
		pNode->LinkEndChild(pValue);
		pDoc->SaveFile((char*)XmlFile.c_str());
		//return true;

	}
}