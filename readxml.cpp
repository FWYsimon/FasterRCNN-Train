#include "readxml.h"


readxml::readxml()
{
}


readxml::~readxml()
{
}



/**
* ͨ�����ڵ�ͽڵ�����������ָ���ڵ㣬����ŵ��ڵ�����NodeVector��
* @param pRootEle xml�ļ��ĸ��ڵ�
* @param strNodeName Ҫ��ѯ�Ľڵ���
* @param NodeVector ��ѯ���Ľڵ�ָ������
* @return �ҵ�����һ����Ӧ�ڵ㣬����true������false
*/
bool readxml::GetAllNodePointerByName(TiXmlElement* pRootEle, string strNodeName, vector<TiXmlElement*> &NodeVector)
{
	//���NodeName���ڸ��ڵ���������NodeVector����
	
	if (strNodeName == pRootEle->Value())
	{
		NodeVector.push_back(pRootEle);//��ӵ�����ĩβ  
		//�������VOC Annotation��XML�ļ���ʽ����Ϊ��ͬ�ڵ����Ľڵ㲻���и��ӹ�ϵ������������ͬ�ڵ����Ľڵ㶼��ͬһ������  
		//ֻҪ�ҵ���һ����ʣ�µĿ϶��������ֵܽڵ�����  
		for (TiXmlElement * pElement = pRootEle->NextSiblingElement(); pElement; pElement = pElement->NextSiblingElement())
		if (strNodeName == pElement->Value())
			NodeVector.push_back(pElement);
		return true;
	}
	TiXmlElement * pEle = pRootEle;
	for (pEle = pRootEle->FirstChildElement(); pEle; pEle = pEle->NextSiblingElement())
	{
		//�ݹ鴦���ӽڵ㣬��ȡ�ڵ�ָ��  
		if (GetAllNodePointerByName(pEle, strNodeName, NodeVector))
			return true;
	}
	return false;//û�ҵ�  
}

/**
* ����Ŀ��������Ŀ��ڵ�����,ɾ������Ŀ��������objectName��Ԫ��
* @param NodeVector Ҫ������TiXmlElementԪ��ָ������
* @param objectName ָ����Ŀ������ɾ������Ŀ��������objectName��Ԫ��
* @return ���˺�Ŀ������Ϊ�գ�����false�����򷵻�true
*/
bool readxml::FiltObject(vector<TiXmlElement*> &NodeVector, string objectName)
{
	TiXmlElement * pEle = NULL;
	vector<TiXmlElement *>::iterator iter = NodeVector.begin();//����ĵ�����  
	for (; iter != NodeVector.end();)
	{
		pEle = *iter;//��i��Ԫ��  
		//��Ŀ��������objectName��ɾ���˽ڵ�  
		if (objectName != pEle->FirstChildElement()->GetText())
		{
			//cout<<"ɾ����Ŀ��ڵ㣺"<<pEle->FirstChildElement()->GetText() <<endl;  
			iter = NodeVector.erase(iter);//ɾ��Ŀ��������objectName�ģ�������һ��Ԫ�ص�ָ��  
		}
		else
			iter++;
	}
	if (0 == NodeVector.size())//���˺�Ŀ������Ϊ�գ�˵��������ָ��Ŀ��  
		return false;
	else
		return true;
}

/**
* @param NodeVector Ŀ��ڵ�����
* @return xml��Ŀ����Ϣ
*/
void readxml::GetBoundingBox( vector<TiXmlElement*> NodeVector)
{
	int xmin, ymin, xmax, ymax;//��Ŀ��ڵ��ж����İ�Χ�в���  
	char fileName[256];//���ú��ͼƬ����ˮƽ��תͼƬ���ļ���  

	//����Ŀ������  
	vector<TiXmlElement *>::iterator iter = NodeVector.begin();//����ĵ�����  
	for (; iter != NodeVector.end(); iter++)
	{
		//����ÿ��Ŀ����ӽڵ�  
		TiXmlElement *pEle = (*iter)->FirstChildElement();//��i��Ԫ�صĵ�һ������  
		int ncount = 0;
		XMLInfo tmpXmlInfo;
		for (; pEle; pEle = pEle->NextSiblingElement())
		{
		
			

			//�ҵ���Χ��"bndbox"�ڵ�  
			if (string("name")==pEle->Value())
			{
				string name = pEle->GetText();
				className_ = name;
			}

			if (string("bndbox") == pEle->Value())
			{
				ncount++;
				TiXmlElement * pCoord = pEle->FirstChildElement();//��Χ�еĵ�һ������ֵ  
				//���α�����Χ�е�4������ֵ���������ͱ�����  
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
				//���ݶ�ȡ�İ�Χ����������ͼ��ROI  
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
* ����XML�ļ�����ͼ���м��ó�objectNameĿ��
* @param XMLFile XML�ļ���
* @return ��ͼ���а���objectNameĿ�꣬����Ŀ������λ����Ϣ
*/
vector<XMLInfo> readxml::GetXmlInfo(string XMLFile, string objectName)
{
	vecXmlInfo_.clear();
	TiXmlDocument * pDoc = new TiXmlDocument();//����XML�ĵ�  
	bool btrue = pDoc->LoadFile(XMLFile.c_str()); //װ��XML�ļ�  
	if (btrue==false)
		return vecXmlInfo_;

	vector<TiXmlElement*> nodeVector;//�ڵ�����  

	//�������нڵ�����object�Ľڵ㣬��Ŀ��ڵ㣬����ŵ��ڵ�����nodeVector��  
	if (false == GetAllNodePointerByName(pDoc->RootElement(), "object", nodeVector))//δ�ҵ�ָ��Ŀ��  
		return vecXmlInfo_;
	//cout<<"����Ŀ�������"<<nodeVector.size()<<endl;  

	//���˽ڵ����飬ɾ�����нڵ�������objectName�Ľڵ�  
	//if (false == FiltObject(nodeVector, objectName))//Ŀ��������û��ָ��Ŀ��  
	//	return vecXmlInfo_;
	//cout<<"���˺��Ŀ�������"<<nodeVector.size()<<endl;  

	//����ÿ��Ŀ���BoundingBox������ͼ�񣬱���Ϊ�ļ�  
	GetBoundingBox(nodeVector);
	return vecXmlInfo_;
}


bool readxml::GetNodePointerByName(TiXmlElement* pRootEle, std::string &strNodeName, vector<TiXmlElement*> &NodeVector)
{
	// ������ڸ��ڵ��������˳�  
	if (strNodeName == pRootEle->Value())
	{
		NodeVector.push_back(pRootEle);
		//Node = pRootEle;
		//return true;
	}
	TiXmlElement* pEle = pRootEle;
	for (pEle = pRootEle->FirstChildElement(); pEle; pEle = pEle->NextSiblingElement())
	{
		//�ݹ鴦���ӽڵ㣬��ȡ�ڵ�ָ��  
		GetNodePointerByName(pEle, strNodeName, NodeVector);
		//if (GetNodePointerByName(pEle, strNodeName, NodeVector))
		//return true;
	}
	return false;
}

bool readxml::ModifyNode_Text(std::string& XmlFile, std::string& strNodeName, const std::string& strText)
{
	// ����һ��TiXmlDocument��ָ��
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
	vector<TiXmlElement*> nodeVector;//�ڵ�����  
	GetNodePointerByName(pRootEle, strNodeName, nodeVector);
	for (int i = 0; i < nodeVector.size(); i++)
	{
		pNode = nodeVector[i];   //��i��Ԫ��  
		pNode->Clear();  // ������������ı�
		// Ȼ������ı��������ļ�
		TiXmlText *pValue = new TiXmlText((char*)strText.c_str());
		pNode->LinkEndChild(pValue);
		pDoc->SaveFile((char*)XmlFile.c_str());
		//return true;

	}
}