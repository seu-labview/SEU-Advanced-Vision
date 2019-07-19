#include<iostream>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/videoio.hpp>
#include <string>
using namespace cv;
using namespace std;

int main() {
	int i = 1;
	
	VideoCapture capture(1);    // ������ͷ


	if (!capture.isOpened())    // �ж��Ƿ�򿪳ɹ�
	{
		return -1;
	}
	namedWindow("camera");
	
	while (true) {
		Mat frame;
		capture >> frame;    // ��ȡͼ��֡��frame	
		if (!frame.empty())	// �ж��Ƿ�Ϊ��		
		{
			stringstream ss;
			ss << i;
			string b = ss.str();
			string filename="C://Users//28997//Desktop//camera//"+b + ".jpg";
			imwrite(filename, frame);
		}
		i++;
		if (i == 1001)
			break;
		waitKey(100);
	}


}