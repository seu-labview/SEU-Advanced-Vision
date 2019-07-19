#include<iostream>
#include<opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/videoio.hpp>
#include <string>
using namespace cv;
using namespace std;

int main() {
	int i = 1;
	
	VideoCapture capture(1);    // 打开摄像头


	if (!capture.isOpened())    // 判断是否打开成功
	{
		return -1;
	}
	namedWindow("camera");
	
	while (true) {
		Mat frame;
		capture >> frame;    // 读取图像帧至frame	
		if (!frame.empty())	// 判断是否为空		
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