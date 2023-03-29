
#include <iostream>  
#include <fstream>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/stitching.hpp>

using namespace std;  
using namespace cv;  

bool try_use_gpu = false;
vector<Mat> imgs;
Stitcher::Mode mode = Stitcher::PANORAMA;
int main(int argc, char* argv[])  
{
    Mat img1 = imread("./track_left.png");
    Mat img2 = imread("./track_right.png");

    if (img1.empty() || img2.empty())
    {
        cout << "Can't read image" << endl;
        return -1;
    }

    int n = 2;
    vector<Mat> imgs;
    imgs.push_back(img1); 
    imgs.push_back(img2); 
    cout<<"......"<<endl;
    Mat pano;  
    double start = getTickCount();
    Ptr<Stitcher> stitcher = Stitcher::create(mode);
    Stitcher::Status status = stitcher->stitch(imgs, pano);  
    double end = getTickCount();
    double useTime = (end - start) / getTickFrequency();
    cout << "use-time : " << useTime << "s" << endl;
    if (status != Stitcher::OK)  
    {  
        cout << "无法拼接！" << endl;  
        return -1;  
    }  
    imwrite("result.jpg", pano); 

	return 0;  
}