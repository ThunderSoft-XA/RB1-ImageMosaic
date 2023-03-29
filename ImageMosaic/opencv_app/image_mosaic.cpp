#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "image_mosaic.hpp"

using namespace std;
using namespace cv;

int main(int argcm, char **argv)
{
    Mat img1 = imread("./track_left.png");
    Mat img2 = imread("./track_right.png");

    if (img1.empty() || img2.empty())
    {
        cout << "Can't read image" << endl;
        return -1;
    }

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
    // cv::resize(img1, img1, cv::Size(640,480));
    // cv::resize(img2, img2, cv::Size(640,480));
    cv::Mat mosaic_mat = image_mosaic_orb(img1,img2);
    gettimeofday(&stop_time, nullptr);
    std::cout << "local time: "
        << ((stop_time.tv_sec * 1000000 + stop_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec)) / 1000
        <<  " ms" << std::endl;

    imwrite("mosiac_result.png", mosaic_mat);

    return 0;

}