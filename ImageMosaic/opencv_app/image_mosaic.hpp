#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
// #include <gst/gst.h>

#include <omp.h>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

void CalcCorners(const Mat& H, const Mat& src)
{
    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

    V1 = H * V2;
    //左上角(0,0,1)
    cout << "V2: " << V2 << endl;
    cout << "V1: " << V1 << endl;
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    //左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    //右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    //右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
    V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];

}

//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
    int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界

    double processWidth = img1.cols - start;//重叠区域的宽度
    int rows = dst.rows;
    int cols = img1.cols; //注意，是列数*通道数
    double alpha = 1;//img1中像素的权重
    for (int i = 0; i < rows; i++)
    {
        uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
        uchar* t = trans.ptr<uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = start; j < cols; j++)
        {
            //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
            if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
            {
                alpha = 1;
            }
            else
            {
                //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
                alpha = (processWidth - (j - start)) / processWidth;
            }
            d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
            d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
            d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
        }
    }
}

//计算配准图的四个顶点坐标
cv::Mat image_mosaic(cv::Mat &left, cv::Mat &right)
{
    assert(left.empty() != true && right.empty() != true);
    //1.特征点提取和匹配
    //创建SURF对象
    //create 参数 海森矩阵阈值
    cv::Ptr<SURF> surf;
    surf = SURF::create(5000);

    //暴力匹配器
    BFMatcher matcher;

    vector<KeyPoint> key1, key2;
    Mat c, d;

    //寻找特征点
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, nullptr);
#pragma omp parallel sections    
{  
#pragma omp section    
    {
        surf->detectAndCompute(left, Mat(), key2, d);
    }
#pragma omp section    
    {
        surf->detectAndCompute(right, Mat(), key1, c);
    }
}
    gettimeofday(&stop_time, nullptr);
    std::cout << "detectAndCompute local time: "
        << ((stop_time.tv_sec * 1000000 + stop_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec)) / 1000
        <<  " ms" << std::endl;
    
    //特征点对比 保存
    vector<DMatch>matches;

    //使用暴力匹配器匹配特征点 保存
    matcher.match(d, c, matches);

    //排序 从小到大
    sort(matches.begin(), matches.end());

    //保留最优的特征点收集
    vector<DMatch>good_matches;

    int ptrPoint = std::min(30, (int)(matches.size()*0.4));

    for(int i=0; i<ptrPoint; i++)
        good_matches.push_back(matches[i]);

    //最佳匹配的特征点连成一线
    Mat outimg;
    gettimeofday(&start_time, nullptr);
    drawMatches(left, key2, right, key1, good_matches, outimg,
                Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    gettimeofday(&stop_time, nullptr);
    std::cout << "drawMatches local time: "
        << ((stop_time.tv_sec * 1000000 + stop_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec)) / 1000
        <<  " ms" << std::endl;
    //imshow("outimg", outimg);

    //2.图像配准
    //特征点配准
    vector<Point2f>imagepoint1, imagepoint2;
    for(int i=0; i<good_matches.size(); i++)
    {
        imagepoint1.push_back(key1[good_matches[i].trainIdx].pt);
        imagepoint2.push_back(key2[good_matches[i].queryIdx].pt);
    }

    //透视转换
    gettimeofday(&start_time, nullptr);
    Mat homo = findHomography(imagepoint1, imagepoint2, cv::RANSAC);
    gettimeofday(&stop_time, nullptr);
    std::cout << "findHomography local time: "
        << ((stop_time.tv_sec * 1000000 + stop_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec)) / 1000
        <<  " ms" << std::endl;
    //imshow("homo", homo);

    //右图四个顶点坐标转换计算

    CalcCorners(homo, right);

    Mat imageTransform;
    gettimeofday(&start_time, nullptr);
    warpPerspective(right, imageTransform, homo,
                    Size(MAX(corners.right_top.x, corners.right_bottom.x), left.rows));
    //imshow("imageTransform", imageTransform);
    gettimeofday(&stop_time, nullptr);
    std::cout << "warpPerspective local time: "
        << ((stop_time.tv_sec * 1000000 + stop_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec)) / 1000
        <<  " ms" << std::endl;
    
    //3.图像拷贝
    int dst_width = imageTransform.cols;
    int dst_height = imageTransform.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform.copyTo(dst(Rect(0, 0, imageTransform.cols, imageTransform.rows)));
    left.copyTo(dst(Rect(0, 0, left.cols, left.rows)));

    //4.优化拼接最终结果图，去除黑边
    gettimeofday(&start_time, nullptr);
    OptimizeSeam(left, imageTransform, dst);
    gettimeofday(&stop_time, nullptr);
    std::cout << "OptimizeSeam local time: "
        << ((stop_time.tv_sec * 1000000 + stop_time.tv_usec) - (start_time.tv_sec * 1000000 + start_time.tv_usec)) / 1000
        <<  " ms" << std::endl;
    //imshow("dst", dst);

    // waitKey(0);

    return dst;
}


cv::Mat image_mosaic_orb(cv::Mat &left, cv::Mat &right)
{
    Mat imageRight = right; //imread("images/flowerR.jpg", 1);    //右图
    Mat imageLeft = left; //imread("images/flowerL.jpg", 1);    //左图

    Mat image_r, image_l;
    Rect rect_right,rect_left;
    Mat image_r_rect, image_l_rect;
    double start = getTickCount();
        //提取特征点    
    Ptr<FeatureDetector> ORBDetector = ORB::create(1000);
 
    vector<KeyPoint> keyPoints_r, keyPoints_l;
    //特征点描述，为下边的特征点匹配做准备  
    Ptr<DescriptorExtractor> ORBDescriptor = ORB::create(1000);
    Mat imageDesc_r, imageDesc_l;
#pragma omp parallel sections    
{  
#pragma omp section    
    {
        //灰度图转换  
        cvtColor(imageRight, image_r, COLOR_BGR2GRAY);
        //直接从可能重复的区域提取特征点匹配 当前是左右图在拼接处大概有1/3是重复的
        // rect_right = Rect(0, 0, 11 * imageRight.cols / 12, imageRight.rows);
        // image_r_rect = imageRight(Rect(rect_right));
        ORBDetector->detect(image_r, keyPoints_r);
        ORBDescriptor->compute(image_r, keyPoints_r, imageDesc_r);
    }  
#pragma omp section    
    {
        cvtColor(imageLeft, image_l, COLOR_BGR2GRAY);
        // rect_left = Rect(imageLeft.cols / 12, 0, (11 * imageLeft.cols/12) -1 , imageLeft.rows);
        // image_l_rect = imageLeft(Rect(rect_left));
        ORBDetector->detect(image_l, keyPoints_l);
        ORBDescriptor->compute(image_l, keyPoints_l, imageDesc_l);
    }  
}
    printf("===========1==============\n");
 
    flann::Index flannIndex(imageDesc_r, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
 
    vector<DMatch> GoodMatchePoints;
 
    Mat macthIndex(imageDesc_l.rows, 2, CV_32SC1), matchDistance(imageDesc_l.rows, 2, CV_32FC1);
    flannIndex.knnSearch(imageDesc_l, macthIndex, matchDistance, 2, flann::SearchParams());

    printf("===========2==============\n");
    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchDistance.rows; i++)
    {
        if (matchDistance.at<float>(i, 0) < 0.4 * matchDistance.at<float>(i, 1))
        {
            DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
            GoodMatchePoints.push_back(dmatches);
        }
    }
    printf("===========3==============\n");

    Mat first_match;
    //drawMatches(imageLeft, keyPoints_l, imageRight, keyPoints_r, GoodMatchePoints, first_match);
    drawMatches(image_l, keyPoints_l, image_r, keyPoints_r, GoodMatchePoints, first_match);
    //namedWindow("first_match ", 2);
    //imshow("first_match ", first_match);
    //waitKey();
    printf("===========4==============\n");
 
    vector<Point2f> imagePoints1, imagePoints2;
 
    for (int i = 0; i < GoodMatchePoints.size(); i++) {
        imagePoints2.push_back(keyPoints_l[GoodMatchePoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoints_r[GoodMatchePoints[i].trainIdx].pt);
    }

    printf("===========5==============\n");
    
    printf("imagePoints1.size() = %ld, imagePoints2.size() = %ld\n",imagePoints1.size(),imagePoints2.size());
    if (imagePoints1.size() <= 10 || imagePoints2.size() <= 10) {
        printf("There is little keypoints\n");
    }
 
    //将左图的坐标转化到原图的位置，否则其变换矩阵在x方向的平稳不对
    for (auto iter = imagePoints2.begin(); iter != imagePoints2.end(); iter++) {
        (*iter).x  += imageLeft.cols / 2;
    }
 
 
    //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
    Mat homo = findHomography(imagePoints1, imagePoints2, RANSAC);
   // 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差
    //Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
    cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵      
          
     //计算配准图的四个顶点坐标
    CalcCorners(homo, imageRight);
 
    //cout << "left_top:" << corners.left_top << endl;
    //cout << "left_bottom:" << corners.left_bottom << endl;
    //cout << "right_top:" << corners.right_top << endl;
    //cout << "right_bottom:" << corners.right_bottom << endl;
 
 
    //图像配准  
    Mat imageTransform1, imageTransform2;
    warpPerspective(imageRight, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), imageLeft.rows));
    rectangle(imageRight, Rect(imageRight.cols - MAX(corners.right_top.x, corners.right_bottom.x), 0, MAX(corners.right_top.x, corners.right_bottom.x), 500), (0, 0, 255));
    //imshow("rectangle", imageRight);
    //imshow("透视矩阵变换Right", imageTransform1);
    //imwrite("trans1.jpg", imageTransform1);
    //waitKey();
 
 
    //创建拼接后的图,需提前计算图的大小
    int move_x = 900;
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
 
    int dst_height = imageLeft.rows;
 
    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);
 
    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    //imshow("transform1", dst);
 
    imageLeft.copyTo(dst(Rect(0, 0, imageLeft.cols, imageLeft.rows)));
    //imshow("b_dst", dst);
    //waitKey();
 
    OptimizeSeam(imageLeft, imageTransform1, dst);
 
    double end = getTickCount();
    double useTime = (end - start) / getTickFrequency();
    cout << "use-time : " << useTime << "s" << endl;
 
    //imshow("dstOptimize", dst);
    //imwrite("dst.jpg", dst);
 
    //waitKey();
 
    return dst;
}