#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#include "image_mosaic.hpp"
#include "camera2appsink.hpp"
#include "appsrc2rtsp.hpp"

int main(int argcm, char **argv)
{
    std::string json_file = "./gst_mosaic_config.json";

    /* Initialize GStreamer */
    gst_init (nullptr, nullptr);
    GMainLoop *main_loop;  /* GLib's Main Loop */
    /* Create a GLib Main Loop and set it to run */
    main_loop = g_main_loop_new (NULL, FALSE);
    Queue<cv::Mat> mat_queue;
    // Queue<cv::Mat> mat_queue_1;
    CameraPipe camera_pipe(json_file);

    camera_pipe.initPipe();
    camera_pipe.rgb_mat_queue = mat_queue;

    camera_pipe.checkElements();

    camera_pipe.setProperty();

    camera_pipe.runPipeline();

    g_print("json_file : %s\n",json_file.c_str());
	Queue<cv::Mat> push_queue;
	APPSrc2RtspSink push_pipe(json_file);
	if( push_pipe.initPipe() == -1 ) {
		return 0;
	}

	push_pipe.push_mat_queue = push_queue;

	if(push_pipe.checkElements() == -1) {
		return 0;
	}

	push_pipe.setProperty();

    cv::Mat mosaic_mat;
    cv::Mat left_mat,right_mat;
    left_mat = cv::imread("./track_left.png");
    std::thread([&]() {
        while(1) {
            if(camera_pipe.rgb_mat_queue.empty()) {
                continue;
            }
            
            right_mat = camera_pipe.rgb_mat_queue.pop();
            mosaic_mat = image_mosaic_orb(left_mat,right_mat);
            if(mosaic_mat.empty()) {
                continue;
            }
            cv::resize(mosaic_mat,mosaic_mat,{640,480});

            push_pipe.push_mat_queue.push_back(mosaic_mat);
        }
    }).detach();

    sleep(4);
    std::thread([&](){
		push_pipe.runPipe();
    }).detach();

    g_main_loop_run (main_loop);

    camera_pipe.~CameraPipe();
    g_main_loop_unref (main_loop);

    return 0;

}