#include "ros/ros.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <opencv2/aruco.hpp>
#include <iostream>

#include <bits/stdc++.h> //円周率用

#include "ros_to_cv2_conv.hpp"

using namespace std;
using namespace cv;
using namespace aruco;

void ar_broadcast(int id, Vec3d tvec, Vec3d rvec)
{
    static tf2_ros::TransformBroadcaster tfb;
    geometry_msgs::TransformStamped transformStamped;

    std::string str = "ar_mark_" + std::to_string(id);

    transformStamped.header.frame_id = "camera_color_frame";
    transformStamped.child_frame_id = str.c_str();
    transformStamped.transform.translation.x = tvec[2];
    transformStamped.transform.translation.y = tvec[0];
    transformStamped.transform.translation.z = tvec[1];
    tf2::Quaternion q;
    q.setRPY(rvec[2], rvec[0], rvec[1]);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    transformStamped.header.stamp = ros::Time::now();
    tfb.sendTransform(transformStamped);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "aruco_tracker");
    RosImageConverter rosimg(sensor_msgs::image_encodings::BGR8);
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub;
    img_pub = it.advertise("/aruco_tracker/result", 1);

    //解像度の設定
    int frameSize = 0;
    int w = 640, h = 480;
    if (frameSize = 1)
        int w = 848, h = 480;
    else if (frameSize = 2)
        int w = 960, h = 540;
    else if (frameSize = 3)
        int w = 1280, h = 720;
    else if (frameSize = 4)
        int w = 1920, h = 1080;

    int dictionaryId = 8;
    bool estimatePose = true;
    float markerLength = 0.065;
    float cx = (w - 1) / 2, cy = (h - 1) / 2;
    //cv::Mat camMatrix = (cv::Mat_<double>(3,3) << 614.711181640625, 0.0, 326.2210388183594, 0.0, 614.7484741210938, 233.19827270507812, 0.0, 0.0, 1.0);//オリジナルのカメラ行列
    cv::Mat camMatrix = (cv::Mat_<double>(3, 3) << 614.711181640625, 0.0, cx, 0.0, 614.7484741210938, cy, 0.0, 0.0, 1.0);
    cv::Mat distCoeffs;

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    vector<int> ids;
    vector<vector<Point2f>> corners, rejected;
    vector<Vec3d> rvecs, tvecs;

    ros::Rate rate(10);

    while(ros::ok())
    {
        cv_bridge::CvImagePtr cv_ptr;
        rosimg.get_img(cv_ptr);
        if (cv_ptr){

            // マーカ検出(入力: 引数1,2,5  出力: 引数3,4,6
            aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids, detectorParams, rejected);

            //マーカ姿勢推定(入力: 引数1,2,3,4  出力: 引数5,6)
            if (estimatePose && ids.size() > 0)
            {
                aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);
            }

            if (ids.size() > 0)
            { //マーカ検出の結果
                aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);

                if (estimatePose)
                { //マーカ姿勢推定の結果
                    for (unsigned int i = 0; i < ids.size(); i++)
                    {
                        cout
                            << "id: " << ids[1]
                            << " Txyz: " << tvecs[1]
                            << " Rxyz: " << rvecs[1] * 180 / M_PI
                            << endl; //確認用

                        ar_broadcast(ids[i], tvecs[i], rvecs[i]);
                    }
                }
            }

            img_pub.publish(cv_ptr->toImageMsg());
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
