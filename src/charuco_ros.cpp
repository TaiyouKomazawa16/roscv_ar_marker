#include "ros/ros.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <opencv2/calib3d.hpp>   // 回転ベクトル↔回転行列の変換に使用
#include <eigen3/Eigen/Geometry> // 四元数の計算に使用

#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <vector>
#include <iostream>

#include <bits/stdc++.h> //円周率用

#include "ros_to_cv2_conv.hpp"

using namespace std;
using namespace cv;
using namespace aruco;

const int target_id = 6;
std::string ar_frame = "ar_mark_" + std::to_string(target_id);

tf2::Quaternion getQuaternion(Vec3d rvec) {
    // 回転ベクトルrvecを3x3の回転行列Rに変換
    Mat R;
    Rodrigues(rvec, R);
    //cout << "R: \n" << R << endl;// 確認用

    // Mat型配列の要素をMatrix3d型配列に格納
    Eigen::Matrix3d eigen_R;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            eigen_R(i, j) = R.at<double>(i, j);
        }
    }

    // 回転行列Rを四元数Qに変換
    Eigen::Quaterniond Q(eigen_R);
    tf2::Quaternion q_tf(Q.z(),-Q.x(),-Q.y(),Q.w());

    return q_tf;
}

void ar_broadcast(Vec3d tvec, Vec3d rvec)
{
    static tf2_ros::TransformBroadcaster tfb;
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.frame_id = "camera_color_frame";
    transformStamped.child_frame_id = ar_frame.c_str();
    transformStamped.transform.translation.x = tvec[2];
    transformStamped.transform.translation.y = -tvec[0];
    transformStamped.transform.translation.z = -tvec[1];
    tf2::Quaternion q;
    q = getQuaternion(rvec);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    ROS_INFO("T:[%f,%f,%f]\tR:[%f,%f,%f,%f]",
             tvec[0], tvec[1], tvec[2],
             q.x(), q.y(), q.z(), q.w());

    transformStamped.header.stamp = ros::Time::now();
    tfb.sendTransform(transformStamped);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "charuco_tracker");
    RosImageConverter rosimg;
    ros::NodeHandle nh;
    //image_transport::ImageTransport it(nh);
    //image_transport::Publisher img_pub;
    //img_pub = it.advertise("/charuco_tracker/result", 1);

    //解像度の設定
    int w = 640, h = 480;

    int dictionaryId = 8;
    float markerLength = 0.044; //マーカーの正方形の長さ[m]

    float squareLength = 0.065; //チェス盤の正方形の長さ[m]

    float cx = (w - 1) / 2, cy = (h - 1) / 2;
    //cv::Mat camMatrix = (cv::Mat_<double>(3,3) << 614.711181640625, 0.0, 326.2210388183594, 0.0, 614.7484741210938, 233.19827270507812, 0.0, 0.0, 1.0);//オリジナルのカメラ行列
    cv::Mat camMatrix = (cv::Mat_<double>(3, 3) << 614.711181640625, 0.0, cx, 0.0, 614.7484741210938, cy, 0.0, 0.0, 1.0);
    cv::Mat distCoeffs;

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    static tf2_ros::TransformBroadcaster tfb;
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.frame_id = "camera_color_frame";
    transformStamped.child_frame_id = ar_frame.c_str();
    transformStamped.header.stamp = ros::Time::now();
    tfb.sendTransform(transformStamped);

    vector<int> ids, charucoIds;
    vector<Vec4i> diamondIds;
    vector<vector<Point2f>> corners, rejected, diamondCorners;
    vector<Point2f> charucoCorners;
    vector<Vec3d> rvecs, tvecs;

    ros::Rate rate(30);

    while (ros::ok())
    {
        cv_bridge::CvImagePtr cv_ptr;
        rosimg.get_img(cv_ptr);
        if (cv_ptr)
        {

            // マーカ検出(入力: 引数1,2,5  出力: 引数3,4,6
            aruco::detectMarkers(cv_ptr->image, dictionary, corners, ids, detectorParams, rejected);

            if (ids.size() > 3)
            {
                aruco::detectCharucoDiamond(cv_ptr->image, corners, ids, squareLength / markerLength, diamondCorners, diamondIds, camMatrix, distCoeffs);
                if (diamondIds.size() > 0)
                {
                    aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, camMatrix, distCoeffs, rvecs, tvecs);
                }

                if (diamondIds.size() > 0)
                { //マーカ検出の結果
                    //aruco::drawDetectedMarkers(cv_ptr->image, diamondCorners, ids);
                    //マーカ姿勢推定の結果
                    for (unsigned int i = 0; i < diamondIds.size(); i++)
                    {
                        //aruco::drawAxis(cv_ptr->image, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 0.5f);

                        Vec4i id = diamondIds[i];

                        if (target_id == id[0]+id[1]+id[2]+id[3])
                            ar_broadcast(tvecs[i], rvecs[i]);
                    }
                }
            }

            //cv::imshow("Aruco", imageCopy); //結果をウィンドウ(Aruco)に表示
            //img_pub.publish(cv_ptr->toImageMsg());
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
