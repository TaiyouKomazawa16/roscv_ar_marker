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

#define DEPTH_LIMIT_U16 15000 //mm

const int target_id = 6;
std::string ar_frame = "ar_mark_" + std::to_string(target_id);

tf2::Quaternion getQuaternion(cv::Vec3d rvec) {
    // 回転ベクトルrvecを3x3の回転行列Rに変換
    cv::Mat R;
    Rodrigues(rvec, R);

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

    ROS_INFO("marker T:[%f,%f,%f]\tR:[%f,%f,%f,%f]",
             tvec[0], tvec[1], tvec[2],
             q.x(), q.y(), q.z(), q.w());

    transformStamped.header.stamp = ros::Time::now();
    tfb.sendTransform(transformStamped);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "charuco_tracker");
    RosImageConverter rosimg("/camera/color/image_rect_color", "/camera/color/camera_info", 
                                    sensor_msgs::image_encodings::BGR8);
    RosImageConverter rosdepth("/camera/aligned_depth_to_color/image_raw", "/camera/aligned_depth_to_color/camera_info",
                                    sensor_msgs::image_encodings::TYPE_16UC1);
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Publisher img_pub;
    img_pub = it.advertise("/charuco_tracker/result", 1);

    //解像度の設定
    cv::Size img_size = rosimg.get_size();

    int dictionaryId = 8;
    float markerLength = 0.044; //マーカーの正方形の長さ[m]

    float squareLength = 0.065; //チェス盤の正方形の長さ[m]

    cv::Mat imgMatrix = rosimg.getK();
    cv::Mat depthMatrix = rosdepth.getK();
    cv::Mat distCoeffs;

    cv::Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    cv::Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();

    static tf2_ros::TransformBroadcaster tfb;
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.frame_id = "camera_color_frame";
    transformStamped.child_frame_id = ar_frame.c_str();
    transformStamped.header.stamp = ros::Time::now();
    tfb.sendTransform(transformStamped);

    vector<int> ids, charucoIds;
    vector<cv::Vec4i> diamondIds;
    vector<vector<cv::Point2f>> corners, rejected, diamondCorners;
    vector<cv::Point2f> charucoCorners;
    vector<cv::Vec3d> rvecs, tvecs;

    ros::Rate rate(30);

    while (ros::ok())
    {
        cv_bridge::CvImagePtr rgb_ptr, depth_ptr;
        rosimg.get_img(rgb_ptr);
        rosdepth.get_img(depth_ptr);
        if (rgb_ptr)
        {

            // マーカ検出(入力: 引数1,2,5  出力: 引数3,4,6
            aruco::detectMarkers(rgb_ptr->image, dictionary, corners, ids, detectorParams, rejected);

            if (ids.size() > 3)
            {
                aruco::detectCharucoDiamond(rgb_ptr->image, corners, ids, squareLength / markerLength, diamondCorners, diamondIds, imgMatrix, distCoeffs);
                if (diamondIds.size() > 0)
                {
                    aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, imgMatrix, distCoeffs, rvecs, tvecs);
                }

                if (diamondIds.size() > 0)
                { //マーカ検出の結果
                    //aruco::drawDetectedMarkers(rgb_ptr->image, diamondCorners, ids);
                    //マーカ姿勢推定の結果
                    //aruco::drawDetectedDiamonds(rgb_ptr->image, diamondCorners, diamondIds);
                    for (unsigned int i = 0; i < diamondIds.size(); i++)
                    {
                        //aruco::drawAxis(rgb_ptr->image, imgMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 0.5f);
                        cv::Vec4i id = diamondIds[i];

                        if (target_id == id[0]+id[1]+id[2]+id[3]){
                            vector<cv::Point2f> dcor = diamondCorners[i];

                            cv::Point pt[4] = {dcor[0], dcor[1], dcor[2], dcor[3]};
                            /*マーカの重心座標の検出*/
                            double s1 = ((pt[3].x-pt[1].x)*(pt[0].y-pt[1].y)-(pt[3].y-pt[1].y)*(pt[0].x-pt[1].x)) / 2.0;
                            double s2 = ((pt[3].x-pt[1].x)*(pt[1].y-pt[2].y)-(pt[3].y-pt[1].y)*(pt[1].x-pt[2].x)) / 2.0;

                            cv::Point ptc; //重心座標
                            ptc.x = pt[0].x + (pt[2].x - pt[0].x) * s1 / (s1 + s2);
                            ptc.y = pt[0].y + (pt[2].y - pt[0].y) * s1 / (s1 + s2);

                            /*マーカのデプス領域の切り出し*/
                            cv::Point min_pt = pt[0];
                            cv::Point max_pt = pt[0];
                            for(int i = 1; i < 4; i++){
                                if(min_pt.x > pt[i].x)
                                    min_pt.x = pt[i].x;
                                if(max_pt.x < pt[i].x)
                                    max_pt.x = pt[i].x;
                                if(min_pt.y > pt[i].y)
                                    min_pt.y = pt[i].y;
                                if(max_pt.y < pt[i].y)
                                    max_pt.y = pt[i].y;
                            }
                            cv::Mat dmarker = depth_ptr->image(cv::Rect(min_pt.x,min_pt.y,max_pt.x-min_pt.x,max_pt.y-min_pt.y));
                            cv::Size sz = dmarker.size();

                            /*最小二乗法による点群の平面フィッティング*/
                            /* M*U = V */
                            cv::Mat M = cv::Mat::zeros(3,3, CV_64F);
                            cv::Mat U = cv::Mat::zeros(3,1, CV_64F);
                            cv::Mat V = cv::Mat::zeros(3,1, CV_64F);

                            for(int i = 0; i < sz.height; i++){ //画像y方向
                                for(int j = 0; j < sz.width; j++){ //画像x方向
                                    ushort depth = dmarker.at<ushort>(j, i);
                                    if(DEPTH_LIMIT_U16 >= depth && depth != 0){
                                        double _i = (double)i / sz.height;  //y方向の画質の正規化
                                        double _j = (double)j / sz.width;   //x方向の画質の正規化
                                        M.at<double>(0,0) += 1.0;
                                        M.at<double>(1,1) += _j*_j;
                                        M.at<double>(2,2) += _i*_i;
                                        M.at<double>(1,2) += _j*_i;
                                        M.at<double>(2,1) += _j*_i;
                                        M.at<double>(0,1) += _j;
                                        M.at<double>(1,0) += _j;
                                        M.at<double>(0,2) += _i;
                                        M.at<double>(2,0) += _i;
                                        U.at<double>(0,0) += depth;
                                        U.at<double>(1,0) += _j*depth;
                                        U.at<double>(2,0) += _i*depth;
                                    }
                                }
                            }

                            cv::solve(M, U, V, DECOMP_LU); //LU分解にて平面関数の定数行列(V)を導出

                            //マーカ中心部の推定距離を測定
                            double depth_center = V.at<double>(0,0) + V.at<double>(1,0)*(0.5) + V.at<double>(2,0)*(0.5);
                            //カメラパラメータより実距離のx,yを求める。
                            double x = ((ptc.x - depthMatrix.at<double>(0, 2))/depthMatrix.at<double>(0, 0))*depth_center;
                            double y = ((ptc.y - depthMatrix.at<double>(1, 2))/depthMatrix.at<double>(1, 1))*depth_center;

                            ROS_INFO("depth:[%lf,%lf,%lf]", x/1E3, y/1E3, depth_center/1E3);
                            
                            cv::Mat flat = cv::Mat::zeros(50,50, CV_16UC1);

                            for(int i = 0; i < 50; i++){
                                for(int j = 0; j < 50; j++){
                                    double depth = V.at<double>(0,0) + V.at<double>(1,0)*(j/50.0) + V.at<double>(2,0)*(i/50.0);
                                    if(depth < 0){ depth = 0; }
                                    flat.at<ushort>(j,i) = depth;
                                }
                            }

                            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "16UC1", flat).toImageMsg();
                            img_pub.publish(msg);

                            ar_broadcast(tvecs[i], rvecs[i]);
                        }
                    }
                }
            }

            //cv::imshow("Aruco", imageCopy); //結果をウィンドウ(Aruco)に表示
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
