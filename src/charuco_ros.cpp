#include "ros/ros.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>

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

const int target_id = 6;
std::string ar_frame = "ar_mark_" + std::to_string(target_id);
std::string ar_frame_depth = "ar_mark_" + std::to_string(target_id) + "_depth";

cv::Vec3d getEuler(cv::Vec3d rvec)
{
    // 回転ベクトルrvecを3x3の回転行列Rに変換
    cv::Mat R;
    Rodrigues(rvec, R);
    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6; // If

    float x, y, z;
    if(!singular){
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }else{
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    cv::Vec3d ea_;
    ea_[0] = -z;
    ea_[2] = y;
    ea_[1] = x;

    return ea_;
}

void ar_broadcast(ros::Publisher &pub, std::string ar_frame, Vec3d tvec, Vec3d rpy)
{
    static tf2_ros::TransformBroadcaster tfb;
    geometry_msgs::TransformStamped transformStamped;
    geometry_msgs::PoseWithCovarianceStamped msg;

    transformStamped.header.frame_id = "camera_color_frame";
    msg.header.frame_id = "camera_color_frame";
    transformStamped.child_frame_id = ar_frame.c_str();
    transformStamped.transform.translation.x = tvec[2];
    transformStamped.transform.translation.y = -tvec[0];
    transformStamped.transform.translation.z = -tvec[1];
    tf2::Quaternion q;
    q.setRPY(rpy[0], rpy[1], rpy[2]);
    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    msg.pose.pose.position.x = transformStamped.transform.translation.x;
    msg.pose.pose.position.y = transformStamped.transform.translation.y;
    msg.pose.pose.position.z = transformStamped.transform.translation.z;
    msg.pose.pose.orientation = transformStamped.transform.rotation;

    //ROS_INFO("marker T:[%f,%f,%f]\tR:[%f,%f,%f,%f]",
    //         tvec[0], tvec[1], tvec[2],
    //         q.x(), q.y(), q.z(), q.w());

    msg.header.stamp = ros::Time::now();
    pub.publish(msg);
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

    ros::Publisher rgb_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>
                                    (ar_frame + "/rgb_pose", 10);
    ros::Publisher depth_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>
                                    (ar_frame + "/depth_pose", 10);

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

    cv::Vec3d rgb_rpy, rgb_tvec;
    cv::Vec3d depth_rpy, depth_tvec;

    ros::Rate rate(30);

    while (ros::ok())
    {
        cv_bridge::CvImagePtr rgb_ptr, depth_ptr;
        rosimg.get_img(rgb_ptr);
        rosdepth.get_img(depth_ptr);
        if (rgb_ptr && depth_ptr)
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

                            rgb_tvec = tvecs[i];
                            rgb_rpy = getEuler(rvecs[i]);

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

                            double pt_c_x = (ptc.x - min_pt.x); //マーカー領域の中心位置x[px]
                            double pt_c_y = (ptc.y - min_pt.y); //マーカー領域の中心位置y[px]

                            for(int i = 0; i < sz.height; i++){ //画像y方向
                                for(int j = 0; j < sz.width; j++){ //画像x方向
                                    ushort depth = dmarker.at<ushort>(j, i); //z[mm]
                                    if(depth != 0){
                                        double _i = (double)(i - pt_c_x) / depthMatrix.at<double>(0, 0) * depth; //y[mm]
                                        double _j = (double)(j - pt_c_y) / depthMatrix.at<double>(1, 1) * depth; //x[mm]
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

                            cv::solve(M, U, V, DECOMP_LU); //LU分解にて平面の定数行列(V)を導出
                            //平面の式: z[mm] = V1 * x[mm] + V2 * y[mm] + V0

                            //マーカ中心部の推定距離を測定
                            double z_c = V.at<double>(0, 0);
                            //カメラパラメータより実距離のx,yを求める。
                            double x_c = ((ptc.x - depthMatrix.at<double>(0, 2)) / depthMatrix.at<double>(0, 0)) * z_c;
                            double y_c = ((ptc.y - depthMatrix.at<double>(1, 2)) / depthMatrix.at<double>(1, 1)) * z_c;

                            cv::Vec3d rpy, tvec;
                            tvec[0] = x_c / 1E3;
                            tvec[1] = y_c / 1E3;
                            tvec[2] = z_c / 1E3;

                            double pt_1c_x = pt[1].x - ptc.x; //中心と45度方向のマーカー角1との位置ベクトルx[px]
                            double pt_1c_y = pt[1].y - ptc.y; //中心と45度方向のマーカー角1との位置ベクトルy[px]

                            double rxy = std::atan2(pt_1c_y / sz.height, pt_1c_x / sz.width) + M_PI / 4.0;

                            // V1*x + V2*y - 1*z + V0 = 0
                            cv::Mat NV = cv::Mat::zeros(3, 1, CV_64F); //法線ベクトル
                            NV.at<double>(0) = V.at<double>(1, 0);
                            NV.at<double>(1) = V.at<double>(2, 0);
                            NV.at<double>(2) = -1.0;

                            double roll = atan2(NV.at<double>(2), NV.at<double>(0)) + M_PI/2;
                            double pitch = atan2(NV.at<double>(2), NV.at<double>(1)) - M_PI/2;
                            double yaw = -rxy - M_PI;
                            rpy[1] = roll;
                            rpy[2] = pitch;
                            rpy[0] = yaw;

                            depth_tvec = tvec;
                            depth_rpy = rpy;


                        }
                    }
                }
            }

            //cv::imshow("Aruco", imageCopy); //結果をウィンドウ(Aruco)に表示
        }
        ar_broadcast(rgb_pub, ar_frame, rgb_tvec, rgb_rpy);
        ar_broadcast(depth_pub, ar_frame_depth, depth_tvec, depth_rpy);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
