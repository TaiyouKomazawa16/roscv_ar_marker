#include "ros/ros.h"

#include <vector>
#include <iostream>

#include <bits/stdc++.h> //円周率用
//roscvイメージング用パッケージ
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
//ros姿勢管理用パッケージ
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <opencv2/calib3d.hpp>   // 回転ベクトル↔回転行列の変換に使用
#include <eigen3/Eigen/Geometry> // 四元数の計算に使用

//マーカ追跡パッケージ
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
//画像サブスクライバ
#include "ros_to_cv2_conv.hpp"

#define SAMPLING_NUM 20 //共分散行列のサンプリング数

#define LPF_K 0.25 //法線ベクトル導出過程におけるローパスフィルタのパラメータ

using namespace std;
using namespace cv;

const int target_id = 6;
std::string ar_frame = "ar_mark_" + std::to_string(target_id);
std::string ar_frame_depth = "ar_mark_" + std::to_string(target_id) + "_depth";

void setRosCov(cv::Mat src, double dst[36]) //6x6共分散行列をros形式にする
{
    for(int i = 0; i < 6; i++){
        for(int j = 0; j < 6; j++){
            dst[6*i + j] = src.at<double>(i, j);
        }
    }
}

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

void ar_broadcast(ros::Publisher &pub, std::string ar_frame, Vec3d tvec, tf2::Quaternion q, double cov[36])
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

    transformStamped.transform.rotation.x = q.x();
    transformStamped.transform.rotation.y = q.y();
    transformStamped.transform.rotation.z = q.z();
    transformStamped.transform.rotation.w = q.w();

    msg.pose.pose.position.x = transformStamped.transform.translation.x;
    msg.pose.pose.position.y = transformStamped.transform.translation.y;
    msg.pose.pose.position.z = transformStamped.transform.translation.z;
    msg.pose.pose.orientation = transformStamped.transform.rotation;

    for(int i = 0; i < 36; i++)
        msg.pose.covariance[i] = cov[i];

    /*ROS_INFO("%s:\nT:[%.3f,%.3f,%.3f] \nR:[%f,%f,%f,%f]",
             ar_frame.c_str(), tvec[0], tvec[1], tvec[2],
             q.x(), q.y(), q.z(), q.w());
*/
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

    cv::Vec3d rgb_rpy, rgb_tvec, rgb_rvec;
    cv::Vec3d depth_rpy, depth_tvec;

    tf2::Quaternion rgb_quat, depth_quat;

    //平面の法線ベクトル
    cv::Mat V = cv::Mat::zeros(3,1, CV_64F);

    //６軸姿勢ベクトル集合 ROW:[x,y,z,r,p,y], COL:[サンプリング数]
    cv::Mat rgb_poses = cv::Mat::zeros(6, SAMPLING_NUM, CV_64F);
    cv::Mat depth_poses = cv::Mat::zeros(6, SAMPLING_NUM, CV_64F);
    int poses_head = 0;
    //分散共分散行列
    double rgb_cov[36] = {};
    double depth_cov[36] = {};

    ros::Rate rate(30); //hz

    while (ros::ok()){
        cv_bridge::CvImagePtr rgb_ptr, depth_ptr;
        rosimg.get_img(rgb_ptr);
        rosdepth.get_img(depth_ptr);

        if (!rgb_ptr || !depth_ptr){ //画像データなし
            rgb_quat = getQuaternion(rgb_rvec);
            ar_broadcast(rgb_pub, ar_frame, rgb_tvec, rgb_quat, rgb_cov);
            depth_quat.setRPY(depth_rpy[0], depth_rpy[1], depth_rpy[2]);
            ar_broadcast(depth_pub, ar_frame_depth, depth_tvec, depth_quat, depth_cov);
            ros::spinOnce();
            rate.sleep();
            continue;
        }

        // マーカ検出
        aruco::detectMarkers(rgb_ptr->image, dictionary, corners, ids, detectorParams, rejected);
        if (ids.size() <= 3){ //ダイヤモンドを構成するマーカ(4つ必要)が不足
            rgb_quat = getQuaternion(rgb_rvec);
            ar_broadcast(rgb_pub, ar_frame, rgb_tvec, rgb_quat, rgb_cov);
            depth_quat.setRPY(depth_rpy[0], depth_rpy[1], depth_rpy[2]);
            ar_broadcast(depth_pub, ar_frame_depth, depth_tvec, depth_quat, depth_cov);
            ros::spinOnce();
            rate.sleep();
            continue;
        }

        // ダイヤモンドマーカ検出
        aruco::detectCharucoDiamond(rgb_ptr->image, corners, ids, squareLength / markerLength, diamondCorners, diamondIds, imgMatrix, distCoeffs);
        if (diamondIds.size() <= 0) { //ダイヤモンドマーカ未検出
            rgb_quat = getQuaternion(rgb_rvec);
            ar_broadcast(rgb_pub, ar_frame, rgb_tvec, rgb_quat, rgb_cov);
            depth_quat.setRPY(depth_rpy[0], depth_rpy[1], depth_rpy[2]);
            ar_broadcast(depth_pub, ar_frame_depth, depth_tvec, depth_quat, depth_cov);
            ros::spinOnce();
            rate.sleep();
            continue;
        }

        // ダイヤモンドマーカ測定
        aruco::estimatePoseSingleMarkers(diamondCorners, squareLength, imgMatrix, distCoeffs, rvecs, tvecs);
        
        for (unsigned int i = 0; i < diamondIds.size(); i++){
            cv::Vec4i id = diamondIds[i];
            if (target_id == id[0]+id[1]+id[2]+id[3]){
                /* ---rgbカメラ系の処理--- */
                rgb_tvec = tvecs[i];
                rgb_rvec = rvecs[i];
                tf2::Matrix3x3(rgb_quat).getRPY(rgb_rpy[0], rgb_rpy[1], rgb_rpy[2]);
                
                /* ---depthカメラ系の処理--- */
                vector<cv::Point2f> dcor = diamondCorners[i];
                cv::Point pt[4] = {dcor[0], dcor[1], dcor[2], dcor[3]};
                //マーカの重心座標の検出
                double s1 = ((pt[3].x-pt[1].x)*(pt[0].y-pt[1].y)-(pt[3].y-pt[1].y)*(pt[0].x-pt[1].x)) / 2.0;
                double s2 = ((pt[3].x-pt[1].x)*(pt[1].y-pt[2].y)-(pt[3].y-pt[1].y)*(pt[1].x-pt[2].x)) / 2.0;
                cv::Point ptc; //重心座標
                ptc.x = pt[0].x + (pt[2].x - pt[0].x) * s1 / (s1 + s2);
                ptc.y = pt[0].y + (pt[2].y - pt[0].y) * s1 / (s1 + s2);
                //マーカのデプス領域の切り出し
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
                cv::Mat raw_V = cv::Mat::zeros(3,1, CV_64F);
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
                //平面の式: z[mm] = V1 * x[mm] + V2 * y[mm] + V0
                cv::solve(M, U, raw_V, DECOMP_LU); //LU分解にて平面の定数行列(V)を導出
                
                //法線ベクトルの高周波数ノイズをカット
                for(int i = 0; i < 3; i++)
                    V.at<double>(i,0) += LPF_K * (raw_V.at<double>(i,0) - V.at<double>(i,0));

                cv::Vec3d rpy, tvec;

                //マーカ中心部の推定距離を測定
                double z_c = V.at<double>(0, 0);
                //カメラパラメータより実距離のx,yを求める。
                double x_c = ((ptc.x - depthMatrix.at<double>(0, 2)) / depthMatrix.at<double>(0, 0)) * z_c;
                double y_c = ((ptc.y - depthMatrix.at<double>(1, 2)) / depthMatrix.at<double>(1, 1)) * z_c;

                //並進成分のデータを格納 単位換算[mm]->[m]
                tvec[0] = x_c / 1E3;
                tvec[1] = y_c / 1E3;
                tvec[2] = z_c / 1E3;

                //対角線の大体の距離を導出(奥行きはz_cで代用)
                double x_13 = ((pt[1].x-pt[3].x) / depthMatrix.at<double>(0, 0)) * z_c;
                double y_13 = ((pt[1].y-pt[3].y) / depthMatrix.at<double>(1, 1)) * z_c;
                double x_20 = ((pt[2].x-pt[0].x) / depthMatrix.at<double>(0, 0)) * z_c;
                double y_20 = ((pt[2].y-pt[0].y) / depthMatrix.at<double>(1, 1)) * z_c;

                //roll角を算出(2つ求めたのは歪み対策のため)
                double rxy13 = -std::atan2(y_13, x_13) + M_PI * 3.0 / 4.0;
                double rxy02 = std::atan2(x_20, y_20) + M_PI * 3.0 / 4.0;

                if(fabs(rxy13 - rxy02) > M_PI/2)
                    rxy13 = rxy02;

                // V1*x + V2*y - 1*z + V0 = 0
                cv::Mat NV = cv::Mat::zeros(3, 1, CV_64F); //法線ベクトル
                NV.at<double>(0) = V.at<double>(1, 0);
                NV.at<double>(1) = V.at<double>(2, 0);
                NV.at<double>(2) = -1.0;

                rpy[0] = (rxy13 + rxy02) / 2.0; //x軸回転(roll)
                rpy[1] = atan2(NV.at<double>(2), NV.at<double>(0)) + M_PI/2; //y軸回転(pitch)
                rpy[2] = atan2(NV.at<double>(2), NV.at<double>(1)) - M_PI/2; //z軸回転(yaw)

                ROS_INFO("%f %f", NV.at<double>(0), NV.at<double>(1));

                depth_tvec = tvec;
                depth_rpy = rpy;

                /*共分散の導出部*/
                cv::Mat rgb_cov_ = cv::Mat::eye(6, 6, CV_64F);
                cv::Mat depth_cov_ = cv::Mat::eye(6, 6, CV_64F);

                for(int i = 0; i < 3; i++) //rgb_xyz
                    rgb_poses.at<double>(i, poses_head) = rgb_tvec[i];
                for(int i = 0; i < 3; i++) //rgb_rpy
                    rgb_poses.at<double>(i+3, poses_head) = rgb_rpy[i];
                for(int i = 0; i < 3; i++) //depth_xyz
                    depth_poses.at<double>(i, poses_head) = depth_tvec[i];
                for(int i = 0; i < 3; i++) //depth_rpy
                    depth_poses.at<double>(i+3, poses_head) = depth_rpy[i];

                cv::Mat rgb_means = cv::Mat::zeros(6, 1, CV_64F);
                cv::Mat depth_means = cv::Mat::zeros(6, 1, CV_64F);

                //分散共分散行列を計算(不偏分散)
                cv::calcCovarMatrix(rgb_poses, rgb_cov_, rgb_means, 
                                            CV_COVAR_NORMAL | CV_COVAR_COLS);
                rgb_cov_ /= (rgb_poses.rows - 1);
                cv::calcCovarMatrix(depth_poses, depth_cov_, depth_means, 
                                            CV_COVAR_NORMAL | CV_COVAR_COLS);
                depth_cov_ /= (depth_poses.rows - 1);

                setRosCov(rgb_cov_, rgb_cov);
                setRosCov(depth_cov_, depth_cov);

                poses_head += (poses_head < SAMPLING_NUM-1) ? 1 : -poses_head;
            }
        }
        rgb_quat = getQuaternion(rgb_rvec);
        ar_broadcast(rgb_pub, ar_frame, rgb_tvec, rgb_quat, rgb_cov);
        depth_quat.setRPY(depth_rpy[0], depth_rpy[1], depth_rpy[2]);
        ar_broadcast(depth_pub, ar_frame_depth, depth_tvec, depth_quat, depth_cov);
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
