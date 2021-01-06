#include "ros/ros.h"

#include <iostream>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

class RosImageConverter
{
  const std::string _img_format;
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Subscriber cam_info_pub_;
  //image_transport::Publisher image_pub_;

  cv_bridge::CvImagePtr _cv_ptr;

  int height_, width_;
  cv::Mat K_;

public:
  RosImageConverter(const std::string img, const std::string camera_info, const std::string img_format)
    : it_(nh_), _img_format(img_format)
  {
    image_sub_ = it_.subscribe(img, 1,
      &RosImageConverter::imageCb, this);
    cam_info_pub_ = nh_.subscribe(camera_info, 1,
      &RosImageConverter::caminfoCb, this);
    //image_pub_ = it_.advertise("/image_converter/output_video", 1);
    K_ = cv::Mat::zeros(3, 3, CV_64F);
  }

  void get_img(cv_bridge::CvImagePtr &cv_ptr)
  {
    cv_ptr = _cv_ptr;
  }

  cv::Size get_size()
  {
    return cv::Size(width_, height_);
  }

  cv::Mat getK()
  {
    return K_;
  }

  void caminfoCb(const sensor_msgs::CameraInfo& msg)
  {
    height_ = msg.height;
    width_ = msg.width;
    K_.at<double>(0, 0) = msg.K[0];
    K_.at<double>(0, 2) = msg.K[2];
    K_.at<double>(1, 1) = msg.K[4];
    K_.at<double>(1, 2) = msg.K[5];
    K_.at<double>(2, 2) = msg.K[8];
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, _img_format);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    _cv_ptr = cv_ptr;
  }
};

