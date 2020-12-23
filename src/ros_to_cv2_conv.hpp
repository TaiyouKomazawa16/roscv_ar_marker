#include "ros/ros.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

class RosImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  //image_transport::Publisher image_pub_;

  cv_bridge::CvImagePtr _cv_ptr;

public:
  RosImageConverter()
    : it_(nh_)
  {
    image_sub_ = it_.subscribe("/camera/color/image_raw", 1,
      &RosImageConverter::imageCb, this);
    //image_pub_ = it_.advertise("/image_converter/output_video", 1);
  }

  void get_img(cv_bridge::CvImagePtr &cv_ptr)
  {
    cv_ptr = _cv_ptr;
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    _cv_ptr = cv_ptr;
  }
};

