// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               Livox@gmail.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <pcl/registration/gicp.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64MultiArray.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include <eigen3/Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/impl/instantiate.hpp>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <time.h>

#include <chrono>

// add jxf
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// BALM
#include <pcl/filters/voxel_grid.h>
#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <malloc.h>

#include <map>
// add jxf

#define PI 3.14159265

using namespace std;
// add jxf
typedef pcl::PointXYZI PointType; // 法向量信息没用到，就改为XYZI
// add jxf
//  typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

inline double rad2deg(double radians) { return radians * 180.0 / M_PI; }

inline double deg2rad(double degrees) { return degrees * M_PI / 180.0; }

extern const bool loopClosureEnableFlag = true;
extern const double mappingProcessInterval = 0.00; // 0.3

extern const float DISTANCE_SQ_THRESHOLD = 60.0;

extern const float scanPeriod = 0.1;
extern const int systemDelay = 0;
extern const int imuQueLength = 200;
extern const std::string imuTopic = "/imu/data";
// add jxf
extern const std::string odometryFrame = "camera_init"; // 坐标系要与前面几个一致 不然无效
extern const std::string lidarFrame = "body";
extern const bool useGpsElevation = false;
extern const float gpsCovThreshold = 2.0;
extern const float poseCovThreshold = 25.0; 

// add jxf

extern const float surroundingKeyframeSearchRadius = 50.0;
extern const int surroundingKeyframeSearchNum = 50;

extern const float gnorm = 9.805;

extern const float historyKeyframeSearchRadius = 2;
extern const int historyKeyframeSearchNum = 20;
extern const float historyKeyframeFitnessScore = 0.02; // jxf 0.1

extern const float globalMapVisualizationSearchRadius = 500.0;

// 各变量的首元素索引值
static constexpr unsigned int pos_ = 0;
static constexpr unsigned int vel_ = 3;
static constexpr unsigned int att_ = 6;
static constexpr unsigned int acc_ = 9;
static constexpr unsigned int gyr_ = 12;
static constexpr unsigned int gra_ = 15;

/*!@EARTH COEFFICIENTS */
extern const double G0 = gnorm;                // gravity
extern const double deg = M_PI / 180.0;        // degree
extern const double rad = 180.0 / M_PI;        // radian
extern const double dph = deg / 3600.0;        // degree per hour
extern const double dpsh = deg / sqrt(3600.0); // degree per square-root hour
extern const double mg = G0 / 1000.0;          // mili-gravity force
extern const double ug = mg / 1000.0;          // micro-gravity force
extern const double mgpsHz = mg / sqrt(1.0);   // mili-gravity force per second
extern const double ugpsHz = ug / sqrt(1.0);   // micro-gravity force per second
extern const double Re = 6378137.0;            ///< WGS84 Equatorial radius in meters
extern const double Rp = 6356752.31425;
extern const double Ef = 1.0 / 298.257223563;
extern const double Wie = 7.2921151467e-5;
extern const double Ee = 0.0818191908425;
extern const double EeEe = Ee * Ee;

string bagPath2;
double angleBetween = 5.0; // 平面匹配 角度阈值
// int indexs = 1;
struct smoothness_t
{
    float value;
    size_t ind;
};

// add jxf
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}
void savePcd(pcl::PointCloud<PointType>::Ptr saveCloud, string fileName)
{
    pcl::PCDWriter writer;
    writer.write<PointType>(string(bagPath2  + "/pointclouds/PCD/scans_")  + fileName + ".pcd", *saveCloud, true);
}
//-----------------------------------------平面相关判断函数----------------------------//
double calculateAngles(gtsam::Vector4 plane1) // 计算法向量with ground夹角
{
    double angle = acos(plane1(2)) * 180 / M_PI;
    return fabs(angle);
}

double calculateAngles(gtsam::Vector4 plane1, gtsam::Vector4 plane2) // 计算法向量夹角和平面距离
{
    double angle = acos(plane1(0) * plane2(0) + plane1(1) * plane2(1) + plane1(2) * plane2(2)) * 180 / M_PI; // 与垂直平面夹角
    return fabs(angle);
}

double calculateAngles(const gtsam::Vector4 plane1, const gtsam::Vector4 plane2, double &d) // 计算法向量夹角
{
    double angle = acos(plane1(0) * plane2(0) + plane1(1) * plane2(1) + plane1(2) * plane2(2)) * 180 / M_PI; // 与垂直平面夹角
    d = fabs(plane1(3) - plane2(3));

    return fabs(angle);
}

pcl::PointCloud<PointType>::Ptr readPcd(string i)
{
string filename =string(bagPath2  + "/pointclouds/PCD/scans_")   + i+ ".pcd";//B filename
pcl::PointCloud<PointType> pl_tem;
pcl::PointCloud<PointType>::Ptr pl_new(new pcl::PointCloud<PointType>());
pcl::io::loadPCDFile(filename, pl_tem);
for (pcl::PointXYZI &pp : pl_tem.points)
{
    PointType ap;
    ap.x = pp.x;
    ap.y = pp.y;
    ap.z = pp.z;
    ap.intensity = pp.intensity;
    pl_new->push_back(ap);
}
pl_new->header = pl_tem.header;
pl_new->width = pl_tem.width;
pl_new->height = pl_tem.height;
pl_new->is_dense = pl_tem.is_dense;
pl_new->sensor_origin_ = pl_tem.sensor_origin_;
pl_new->sensor_orientation_ = pl_tem.sensor_orientation_;
pl_new->points.resize(pl_tem.points.size());

return pl_new;
}
double calculateDistance(Eigen::Vector4f p1, Eigen::Vector4f p2)
{
    double distance = sqrt(pow((p1(0) - p2(0)), 2) + pow((p1(1) - p2(1)), 2) + pow((p1(2) - p2(2)), 2));
    return distance;
}
double calculateDistance(gtsam::Point3 p1, gtsam::Point3 p2)
{
    double distance = sqrt(pow((p1(0) - p2(0)), 2) + pow((p1(1) - p2(1)), 2) + pow((p1(2) - p2(2)), 2));
    return distance;
}

bool cmp(const pair<int, int> &p1, const pair<int, int> &p2)
{
    return p1.second > p2.second;
}

vector<pair<int, int>> sortMap(map<int, int> mp) // 对map容器按value值降序排列
{
    vector<pair<int, int>> vpr;
    for (map<int, int>::iterator it = mp.begin(); it != mp.end(); it++)
    {
        vpr.push_back(make_pair(it->first, it->second));
    }
    sort(vpr.begin(), vpr.end(), cmp);
    return vpr;
}

double uniformRand(double lowerBndr, double upperBndr)
{
    return lowerBndr + ((double)std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}



//--------------------------------点云滤波------------------------------------------//

struct PointXYZIQT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float qx;
    float qy;
    float qz;
    float qw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIQT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, qx, qx)(float, qy, qy)(float, qz, qz)(float, qw, qw)(double, time, time))
typedef PointXYZIQT PointTypePose;
