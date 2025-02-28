#include <iostream>
#include "livox_backend/common.h"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/PoseRotationPrior.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/navigation/GPSFactor.h>
#include "std_msgs/String.h"
#include <gtsam/nonlinear/ISAM2.h>
#include <csignal>

using namespace gtsam;

class globalOptimization
{
private:
    /*************因子图优化相关参数*************/
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;
    /*****************************************/

    /*************ROS通讯参数*************/
    ros::NodeHandle nh;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;
    ros::Subscriber subLoopParam;
    ros::Subscriber subLaserCloud;
    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubFinalCloud;
    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubLoopConstraintEdge;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform aftMappedOdomTrans;
    /*****************************************/

    /*************点云配准相关变量*************/
    pcl::PointCloud<PointType>::Ptr laserCloudLast;
    std::vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;

    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistorySurfKeyFrameCloudDS;

    std::deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    std::vector<int> surroundingExistingKeyPosesID;
    std::deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;

    pcl::VoxelGrid<PointType> downSizeFilter;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;
    /*****************************************/

    /*************位姿图优化相关变量*************/
    int latestFrameID;
    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;
    Eigen::Quaternionf previousRobotAttPoint;
    Eigen::Quaternionf currentRobotAttPoint;

    // PointType(pcl::PointXYZI)的XYZI分别保存3个方向上的平移和一个索引(cloudKeyPoses3D->points.size())
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3DforLoop;
    // PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    Eigen::Quaternionf q_transformSum;
    Eigen::Vector3f v_transformSum;

    Eigen::Quaternionf q_transformIncre;
    Eigen::Vector3f v_transformIncre;

    Eigen::Quaternionf q_transformTobeMapped;
    Eigen::Vector3f v_transformTobeMapped;

    Eigen::Quaternionf q_transformBefMapped;
    Eigen::Vector3f v_transformBefMapped;

    Eigen::Quaternionf q_transformAftMapped;
    Eigen::Vector3f v_transformAftMapped;

    Eigen::Quaternionf q_transformLast;
    Eigen::Vector3f v_transformLast;

    Eigen::Quaternionf Imurpy;
    Eigen::Quaternionf ImuKeyFrame;
    Eigen::Quaternionf ImuKeyFrame2;
    Eigen::Quaternionf ImuLast;

    int ImuLastTime;
    /*****************************************/

    /*************线程控制相关参数*************/
    double timeLaserOdometry;
    double timeLaserCloudLast;
    bool newLaserOdometry;
    bool newLaserCloudLast;
    bool damn;
    std::mutex mtx;
    double timeLastProcessing;
    /*****************************************/

    /*************回环检测相关参数*************/
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
    bool potentialLoopFlag;
    double timeSaveFirstCurrentScanForLoopClosure;
    int closestHistoryFrameID;
    int latestFrameIDLoopCloure;
    bool aLoopIsClosed;
    map<int, int> loopIndexContainer; // from new to old
    /*****************************************/

    /*************全局地图可视化相关参数*************/
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;
    /*****************************************/

    /*************路径保存*************/
    string filePath;
    fstream myFiles;
    bool issavePcd;
    int lastCloudFrame = 0;
    fstream odometerfile;
    /*****************************************/
    vector<string> paramAccept;
    string BagPath;
    float loopInterval;
    float loopDistance;
    float icpMatchScoreThre;
    bool isLoop;
     /******************GPS***********************/
     std::deque<nav_msgs::Odometry> gnss_buffer;
     ros::Subscriber subGPS;
     Eigen::MatrixXd poseCovariance;

public:
    globalOptimization() : nh("~")
    {
        // 用于闭环图优化的参数设置，使用gtsam库
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        parameters.factorization = ISAM2Params::QR;
        isam = new ISAM2(parameters);

        // 发布
        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);
        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);

        pubFinalCloud = nh.advertise<sensor_msgs::PointCloud2>("/final_cloud", 2);
        
        // 订阅
        // subGPS   = nh.subscribe<nav_msgs::Odometry> ("/gps/fix"  , 200, &globalOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        subLoopParam = nh.subscribe<std_msgs::String>("/loopParamter", 5, &globalOptimization::loopParamHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/Odometry", 5, &globalOptimization::laserOdometryHandler, this);
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered2", 2, &globalOptimization::laserCloudLastHandler, this);
        odomAftMapped.header.frame_id = odometryFrame;
        odomAftMapped.child_frame_id = "body";

        aftMappedTrans.frame_id_ = odometryFrame;
        aftMappedTrans.child_frame_id_ = lidarFrame;

        aftMappedOdomTrans.frame_id_ = odometryFrame;
        aftMappedOdomTrans.child_frame_id_ = lidarFrame;

        allocateMemory();
    }
    ~globalOptimization()
    {
    }

    //added by gjf @2023.9.7
    void updatePcdAndPose()
    {
        // odometerfile.close();
        //     //读取每一帧点云，和位姿 保存
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        string filePath = bagPath2+"/odometer/odometer.txt";
        fstream myFiles;
        myFiles.open(filePath, ios::out);
        cout.setf(ios::fixed,ios::floatfield);
        cout.precision(20);
        if (issavePcd)
        {
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
            {
                // *globalSurfCloud += *transformPointCloud(readPcd(to_string(i)), &cloudKeyPoses6D->points[i]);
                //  *globalSurfCloud += *readPcd(to_string(i));
                cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
                string names =  to_string(i );
                savePcd(transformPointCloud(readPcd(to_string(i)), &cloudKeyPoses6D->points[i]), names);


                // pcl::PCDWriter writer;
                // writer.write<PointType>(string("/home/ubuntu18/ws-fast-lio-work/src/FAST_LIO/PCD(LOOP)/") + names + ".pcd", *transformPointCloud(readPcd(to_string(i)), &cloudKeyPoses6D->points[i]), true);
                Eigen::Quaterniond quaternion(cloudKeyPoses6D->points[i].qw,cloudKeyPoses6D->points[i].qx,cloudKeyPoses6D->points[i].qy,cloudKeyPoses6D->points[i].qz);
                Eigen::Vector3d eulerAngle=quaternion.matrix().eulerAngles(2,1,0);

                myFiles << i;
                myFiles << " ";
                myFiles << i;
                myFiles << " ";
                myFiles <<std::setprecision(20)<<cloudKeyPoses6D->points[i].time;
                myFiles << " ";
                myFiles <<std::setprecision(6)<< cloudKeyPoses6D->points[i].x;
                myFiles << " ";
                myFiles << cloudKeyPoses6D->points[i].y;
                myFiles << " ";
                myFiles << cloudKeyPoses6D->points[i].z;
                myFiles << " ";
                myFiles << eulerAngle(2);
                myFiles << " ";
                myFiles <<eulerAngle(1);
                myFiles << " ";
                myFiles <<eulerAngle(0);
                myFiles << "\n";
               
            }
             myFiles.close();
            // string names = "scans_all";
            // savePcd(globalSurfCloud, names);
        }
    }
    //added by gjf @2023.9.7

    void allocateMemory()
    {
        
        
        // filePath = "/home/ubuntu18/ws-fast-lio-work/src/FAST_LIO/PCD/xyz.txt";
        // odometerfile.open("/home/ubuntu18/ws-fast-lio-work/src/FAST_LIO/PCD/odomter.txt", ios::out);
        issavePcd = true;

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        cloudKeyPoses3DforLoop.reset(new pcl::PointCloud<PointType>());

        nearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        laserCloudLast.reset(new pcl::PointCloud<PointType>());

        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        downSizeFilterGlobalMapKeyPoses.setLeafSize(0.1, 0.1, 0.1);
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.1, 0.1, 0.1);

        downSizeFilter.setLeafSize(0.1, 0.1, 0.1);
        downSizeFilterSurf.setLeafSize(0.01, 0.01, 0.01);
        downSizeFilterHistoryKeyFrames.setLeafSize(0.1, 0.1, 0.1);

        q_transformSum.setIdentity();
        v_transformSum.setZero();
        q_transformIncre.setIdentity();
        v_transformIncre.setZero();
        q_transformBefMapped.setIdentity();
        v_transformBefMapped.setZero();
        q_transformTobeMapped.setIdentity();
        v_transformTobeMapped.setZero();
        q_transformAftMapped.setIdentity();
        v_transformAftMapped.setZero();
        q_transformLast.setIdentity();
        v_transformLast.setZero();
        Imurpy.setIdentity();
        ImuKeyFrame.setIdentity();
        ImuKeyFrame2.setIdentity();
        ImuLast.setIdentity();

        ImuLastTime = 0;

        timeLastProcessing = -1;

        // 初始化　先验噪声＼里程计噪声
        gtsam::Vector Vector6(6);
        gtsam::Vector OdometryVector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        OdometryVector6 << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        newLaserOdometry = false;
        newLaserCloudLast = false;

        damn = true;

        latestFrameID = 0;
    }
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gnss_buffer.push_back(*gpsMsg);
    }

    void loopParamHandler(const std_msgs::String::ConstPtr &loopParam)
    {
        if(paramAccept.size()<6)
        {
            string loopP = loopParam->data.c_str();

            paramAccept.push_back(loopP);
            if(paramAccept.size()==5)
            {

                        isLoop =bool(stoi(paramAccept[0])==1);
                        icpMatchScoreThre = stof(paramAccept[1]);
                        loopDistance = stof(paramAccept[2]);
                        loopInterval = stof(paramAccept[3]);
                        BagPath = paramAccept[4];
                        cout<<isLoop<<"  "<<icpMatchScoreThre<<" "<<loopDistance<<"  "<<loopInterval<< "  "<<BagPath<<endl;
            }
        }
    }
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
    {
        timeLaserOdometry = laserOdometry->header.stamp.toSec();
        newLaserOdometry = true;

        v_transformSum.x() = laserOdometry->pose.pose.position.x;
        v_transformSum.y() = laserOdometry->pose.pose.position.y;
        v_transformSum.z() = laserOdometry->pose.pose.position.z;
        q_transformSum.x() = laserOdometry->pose.pose.orientation.x;
        q_transformSum.y() = laserOdometry->pose.pose.orientation.y;
        q_transformSum.z() = laserOdometry->pose.pose.orientation.z;
        q_transformSum.w() = laserOdometry->pose.pose.orientation.w;


        // myFiles.open(filePath,ios::out | ios::app);
        // myFiles<< v_transformSum.x();
        // myFiles<<" ";
        // myFiles<< v_transformSum.y();
        // myFiles<<" ";
        // myFiles<< v_transformSum.z();
        // myFiles<<"\n";
        // myFiles.close();
    }
    void laserCloudLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        timeLaserCloudLast = msg->header.stamp.toSec();
        newLaserCloudLast = true;

        laserCloudLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudLast);
    }

    void transformAssociateToMap()
    {
        v_transformIncre = v_transformSum - v_transformBefMapped;
        q_transformIncre = q_transformSum * q_transformBefMapped.inverse();
        v_transformIncre = q_transformBefMapped.inverse() * v_transformIncre;

        // 遵守先平移后旋转的规矩
        q_transformTobeMapped = q_transformIncre * q_transformAftMapped;

        v_transformTobeMapped = v_transformAftMapped + q_transformAftMapped * v_transformIncre;
        q_transformTobeMapped.normalize();

        v_transformAftMapped = v_transformTobeMapped;
        q_transformAftMapped = q_transformTobeMapped;

        q_transformBefMapped = q_transformSum;
        v_transformBefMapped = v_transformSum;

        ImuKeyFrame2 = Imurpy;
    }

    void saveKeyFramesAndFactor()
    {
        // 取优化后的位姿transformAftMapped[]
        // 储存到currentRobotPosPoint
        currentRobotPosPoint.x = v_transformTobeMapped.x();
        currentRobotPosPoint.y = v_transformTobeMapped.y();
        currentRobotPosPoint.z = v_transformTobeMapped.z();

        // 如果两次优化之间欧式距离<0.3，则不保存，不作为关键帧
        bool saveThisKeyFrame = true;
        // if (sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x) * (previousRobotPosPoint.x - currentRobotPosPoint.x) + (previousRobotPosPoint.y - currentRobotPosPoint.y) * (previousRobotPosPoint.y - currentRobotPosPoint.y) + (previousRobotPosPoint.z - currentRobotPosPoint.z) * (previousRobotPosPoint.z - currentRobotPosPoint.z)) < 0.3)
        // {
        //     saveThisKeyFrame = false;
        // }
        //add jxf
        // if(abs(double(1688823732.028011) - timeLaserOdometry) < 1.5 || abs(double(1688823732.028011 + 848.173621 ) - timeLaserOdometry) < 1.5)// stright :1688207073.584190 + 778  L:1688208092.309725+850.427283
        // {
        //     cout<<"gg"<<endl;
        //     saveThisKeyFrame=true;
        // }
        //add jxf

        // 非关键帧 并且 非空，直接返回
        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
            return;

        previousRobotPosPoint = currentRobotPosPoint;
        previousRobotAttPoint = q_transformAftMapped;

        // 如果还没有关键帧
        if (cloudKeyPoses3D->points.empty())
        {
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3(q_transformTobeMapped.cast<double>()), Point3(v_transformTobeMapped.cast<double>())), priorNoise));
            initialEstimate.insert(0, Pose3(Rot3(q_transformTobeMapped.cast<double>()),
                                            Point3(v_transformTobeMapped.cast<double>())));
            q_transformLast = q_transformTobeMapped;
            v_transformLast = v_transformTobeMapped;
        }
        else
        {
            // 非第一帧，添加边
            gtsam::Pose3 poseFrom = Pose3(Rot3(q_transformLast.cast<double>()),
                                          Point3(v_transformLast.cast<double>()));
            gtsam::Pose3 poseTo = Pose3(Rot3(q_transformTobeMapped.cast<double>()),
                                        Point3(v_transformTobeMapped.cast<double>()));

            // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size() - 1,
                                                cloudKeyPoses3D->points.size(),
                                                poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), poseTo);
        }
//add jxf
        //addGPSFactor();
//add jxf

        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;
        PointType thisPose3DforLoop;

        // Compute an estimate from the incomplete linear delta computed during the last update.
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

        Eigen::Vector3d v_latestEstimate = latestEstimate.translation();
        Eigen::Quaterniond q_latestEstimate = latestEstimate.rotation().toQuaternion();
        // GTSAM优化过后的旋转务必进行归一化，否则会计算异常
        q_latestEstimate.normalize();
        // poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);
        // std::cout<<"cloudKeyPoses3D->points = "<<cloudKeyPoses3D->points.size()<<std::endl;
        // std::cout<<"timeLaserOdometry = "<<timeLaserOdometry<<std::endl;

        // 取优化结果
        // 又回到相机坐标系c系以及导航坐标系n'
        thisPose3D.x = v_latestEstimate.x();
        thisPose3D.y = v_latestEstimate.y();
        thisPose3D.z = v_latestEstimate.z();
        thisPose3D.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity;
        thisPose6D.qx = q_latestEstimate.x();
        thisPose6D.qy = q_latestEstimate.y();
        thisPose6D.qz = q_latestEstimate.z();
        thisPose6D.qw = q_latestEstimate.w();
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);
        // add jxf
        ImuKeyFrame = ImuKeyFrame2;
        // add jxf
        thisPose3DforLoop = thisPose3D;
        // thisPose3DforLoop.z = 0;
        cloudKeyPoses3DforLoop->push_back(thisPose3DforLoop);

        // 把新帧作为keypose
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        // pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame2(new pcl::PointCloud<PointType>());
        
        // add jxf
        if (issavePcd )
        {
           string names = to_string(surfCloudKeyFrames.size());
            savePcd(laserCloudLast, names);
        }

        // add jxf
        // 更新transformAftMapped[]
        if (cloudKeyPoses3D->points.size() > 1)
        {
            q_transformAftMapped = q_latestEstimate.cast<float>();
            v_transformAftMapped = v_latestEstimate.cast<float>();

            q_transformLast = q_transformAftMapped;
            v_transformLast = v_transformAftMapped;

            q_transformTobeMapped = q_transformAftMapped;
            v_transformTobeMapped = v_transformAftMapped;
        }

        downSizeFilter.setInputCloud(laserCloudLast);
        downSizeFilter.filter(*laserCloudLast);
        pcl::copyPointCloud(*laserCloudLast, *thisSurfKeyFrame);

        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    }

    void correctPoses()
    {
        if (aLoopIsClosed == true)
        {
            recentSurfCloudKeyFrames.clear();

            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses3DforLoop->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3DforLoop->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3DforLoop->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;

                Eigen::Quaternionf q_rotaion = isamCurrentEstimate.at<Pose3>(i).rotation().toQuaternion().normalized().cast<float>();
                //
                cloudKeyPoses6D->points[i].qw = q_rotaion.w();
                cloudKeyPoses6D->points[i].qx = q_rotaion.x();
                cloudKeyPoses6D->points[i].qy = q_rotaion.y();
                cloudKeyPoses6D->points[i].qz = q_rotaion.z();
            }

            aLoopIsClosed = false;
        }
    }

    void publishTF()
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = q_transformAftMapped.x();
        odomAftMapped.pose.pose.orientation.y = q_transformAftMapped.y();
        odomAftMapped.pose.pose.orientation.z = q_transformAftMapped.z();
        odomAftMapped.pose.pose.orientation.w = q_transformAftMapped.w();
        odomAftMapped.pose.pose.position.x = v_transformAftMapped.x();
        odomAftMapped.pose.pose.position.y = v_transformAftMapped.y();
        odomAftMapped.pose.pose.position.z = v_transformAftMapped.z();
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(q_transformAftMapped.x(),
                                                  q_transformAftMapped.y(),
                                                  q_transformAftMapped.z(),
                                                  q_transformAftMapped.w()));
        aftMappedTrans.setOrigin(tf::Vector3(v_transformAftMapped.x(),
                                             v_transformAftMapped.y(),
                                             v_transformAftMapped.z()));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    void publishFinalCloud()
    {

        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*readPcd(string("final")), cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = odometryFrame;

        pubFinalCloud.publish(cloudMsgTemp);
    }

    void publishKeyPosesAndFrames()
    {

        // 发布关键帧位置
        if (pubKeyPoses.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = odometryFrame;
            pubKeyPoses.publish(cloudMsgTemp);
        }

        // if (pubLaserCloudSurround.getNumSubscribers() != 0){
        //     sensor_msgs::PointCloud2 cloudMsgTemp;
        //     pcl::toROSMsg(*laserCloudLast, cloudMsgTemp);
        //     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        //     cloudMsgTemp.header.frame_id = "/body";
        //     pubLaserCloudSurround.publish(cloudMsgTemp);
        // }
    }

    void run()
    {
        if (newLaserOdometry && newLaserCloudLast &&
            std::abs(timeLaserCloudLast - timeLaserOdometry) < 0.005)
        {

            bagPath2=BagPath;

            // 　数据接收标志位清空
            newLaserCloudLast = false;
            newLaserOdometry = false;

            std::lock_guard<std::mutex> lock(mtx);

            // 确保每次全局优化的时间时间间隔不能太快，大于0.3s
            double time_offset = timeLaserOdometry - timeLastProcessing;
            if ( time_offset > mappingProcessInterval)
            {
                // 储存里程计数据时间戳

                timeLastProcessing = timeLaserOdometry;

                // 将坐标转移到世界坐标系下,得到可用于建图的Lidar坐标，即修改了transformTobeMapped的值
                transformAssociateToMap();

                // 关键帧判断，gtsam优化
                saveKeyFramesAndFactor();

                correctPoses();

                publishTF();

                publishKeyPosesAndFrames();


                // clearCloud();
            }
        }
    }
//add jxf
void addGPSFactor()
{
    if (gnss_buffer.empty())
        return;
    // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
    if (cloudKeyPoses3D->points.empty())
        return;
    else
    {
        if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
            return;
    }
    // 位姿协方差很小，没必要加入GPS数据进行校正
    if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
        return;
    static PointType lastGPSPoint;      // 最新的gps数据
    while (!gnss_buffer.empty())
    {
        // 删除当前帧0.2s之前的里程计
        if (gnss_buffer.front().header.stamp.toSec() < timeLastProcessing - 0.05)
        {
            gnss_buffer.pop_front();
        }
        // 超过当前帧0.2s之后，退出
        else if (gnss_buffer.front().header.stamp.toSec() > timeLastProcessing + 0.05)
        {
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();
            // GPS噪声协方差太大，不能用
            float noise_x = thisGPS.pose.covariance[0];         //  x 方向的协方差
            float noise_y = thisGPS.pose.covariance[7];
            float noise_z = thisGPS.pose.covariance[14];      //   z(高层)方向的协方差
            if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                continue;
            // GPS里程计位置
            float gps_x = thisGPS.pose.pose.position.x;
            float gps_y = thisGPS.pose.pose.position.y;
            float gps_z = thisGPS.pose.pose.position.z;
            if (!useGpsElevation)           //  是否使用gps的高度
            {
                gps_z = v_transformTobeMapped[2];
                noise_z = 0.01;
            }

            // (0,0,0)无效数据
            if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                continue;
            // 每隔5m添加一个GPS里程计
            PointType curGPSPoint;
            curGPSPoint.x = gps_x;
            curGPSPoint.y = gps_y;
            curGPSPoint.z = gps_z;
            if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                continue;
            else
                lastGPSPoint = curGPSPoint;
            // 添加GPS因子
            gtsam::Vector Vector3(3);
            Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
            gtsam::noiseModel::Diagonal::shared_ptr gps_noise = gtsam::noiseModel::Diagonal::Variances(Vector3);
            gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
            gtSAMgraph.add(gps_factor);
            // aLoopIsClosed = true;
            ROS_INFO("GPS Factor Added");
            break;
        }
    }
}
//add jxf
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn,
                                                        const PointTypePose *const tIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        PointType pointTo;

        Eigen::Quaternionf q_tIn(tIn->qw, tIn->qx, tIn->qy, tIn->qz);
        Eigen::Vector3f v_tIn(tIn->x, tIn->y, tIn->z);

        for (int i = 0; i < cloudIn->size(); i++)
        {
            pointTo = cloudIn->points[i];
            Eigen::Vector3f point(cloudIn->points[i].x,
                                  cloudIn->points[i].y,
                                  cloudIn->points[i].z);
            point = q_tIn * point + v_tIn;

            pointTo.x = point.x();
            pointTo.y = point.y();
            pointTo.z = point.z();

            cloudOut->push_back(pointTo);
        }
        return cloudOut;
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint)
    {
        // return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
        Eigen::Affine3f mat;
        mat.setIdentity();
        mat.pretranslate(Eigen::Vector3f(thisPoint.x, thisPoint.y, thisPoint.z));
        mat.rotate(Eigen::Quaternionf(thisPoint.qw, thisPoint.qx, thisPoint.qy, thisPoint.qz));
        return mat;
    }

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        // return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
        //                    Point3(double(thisPoint.z),   double(thisPoint.x),    double(thisPoint.y)));
        return Pose3(Rot3(Eigen::Quaternionf(thisPoint.qw, thisPoint.qx, thisPoint.qy, thisPoint.qz).cast<double>()),
                     Point3(Eigen::Vector3f(thisPoint.x, thisPoint.y, thisPoint.z).cast<double>()));
    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(1);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure()
    {
        latestSurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloud->clear();
        nearHistorySurfKeyFrameCloudDS->clear();
        // 资源分配时初始化
        // 在互斥量被析构前不解锁
        std::lock_guard<std::mutex> lock(mtx);

        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        // 关键帧位置构建kd-tree
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3DforLoop);
        // 进行半径historyKeyframeSearchRadius内的邻域搜索，
        // currentRobotPosPoint：需要查询的点，这里取最新优化后的关键帧位置
        // pointSearchIndLoop：搜索完的邻域点对应的索引
        // pointSearchSqDisLoop：搜索完的每个邻域点与当前点之间的欧式距离
        // 0：返回的邻域个数，为0表示返回全部的邻域点
        PointType currentPoint = currentRobotPosPoint;
        // currentPoint.z = 0.0;
        kdtreeHistoryKeyPoses->radiusSearch(currentPoint, loopDistance, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        closestHistoryFrameID = -1;
        // 遍历kd-tree最近临搜索到的点
        for (int i = 0; i < pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // 需要找到，两个时间差值大于30秒
            // 时间太短的不认为是回环
            // std::cout<<"loop id : "<<id<<std::endl;
            // std::cout<<"cloudKeyPoses6D->points[id].time = "<<cloudKeyPoses6D->points[id].time<<std::endl;
            // std::cout<<"timeLaserOdometry = "<<timeLaserOdometry<<std::endl;
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) >loopInterval)
            {
                closestHistoryFrameID = id;
                break;
            }
        }
        if (closestHistoryFrameID == -1)
        {
            // 找到的点和当前时间上没有超过30秒的
            return false;
        }

        // 取最新优化后的关键帧的cornerCloudKeyFrames和surfCloudKeyFrames

        // 点云的xyz坐标进行坐标系变换(分别绕xyz轴旋转)
        // 根据对应的位姿，将边缘线点和平面点转换到世界坐标系下，并合并在一起
        *latestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);

        /*
        // latestSurfKeyFrameCloud中存储的是下面公式计算后的index(intensity):
        // thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;
        // 滤掉latestSurfKeyFrameCloud中index<0的点??? index值会小于0?
        pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
        // 遍历合并完的latestSurfKeyFrameCloud
        int cloudSize = latestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i){
            // intensity不小于0的点放进hahaCloud队列
            // 初始化时intensity是-1，滤掉那些点
            hahaCloud->push_back(latestSurfKeyFrameCloud->points[i]);
        }
        latestSurfKeyFrameCloud->clear();
        *latestSurfKeyFrameCloud = *hahaCloud;
        */

        // historyKeyframeSearchNum在utility.h中定义为25
        // 取闭环候选关键帧的 前后各25帧，进行拼接
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)//考虑距离最近的帧而不是一个序列上的帧
        {
            if (closestHistoryFrameID + j < 0 || closestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            // 要求closestHistoryFrameID + j在0到cloudKeyPoses3D->points.size()-1之间,不能超过索引
            // 这些点云已经转换到世界坐标系
            *nearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[closestHistoryFrameID + j], &cloudKeyPoses6D->points[closestHistoryFrameID + j]);
        }

        // 下采样滤波减少数据量
        downSizeFilterHistoryKeyFrames.setInputCloud(nearHistorySurfKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*nearHistorySurfKeyFrameCloudDS);

        if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = odometryFrame;
            pubHistoryKeyFrames.publish(cloudMsgTemp);
        }
        // 返回true，表示可能存在闭环
        return true; // modify
    }

    void performLoopClosure()
    {
        // 如果关键帧为空，返回
        // std::cout<<"cloudKeyPoses3D->points = "<<cloudKeyPoses3D->points.size()<<std::endl;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        if (potentialLoopFlag == false)
        {
            // 进行回环检测
            if (detectLoopClosure() == true)
            {
                potentialLoopFlag = true;
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }
            if (potentialLoopFlag == false)
            { // add jxf

                // PointType beforeStation = cloudKeyPoses3D->points[ImuLastTime];
                // PointType nowStation = cloudKeyPoses3D->points[latestFrameIDLoopCloure];
                // Eigen::Vector3f beforeStation_(beforeStation.x, beforeStation.y, beforeStation.z);
                // Eigen::Vector3f nowStation_(nowStation.x, nowStation.y, nowStation.z);

                // Eigen::Vector3f distanceBetween;
                // distanceBetween = beforeStation_ - nowStation_;

                // if (sqrt(distanceBetween(0) * distanceBetween(0) + distanceBetween(1) * distanceBetween(1) + distanceBetween(2) * distanceBetween(2)) > 3.0)
                // {
                //     float noiseScore2 = 0.0001;
                //     gtsam::Vector Vector6IMU(3);
                //     Vector6IMU << noiseScore2, noiseScore2, noiseScore2;
                //     constraintNoise = noiseModel::Diagonal::Variances(Vector6IMU);

                //     // Eigen::Affine3f tImu = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
                //     // Eigen::Affine3f tImuBefore = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[ImuLastTime]);
                //     cout<<"ImuLastTime: "<<ImuLastTime<<endl;
                //     cout<<"latestFrameIDLoopCloure: "<<latestFrameIDLoopCloure<<endl;
                //     // gtsam::Pose3 poseFromImu(Pose3(Rot3(ImuKeyFrame.cast<double>()), Point3(tImu.translation().cast<double>())));
                //     // if (ImuLastTime == 0)
                //     // {
                //         // ImuLast = tImuBefore.rotation();
                //     // }
                //     // gtsam::Pose3 poseToImu(Pose3(Rot3(ImuLast.cast<double>()), Point3(tImuBefore.translation().cast<double>())));

                //     gtSAMgraph.add(PoseRotationPrior<Pose3>(latestFrameIDLoopCloure, gtsam::Rot3(ImuKeyFrame.cast<double>()), constraintNoise));
                //     isam->update(gtSAMgraph);
                //     isam->update();
                //     gtSAMgraph.resize(0);

                //     // ImuLast = ImuKeyFrame;
                //     ImuLastTime = latestFrameIDLoopCloure;
                //     aLoopIsClosed = true;
                //     // add jxf
                // }

                return;
            }
        }
 

        potentialLoopFlag = false;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(100); // Set the maximum distance threshold between two correspondent points in source <-> target. If the distance is larger than this threshold, the points will be ignored in the alignment process.
        icp.setMaximumIterations(100);         // 最大迭代次数
        icp.setTransformationEpsilon(1e-6);    // 设置前后两次迭代的转换矩阵的最大容差（epsilion），一旦两次迭代小于这个最大容差，则认为已经收敛到最优解，迭代停止。迭代停止条件之二，默认值为：0
        icp.setEuclideanFitnessEpsilon(1e-6);  // 设置前后两次迭代的点对的欧式距离均值的最大容差，迭代终止条件之三，默认值为：-std::numeric_limits::max ()
        // 设置RANSAC运行次数
        icp.setRANSACIterations(0);

        // 最新优化后的关键帧对应的点云(在漂移世界坐标系上)
        // std::cout<<"latestSurfKeyFrameCloud : "<<latestSurfKeyFrameCloud->size()<<std::endl;
        icp.setInputSource(latestSurfKeyFrameCloud);
        // 使用detectLoopClosure()函数中下采样刚刚更新nearHistorySurfKeyFrameCloudDS
        // nearHistorySurfKeyFrameCloudDS: 闭环候选关键帧附近的帧构成的局部地图(在世界坐标系上)
        icp.setInputTarget(nearHistorySurfKeyFrameCloudDS);
        // std::cout<<"nearHistorySurfKeyFrameCloudDS : "<<nearHistorySurfKeyFrameCloudDS->size()<<std::endl;
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // 进行icp点云对齐
        icp.align(*unused_result);

        // 接受条件
        if (icp.hasConverged() == false || icp.getFitnessScore() > icpMatchScoreThre)
            return;

        // std::cout << "loop occur, ICP score = " << icp.getFitnessScore() << std::endl;

        // 以下在点云icp收敛并且噪声量在一定范围内进行
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            // icp.getFinalTransformation()的返回值是Eigen::Matrix<Scalar, 4, 4>
            pcl::transformPointCloud(*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = odometryFrame;
            pubIcpKeyFrames.publish(cloudMsgTemp);
        }
        //add jxf
        // gtsam::Pose3 ba ;
        // ba = balmOptimize(nearHistorySurfKeyFrameCloudDS,latestSurfKeyFrameCloud);
        //add jxf

        // 获取配准结果: 从漂移世界坐标系到真世界坐标系的变换
        // float x, y, z, roll, pitch, yaw;
        Eigen::Vector3f translation;
        Eigen::Quaternionf rotation;
        Eigen::Affine3f correctionCameraFrame;

        //add jxf
        // Eigen::Transform<float, 3, Eigen::Affine> correctionCameraFrame (ba.matrix().cast<float>());
        //add jxf

        correctionCameraFrame = icp.getFinalTransformation();

        // 取矫正前的关键帧位姿(转换到载体坐标系及导航坐标系n系)
        // 即当前帧坐标系 到 (漂移)世界坐标系的变换
        Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
        // 计算矫正量
        // tWrong: 矫正前的关键帧 到 世界坐标系的变换 ==> 即当前帧坐标系 到 (漂移)世界坐标系的变换
        // correctionLidarFrame: 从漂移世界坐标系到真世界坐标系的变换
        // tCorrect: 当前关键帧在真世界坐标系的位姿
        Eigen::Affine3f tCorrect = correctionCameraFrame * tWrong;
        // pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        rotation = tCorrect.rotation();
        translation = tCorrect.translation();
        // 矫正后的当前最新优化关键帧位姿
        // gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseFrom = Pose3(Rot3(rotation.cast<double>()), Point3(translation.cast<double>()));
        // 取闭环关键帧位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[closestHistoryFrameID]);
        gtsam::Vector Vector6(6);
        float noiseScore =icp.getFitnessScore() ;//
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        std::lock_guard<std::mutex> lock(mtx);
        // 因子图加入回环边
        gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise));
        isam->update(gtSAMgraph);
        isam->update();
        gtSAMgraph.resize(0);

        loopIndexContainer[latestFrameIDLoopCloure] = closestHistoryFrameID;

        aLoopIsClosed = true;
        lastCloudFrame = latestFrameIDLoopCloure;
    }

    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok())
        {
            rate.sleep();
            publishGlobalMap();
            visualizeLoopClosure();
        }
    }

    void visualizeLoopClosure()
    {
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "/loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "/loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.scale.y = 0.1;
        markerEdge.scale.z = 0.1;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = cloudKeyPoses6D->points[key_cur].x;
            p.y = cloudKeyPoses6D->points[key_cur].y;
            p.z = cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = cloudKeyPoses6D->points[key_pre].x;
            p.y = cloudKeyPoses6D->points[key_pre].y;
            p.z = cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void publishGlobalMap()
    {

        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;

        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 通过KDTree进行最近邻搜索
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // 对globalMapKeyPoses进行下采样
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i)
        {
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // 对globalMapKeyFrames进行下采样
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = odometryFrame;
        pubLaserCloudSurround.publish(cloudMsgTemp);

        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        globalMapKeyFramesDS->clear();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "globalOptimization");

    globalOptimization globalOptimization_;
    // add jxf
    //  globalOptimization_.run();
    ROS_INFO("\033[1;32m----> start globalOptimization. \033[0m");
    // add jxf
    //  std::thread 构造函数，将MO作为参数传入构造的线程中使用
    //  回环线程
    std::thread loopthread(&globalOptimization::loopClosureThread, &globalOptimization_);

    // 该线程中进行的工作是publishGlobalMap(),将数据发布到ros中，可视化
    std::thread visualizeMapThread(&globalOptimization::visualizeGlobalMapThread, &globalOptimization_);
    cout<<"gbs_"<<endl;
    ros::Rate rate(200);
    cout<<"flag1"<<endl;
    while (ros::ok())
    {
        globalOptimization_.run();
        ros::spinOnce();
        rate.sleep();
    }

    loopthread.join();
    visualizeMapThread.join();

    //added by gjf @2023.9.7
    globalOptimization_.updatePcdAndPose();
    // signal(SIGINT, globalOptimization_.sigintHandler_updatePcdAndPose());
    //added by gjf @2023.9.7

    return 0;
}
