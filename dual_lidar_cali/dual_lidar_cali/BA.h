//
// Created by ubuntu18 on 2024/4/8.
//

#ifndef DUAL_LIDAR_CALI_BA_H
#define DUAL_LIDAR_CALI_BA_H
#include "include/bavoxel.hpp"
#include <pcl/filters/voxel_grid.h>
#include <yaml-cpp/yaml.h>
#include "include/tools.hpp"
#include <gtsam/geometry/Pose3.h>

#define MAT4_FROM_ARRAY(v)       v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]



Eigen::Matrix4d read_matrix_from_yml(std::string file_path, std::string field){
    YAML::Node config = YAML::LoadFile(file_path);
    Eigen::Matrix4d matrix;
    std::vector<double> matrix_data;
    if (config[field]) {
        matrix_data = config[field].as<std::vector<double>>();


        if (matrix_data.size() != 16) {
            std::cerr << "Error: '" + field + "' field does not contain 16 elements." << std::endl;
        }
        matrix<<MAT4_FROM_ARRAY(matrix_data);
    } else {
        std::cerr << "Error: 'mapping:" + field + "' field not found in YAML file." << std::endl;
    }
    return matrix;
}


pcl::PointCloud<pcl::PointXYZI>::Ptr read_point_cloud(std::string file_path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    if (pcl::io::loadPCDFile(file_path, *cloud) == -1)
    {
        std::cout << "Error loading point cloud" << std::endl;
        return nullptr;
    }
    return cloud;
}


std::vector<std::vector<double>> read_txt(std::string file_path) {
    std::ifstream infile(file_path);
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }
    return data;
}

std::vector<std::vector<double>> find_corresponding_rows(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    std::vector<std::vector<double>> result;
    int index_tracker = 0;
    for (const auto& row_b : B) {
        double timestamp_b = row_b[1];
        for (int index = index_tracker;index<A.size();index++) {
            double timestamp_a = A[index][2];
            if (timestamp_a >=timestamp_b) {
                result.push_back(A[index]);
                index_tracker = index+1;
                break;
            }
        }
    }
    return result;
}
//

static Eigen::Matrix4d getTransformMatrix(std::vector<double> corresponded_pose)
{
    Eigen::Matrix4d transform_matrix;
    Eigen::Matrix3d rotation_matrix;


    Eigen::Vector3d eulerAngle(corresponded_pose[6], corresponded_pose[7], corresponded_pose[8]);
    Eigen::Vector3d translation_vec(corresponded_pose[3], corresponded_pose[4], corresponded_pose[5]);

    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle(0),Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle(1),Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle(2),Eigen::Vector3d::UnitZ()));

    rotation_matrix=yawAngle*pitchAngle*rollAngle;

    transform_matrix.setIdentity();
    transform_matrix.block<3,3>(0,0) = rotation_matrix;
    transform_matrix.topRightCorner<3,1>() = translation_vec;

    return  transform_matrix.inverse();
}

void save_cali_poses(std::vector<gtsam::Pose3> cali_pose_iters, std::string data_path) {

    std::ofstream outfile(data_path+"dual_lidar_cali/cali.txt", std::ios::out);
    for(auto pose:cali_pose_iters)
    {
        outfile << pose.translation().x() << " " << pose.translation().y() << " " << pose.translation().z() << " " << pose.rotation().rpy().x() << " " << pose.rotation().rpy().y() << " " << pose.rotation().rpy().z() << std::endl;
    }

    outfile.close();
}

void save_error_poses(std::vector<std::vector<double>> cali_error_iters, std::string data_path) {

    std::ofstream outfile(data_path+"dual_lidar_cali/error.txt", std::ios::out);
    for(auto error:cali_error_iters)
    {
        outfile << error[0] << " " << error[1] << std::endl;
    }

    outfile.close();
}

void save_result_poses(gtsam::Pose3 pose, std::string data_path){
    Eigen::Matrix4d matrix = pose.matrix();

    std::ofstream file(data_path+"dual_lidar_cali/result.txt", std::ios::out);

    file << "extrinsic_V2H: [";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            file << matrix(i, j);
            if (j < 3) file << ",";
        }
        if(i < 3) file << ",\n";
        else file << "]";
    }

    file.close();
}




gtsam::Pose3 calculate_average_pose(std::vector<gtsam::Pose3> cali_pose)
{
    std::cout<<"========计算平均位姿=========="<<std::endl;
    gtsam::Rot3 mean_rot=cali_pose[0].rotation();
    gtsam::Vector3 mean_trans=cali_pose[0].translation();
    double error=1e-7;


    while (true)
    {
        gtsam::Pose3 error_rot_init;error_rot_init.identity();
        auto r = gtsam::Rot3::Logmap(error_rot_init.rotation());
        for(int i=0;i<cali_pose.size();++i)
        {
            gtsam::Rot3 rot_error=mean_rot.between(cali_pose[i].rotation());
             r+=gtsam::Rot3::Logmap(rot_error);
        }
        r/=cali_pose.size();
        if(r.norm()<=error)
        {
            break;
        }
        else
        {
            mean_rot = mean_rot*gtsam::Rot3::Expmap(r);
        }
    }


    for(int i=1;i<cali_pose.size();++i)
    {
        mean_trans+=cali_pose[i].translation();
    }
    mean_trans/=cali_pose.size();

    gtsam::Pose3 mean_pose(mean_rot,mean_trans);
    std::cout<<"========计算完成！=========="<<std::endl;
    std::cout<<"平均位姿："<<mean_pose<<std::endl;
    return mean_pose;
}
gtsam::Pose3 up_date_cali(std::vector<gtsam::Pose3> cali_pose)
{
    return calculate_average_pose(cali_pose);
}

bool iter_stop(gtsam::Pose3 pose_curr,gtsam::Pose3 pose_last,std::vector<std::vector<double>> &cali_error_iters)
{
    double angErr = 0, tranErr = 0;
    std::vector<double> cali_error;
    gtsam::Pose3 pose_error = pose_curr.between(pose_last);
    angErr=pose_error.rotation().rpy().norm();
    tranErr=pose_error.translation().norm();
    std::cout<<"angErr: "<<angErr<<" tranErr: "<<tranErr<<std::endl;
    cali_error.push_back(angErr);
    cali_error.push_back(tranErr);
    cali_error_iters.push_back(cali_error);

    return (angErr<1e-7 && tranErr<1e-7);
}

Eigen::Matrix4d IMUST2gtsamPose(IMUST pose)
{
    Eigen::Matrix4d tempPose;
    Eigen::RowVector4d A31 = {0.0, 0.0, 0.0, 1.0};
    tempPose << pose.R, pose.p, A31;
    return tempPose;
}
vector<gtsam::Pose3> balmOptimize(vector<gtsam::Pose3> x_bufs,vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pl_full,int voxelSize = 1,int threshNum = 0.05) 
{
    voxel_size=voxelSize;
    thresh_num =  threshNum;

    int sizes = pl_full.size();
    win_size = sizes;

    vector<IMUST> x_buf;
    for (int i = 0; i < sizes; ++i)
    {
        IMUST ep;
        ep.R = x_bufs[i].rotation().matrix();
        ep.p =x_bufs[i].translation();
        x_buf.push_back(ep);
    }
    IMUST es0 = x_buf[0];
    for (int i = 0; i < x_buf.size(); i++)
    {
        x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
        x_buf[i].R = es0.R.transpose() * x_buf[i].R;
    }
  
    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

    for (int i = 0; i < win_size; i++)
        cut_voxel(surf_map, *pl_full[i], x_buf[i], i);

    VOX_HESS voxhess;
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
    {
        iter->second->recut(win_size);
        iter->second->tras_opt(voxhess, win_size); 
    }

    printf("start optimize\n");
    BALM2 opt_lsv;
    Eigen::VectorXd tempHs= opt_lsv.damping_iter(x_buf, voxhess);

    for (auto iter = surf_map.begin(); iter != surf_map.end();)
    {
        delete iter->second;
        surf_map.erase(iter++);
    }
    surf_map.clear();

    vector<gtsam::Pose3> betweenPose;
    for(int i = 0; i<x_buf.size();++i)
    {
        gtsam::Pose3 betwenn(IMUST2gtsamPose(x_buf[i]));
        betweenPose.push_back(betwenn);
    }

    return betweenPose;
}
#endif //DUAL_LIDAR_CALI_BA_H
