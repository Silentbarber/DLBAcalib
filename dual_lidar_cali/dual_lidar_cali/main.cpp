#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include "BA.h"
#include <gtsam/geometry/Rot3.h>





int main(int argc, char* argv[]) {


    std::string data_path = argv[1];



    pcl::PointCloud<pcl::PointXYZI>::Ptr global_map = read_point_cloud(data_path+"pointclouds/Sub/" + argv[2]);

    std::string yml_path = std::string(ROOT_DIR) + "initV2HParam.yml";
    Eigen::Matrix4d init_extrinsic_V2H = read_matrix_from_yml(yml_path, "extrinsic_V2H");
    std::cout << init_extrinsic_V2H << std::endl;



    std::vector<std::vector<double>> odomter_pose = read_txt(data_path+"odometer/HBA.txt");
    std::vector<std::vector<double>> timestamp = read_txt(data_path+"dual_lidar_cali/vertPerTime.txt");



    std::vector<std::vector<double>> corresponded_pose = find_corresponding_rows(odomter_pose,timestamp);

 
    vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> vec_vertical2H_cloud;
    for (int i = 0; i < corresponded_pose.size(); i++) {

        pcl::PointCloud<pcl::PointXYZI>::Ptr vertical_cloud = read_point_cloud(data_path+"dual_lidar_cali/plane/vert/"+std::to_string(i)+".pcd");
        pcl::PointCloud<pcl::PointXYZI>::Ptr vertical2H_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::transformPointCloud(*vertical_cloud, *vertical2H_cloud, init_extrinsic_V2H);
        vec_vertical2H_cloud.push_back(vertical2H_cloud);
    }


    gtsam::Pose3 vert_to_hori_pose(init_extrinsic_V2H); 
    gtsam::Pose3 vert_to_hori_pose_error = gtsam::Pose3().identity();
    std::vector<gtsam::Pose3> cali_iters;
    std::vector<std::vector<double>> cali_error_iters;


    for(int iter = 0; iter < 20; iter++)
    {
        std::cout<<"number of iterations: "<<iter+1<<std::endl;
        std::vector<gtsam::Pose3> pose_bufs;


        for (int i = 0; i < corresponded_pose.size(); i++) {


            Eigen::Matrix4d transform_matrix = getTransformMatrix(corresponded_pose[i]);


            pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::transformPointCloud(*global_map, *transformed_cloud, transform_matrix);


            pcl::PointCloud<pcl::PointXYZI>::Ptr vertical2H_cloud = vec_vertical2H_cloud.at(i);


            gtsam::Pose3 global_map_pose = gtsam::Pose3().identity();
            std::vector<gtsam::Pose3> ba_pose;
            ba_pose.push_back(global_map_pose);
            ba_pose.push_back(vert_to_hori_pose_error);

            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> ba_clouds;
            ba_clouds.push_back(transformed_cloud);
            ba_clouds.push_back(vertical2H_cloud);


            ba_pose = balmOptimize(ba_pose,ba_clouds);
            pose_bufs.push_back(ba_pose[1]);
            std::cout<<"第"<<i<<"站BA校准结果："<<ba_pose[1]<<std::endl;

        }


        cali_iters.push_back(up_date_cali(pose_bufs));


        if(iter_stop(cali_iters.back(),vert_to_hori_pose_error,cali_error_iters))
        {
            std::cout<<"===========迭代结束======="<<std::endl;
            break;
        }


        vert_to_hori_pose_error = cali_iters.back();
    }

    gtsam::Pose3 final_cali = cali_iters.back();
    std::cout<<"=============最终标定结果============"<<std::endl;
    gtsam::Pose3 result_extrinsic_V2H(final_cali.matrix()*init_extrinsic_V2H);
    std::cout << result_extrinsic_V2H << std::endl;


    save_cali_poses(cali_iters,data_path);
    save_error_poses(cali_error_iters,data_path);
    save_result_poses(result_extrinsic_V2H, data_path);

    return 0;
}
