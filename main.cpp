#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Householder"
#include "eigen3/Eigen/QR"
#include "eigen3/Eigen/SVD"

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "ASensor.hpp"
#include "Timer.hpp"

#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <algorithm>
#include <memory>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <ctime>
#include <filesystem>

static double DEG_TO_RAD = M_PI/180;

Eigen::Matrix3d rpy_to_mat(double roll, double pitch, double yaw){

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitX());

    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;

    Eigen::Matrix3d rotationMatrix = q.matrix();
    return rotationMatrix;
}

int main(int argc, char** argv){

    Timer timer;

    // initialize K
    Eigen::Matrix3d K;
    K << 2666.666, 0, 960,
         0, 2666.666, 540,
         0, 0, 1;
    cv::Mat K_cv = (cv::Mat_<float>(3,3) << K(0,0), 0, K(0,2),
               0, K(1,1), K(1,2),
               0, 0, 1);
    int n_points = 100;
    cv::Point2d principal_pt(K(0,2), K(1,2));
    std::shared_ptr<ASensor> cam(new ASensor(K));
    
    // Let's load images
    std::string image_path0 = std::filesystem::current_path().string() + "/../cube.png";

    cv::Mat img_1 = cv::imread(image_path0, cv::IMREAD_COLOR);

    // Let's compute cube vertices 3D coordinates
    std::vector<Eigen::Vector3d> p3ds;
    std::vector<cv::Point3d> p3ds_cv;;
    for (int i=-1; i<2; i=i+2){
        for(int k=-1; k<2; k=k+2){
            for(int j=-1; j<2; j=j+2){
                p3ds.push_back(Eigen::Vector3d(1*i,1*k,1*j));
                p3ds_cv.push_back(cv::Point3d(1*i,1*k,1*j));
            }
        }
    }

    // Let's add random points inside the cube
    double x,y,z;
    for (int i=0; i<n_points; i++){
        x = static_cast<double>(std::rand() % 100-50)/50;
        y = static_cast<double>(std::rand() % 100-50)/50;
        z = static_cast<double>(std::rand() % 100-50)/50;
        p3ds.push_back(Eigen::Vector3d(x,y,z));
        p3ds_cv.push_back(cv::Point3d(x,y,z));
    }


    // Let's compute the 3D isometry of the two cameras
    Eigen::Matrix3d w_R_cam1;
    Eigen::Quaternion<double> q1(1.0,0.0,0.0,0.0);
    w_R_cam1 = q1.matrix();
    Eigen::Vector3d w_t_cam1 = Eigen::Vector3d(-1, -0.5, 10);
    Eigen::Affine3d w_T_cam1;
    w_T_cam1.translation() = w_t_cam1;
    w_T_cam1.linear() = w_R_cam1;
    

    // Compute keypoints in the cube images
    std::vector<cv::KeyPoint> kps;
    std::vector<cv::Point2d> p2ds_cv;
    std::vector<Eigen::Vector2d>  p2ds;
    for (auto p3d : p3ds){
        Eigen::Vector2d p2d;
        cam->project(w_T_cam1 * p3d, p2d);
        kps.push_back(cv::KeyPoint(cv::Point2d(p2d.x(), p2d.y()),5));
        p2ds_cv.push_back(cv::Point2d(p2d.x(), p2d.y()));
        p2ds.push_back(p2d);
    }
    cv::Mat img_1_debug;
    cv::drawKeypoints(img_1, kps, img_1_debug);
    cv::imshow( "Keypoints image 1", img_1_debug);
    cv::waitKey(0);

    // Let's launch cv::solvePnPRansac
    cv::Mat D;
    cv::Mat tvec, rvec;
    cv::Mat inliers;
    bool use_extrinsic_guess = false;
    float confidence = 0.99;
    uint nmaxiter = 30;
    double errth = 2;

    cv::solvePnPRansac(
                p3ds_cv,
                p2ds_cv,
                K_cv,
                D,
                rvec,
                tvec,
                use_extrinsic_guess,
                nmaxiter,
                errth,
                confidence,
                inliers,
                cv::SOLVEPNP_P3P
        );
    
    std::cout << "Open CV pose estimation" << std::endl;
    std::cout << tvec << std::endl;

    // Let's try with openGV
    std::vector<Eigen::Vector3d> bearings = cam->getRays(p2ds);
    opengv::bearingVectors_t bearings_gv;
    opengv::points_t p3ds_gv;

    for (size_t i = 0 ; i < bearings.size() ; i++){
        bearings_gv.push_back(bearings.at(i));
        p3ds_gv.push_back(p3ds.at(i));
    }

    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearings_gv, p3ds_gv);

    // create a Ransac object
    opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

    // create an AbsolutePoseSacProblem
    // (algorithm is selectable: KNEIP, GAO, or EPNP)
    std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
        absposeproblem_ptr(
        new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
        adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP ) );
    
    // run ransac
    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = 2.0;
    ransac.max_iterations_ = 30;
    ransac.computeModel();
    // get the result
    opengv::transformation_t best_transformation =
        ransac.model_coefficients_;
    
    std::cout << "opengv results" << std::endl;
    std::cout << best_transformation << std::endl;
        

}