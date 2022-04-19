
#ifndef CONFIGREADER_H
#define CONFIGREADER_H

#include <yaml-cpp/yaml.h>
#include <iostream>

// This structure contains the configuration parameters located in the config file.
struct Config {

    // problem parameters
    int n_points;
    double stdev_noise;

    // OpenCV parameters
    int n_iter_cv;
    double err_th_cv;
    double confidence;

    // OpenGV parameters
    double err_th_gv;
    double n_iter_gv;

};

Config readParameterFile(const std::string path){
    YAML::Node yaml_file = YAML::LoadFile(path);
    Config config;

    config.n_points = yaml_file["n_points"].as<int>();
    config.stdev_noise = yaml_file["stdev_noise"].as<double>();

    config.n_iter_cv = yaml_file["n_iter_cv"].as<int>();
    config.n_iter_gv = yaml_file["n_iter_gv"].as<int>();

    config.err_th_cv = yaml_file["err_th_cv"].as<double>();
    config.err_th_gv = yaml_file["err_th_gv"].as<double>();

    config.confidence = yaml_file["confidence"].as<double>();

    return config;
}

#endif