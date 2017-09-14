//
// Created by Eric Fang on 7/26/17.
// edited by Nick Zhang

// this test node should read image published to camera/image
// and detect target in it and lable that and output that image
// this is done through target_detector with log enabled

#include <ros/ros.h>
#include <test_module/target_detector.h>


/**
 * main loop, runs at 60Hz
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv) {

    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh("~");

    ros::Rate rate(60);
    target_detector detector(nh);


//    std::cout << "init complete" << std::endl;

    do {
	
        ros::spinOnce();
        rate.sleep();
    } while(ros::ok());

    return 0;
}
