//
// Created by Eric Fang on 7/26/17.
//

#include <ros/ros.h>


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

    search_controller controller(nh);

//    std::cout << "init complete" << std::endl;

    do {
	
        ros::spinOnce();
        rate.sleep();
    } while(ros::ok());

    return 0;
}
