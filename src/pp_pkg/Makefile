FILE=purepursuit

.PHONY:
	all init gazebo

init: SHELL:=/bin/bash
init:
	source /opt/ros/foxy/setup.bash
	source install/setup.bash

gazebo: export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:'ros2 pkg \ prefix turtlebot3_gazebo \ '/share/turtlebot3_gazebo/models/
gazebo: export TURTLEBOT3_MODEL=burger
gazebo: init
	ros2 launch turtlebot3_gazebo empty_world.launch.py &

run: 
	colcon build
	ros2 run pp_pkg $(FILE)

all: gazebo run