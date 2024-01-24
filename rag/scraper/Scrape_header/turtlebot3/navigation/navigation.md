
[Edit on GitHub](https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/platform/turtlebot3/navigation.md "https://github.com/ROBOTIS-GIT/emanual/blob/master/docs/en/platform/turtlebot3/navigation.md") 

Kinetic 
Melodic
Noetic
Dashing
Foxy
Humble
Windows

# [Navigation](#navigation "#navigation")

**WARNING**: In this instruction, TurtleBot3 may move and rotate. Please place the robot on a safe ground.

**NOTE**

* Please run the Navigation on Remote PC.
* Make sure to launch the [Bringup](/docs/en/platform/turtlebot3/bringup/ "/docs/en/platform/turtlebot3/bringup/") from TurtleBot3 before executing any operation.
* The Navigation uses a map created by the [SLAM](/docs/en/platform/turtlebot3/slam/ "/docs/en/platform/turtlebot3/slam/"). Please prepare a map before running the Navigation.

**[Navigation](http://wiki.ros.org/navigation "http://wiki.ros.org/navigation")** is to move the robot from one location to the specified destination in a given environment. For this purpose, a map that contains geometry information of furniture, objects, and walls of the given environment is required. As described in the previous [SLAM](/docs/en/platform/turtlebot3/slam/ "/docs/en/platform/turtlebot3/slam/") section, the map was created with the distance information obtained by the sensor and the pose information of the robot itself.

The [Navigation](http://wiki.ros.org/navigation "http://wiki.ros.org/navigation") enables a robot to move from the current pose to the designated goal pose on the map by using the map, robot’s encoder, IMU sensor, and distance sensor. The procedure for performing this task is as follows.

## [Run Navigation Nodes](#run-navigation-nodes "#run-navigation-nodes")

1. If `roscore` is not running on the Remote PC, run roscore. **Skip this step if roscore is already running**.

```
$ roscore

```
2. If the `Bringup` is not running on the TurtleBot3 SBC, launch the Bringup. **Skip this step if you have launched bringup previously**.
	* Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and connect to Raspberry Pi with its IP address.
	The default password is **turtlebot**.

```
	$ ssh pi@{IP_ADDRESS_OF_RASPBERRY_PI}
	$ roslaunch turtlebot3_bringup turtlebot3_robot.launch

```
3. Launch the Navigation.  

 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml

```

![](/assets/images/icon_unfold.png) **How to save the TURTLEBOT3\_MODEL parameter?**

The `$ export TURTLEBOT3_MODEL=${TB3_MODEL}` command can be omitted if the **TURTLEBOT3\_MODEL** parameter is predefined in the `.bashrc` file.  

The `.bashrc` file is automatically loaded when a terminal window is created.

* Example of defining `TurtlBot3 Burger` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=burger' >> ~/.bashrc
$ source ~/.bashrc

```
* Example of defining `TurtlBot3 Waffle Pi` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=waffle\_pi' >> ~/.bashrc
$ source ~/.bashrc

```

1. If `roscore` is not running on the Remote PC, run roscore. **Skip this step if roscore is already running**.

```
$ roscore

```
2. If the `Bringup` is not running on the TurtleBot3 SBC, launch the Bringup. **Skip this step if you have launched bringup previously**.
	* Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and connect to Raspberry Pi with its IP address.
	The default password is **ubuntu**.

```
	$ ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
	$ roslaunch turtlebot3_bringup turtlebot3_robot.launch

```
3. Launch the Navigation.  

 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml

```

![](/assets/images/icon_unfold.png) **How to save the TURTLEBOT3\_MODEL parameter?**

The `$ export TURTLEBOT3_MODEL=${TB3_MODEL}` command can be omitted if the **TURTLEBOT3\_MODEL** parameter is predefined in the `.bashrc` file.  

The `.bashrc` file is automatically loaded when a terminal window is created.

* Example of defining `TurtlBot3 Burger` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=burger' >> ~/.bashrc
$ source ~/.bashrc

```
* Example of defining `TurtlBot3 Waffle Pi` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=waffle\_pi' >> ~/.bashrc
$ source ~/.bashrc

```

1. If `roscore` is not running on the Remote PC, run roscore. **Skip this step if roscore is already running**.

```
$ roscore

```
2. If the `Bringup` is not running on the TurtleBot3 SBC, launch the Bringup. **Skip this step if you have launched bringup previously**.
	* Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and connect to Raspberry Pi with its IP address.
	The default password is **turtlebot**. Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
	$ ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
	$ export TURTLEBOT3\_MODEL=${TB3\_MODEL}
	$ roslaunch turtlebot3_bringup turtlebot3_robot.launch

```
3. Launch the Navigation.  

 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml

```

![](/assets/images/icon_unfold.png) **How to save the TURTLEBOT3\_MODEL parameter?**

The `$ export TURTLEBOT3_MODEL=${TB3_MODEL}` command can be omitted if the **TURTLEBOT3\_MODEL** parameter is predefined in the `.bashrc` file.  

The `.bashrc` file is automatically loaded when a terminal window is created.

* Example of defining `TurtlBot3 Burger` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=burger' >> ~/.bashrc
$ source ~/.bashrc

```
* Example of defining `TurtlBot3 Waffle Pi` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=waffle\_pi' >> ~/.bashrc
$ source ~/.bashrc

```

1. If the `Bringup` is not running on the TurtleBot3 SBC, launch the Bringup. **Skip this step if you have launched bringup previously**.
	* Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and connect to Raspberry Pi with its IP address.
	The default password is **ubuntu**.

```
	$ ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
	$ export TURTLEBOT3\_MODEL=${TB3\_MODEL}
	$ ros2 launch turtlebot3_bringup robot.launch.py

```
2. Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and launch the Navigation node. ROS 2 uses [Navigation2](https://navigation.ros.org/ "https://navigation.ros.org/").
 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=$HOME/map.yaml

```

![](/assets/images/icon_unfold.png) **How to save the TURTLEBOT3\_MODEL parameter?**

The `$ export TURTLEBOT3_MODEL=${TB3_MODEL}` command can be omitted if the **TURTLEBOT3\_MODEL** parameter is predefined in the `.bashrc` file.  

The `.bashrc` file is automatically loaded when a terminal window is created.

* Example of defining `TurtlBot3 Burger` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=burger' >> ~/.bashrc
$ source ~/.bashrc

```
* Example of defining `TurtlBot3 Waffle Pi` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=waffle\_pi' >> ~/.bashrc
$ source ~/.bashrc

```

1. If the `Bringup` is not running on the TurtleBot3 SBC, launch the Bringup. **Skip this step if you have launched bringup previously**.
	* Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and connect to Raspberry Pi with its IP address.
	The default password is **ubuntu**.

```
	$ ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
	$ export TURTLEBOT3\_MODEL=${TB3\_MODEL}
	$ ros2 launch turtlebot3_bringup robot.launch.py

```
2. Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and launch the Navigation node. ROS 2 uses [Navigation2](https://navigation.ros.org/ "https://navigation.ros.org/").
 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=$HOME/map.yaml

```

![](/assets/images/icon_unfold.png) **How to save the TURTLEBOT3\_MODEL parameter?**

The `$ export TURTLEBOT3_MODEL=${TB3_MODEL}` command can be omitted if the **TURTLEBOT3\_MODEL** parameter is predefined in the `.bashrc` file.  

The `.bashrc` file is automatically loaded when a terminal window is created.

* Example of defining `TurtlBot3 Burger` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=burger' >> ~/.bashrc
$ source ~/.bashrc

```
* Example of defining `TurtlBot3 Waffle Pi` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=waffle\_pi' >> ~/.bashrc
$ source ~/.bashrc

```

1. If the `Bringup` is not running on the TurtleBot3 SBC, launch the Bringup. **Skip this step if you have launched bringup previously**.
	* Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and connect to Raspberry Pi with its IP address.
	The default password is **ubuntu**.

```
	$ ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
	$ export TURTLEBOT3\_MODEL=${TB3\_MODEL}
	$ ros2 launch turtlebot3_bringup robot.launch.py

```
2. Open a new terminal from Remote PC with `Ctrl` + `Alt` + `T` and launch the Navigation node. ROS 2 uses [Navigation2](https://navigation.ros.org/ "https://navigation.ros.org/").
 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
$ export TURTLEBOT3\_MODEL=burger
$ ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=$HOME/map.yaml

```

![](/assets/images/icon_unfold.png) **How to save the TURTLEBOT3\_MODEL parameter?**

The `$ export TURTLEBOT3_MODEL=${TB3_MODEL}` command can be omitted if the **TURTLEBOT3\_MODEL** parameter is predefined in the `.bashrc` file.  

The `.bashrc` file is automatically loaded when a terminal window is created.

* Example of defining `TurtlBot3 Burger` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=burger' >> ~/.bashrc
$ source ~/.bashrc

```
* Example of defining `TurtlBot3 Waffle Pi` as a default model.

```
$ echo 'export TURTLEBOT3\_MODEL=waffle\_pi' >> ~/.bashrc
$ source ~/.bashrc

```

1. If the `Bringup` is not running on the TurtleBot3, launch the Bringup. **Skip this step if you have launched bringup previously**.

```
> roslaunch turtlebot3_bringup turtlebot3_robot.launch

```
2. Launch the Navigation.  

 Please use the proper keyword among `burger`, `waffle`, `waffle_pi` for the `TURTLEBOT3_MODEL` parameter.

```
> set TURTLEBOT3\_MODEL=burger
> roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=%USERPROFILE%\map.yaml

```

## [Estimate Initial Pose](#estimate-initial-pose "#estimate-initial-pose")

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz2 menu.
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map. 
 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_01.png)
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ ros2 run turtlebot3_teleop teleop_keyboard

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz2 menu.
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map. 
 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_01.png)
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ ros2 run turtlebot3_teleop teleop_keyboard

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz2 menu.
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map. 
 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_01.png)
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
$ ros2 run turtlebot3_teleop teleop_keyboard

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

**Initial Pose Estimation** must be performed before running the Navigation as this process initializes the AMCL parameters that are critical in Navigation. TurtleBot3 has to be correctly located on the map with the LDS sensor data that neatly overlaps the displayed map.

1. Click the `2D Pose Estimate` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_pose_button.png)
2. Click on the map where the actual robot is located and drag the large green arrow toward the direction where the robot is facing.
3. Repeat step 1 and 2 until the LDS sensor data is overlayed on the saved map.
4. Launch keyboard teleoperation node to precisely locate the robot on the map.

```
> roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch

```
5. Move the robot back and forth a bit to collect the surrounding environment information and narrow down the estimated location of the TurtleBot3 on the map which is displayed with tiny green arrows.  

![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_01.png)
![](/assets/images/platform/turtlebot3/navigation/tb3_amcl_particle_02.png)
6. Terminate the keyboard teleoperation node by entering `Ctrl` + `C` to the teleop node terminal in order to prevent different **cmd\_vel** values are published from multiple nodes during Navigation.

## [Set Navigation Goal](#set-navigation-goal "#set-navigation-goal")

1. Click the `2D Nav Goal` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation**

* The robot will create a path to reach to the Navigation Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation will use local path planner to avoid the obstacle.
* Setting a Navigation Goal might fail if the path to the Navigation Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation Goal.
* [Official ROS Navigation Wiki](http://wiki.ros.org/navigation "http://wiki.ros.org/navigation")

1. Click the `2D Nav Goal` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation**

* The robot will create a path to reach to the Navigation Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation will use local path planner to avoid the obstacle.
* Setting a Navigation Goal might fail if the path to the Navigation Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation Goal.
* [Official ROS Navigation Wiki](http://wiki.ros.org/navigation "http://wiki.ros.org/navigation")

1. Click the `2D Nav Goal` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation**

* The robot will create a path to reach to the Navigation Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation will use local path planner to avoid the obstacle.
* Setting a Navigation Goal might fail if the path to the Navigation Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation Goal.
* [Official ROS Navigation Wiki](http://wiki.ros.org/navigation "http://wiki.ros.org/navigation")

1. Click the `Navigation2 Goal` button in the RViz2 menu.
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_02.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation2**

* The robot will create a path to reach to the Navigation2 Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation2 will use local path planner to avoid the obstacle.
* Setting a Navigation2 Goal might fail if the path to the Navigation2 Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation2 Goal.
* [Official ROS2 Navigation2 Wiki](https://navigation.ros.org/ "https://navigation.ros.org/")

1. Click the `Navigation2 Goal` button in the RViz2 menu.
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_02.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation2**

* The robot will create a path to reach to the Navigation2 Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation2 will use local path planner to avoid the obstacle.
* Setting a Navigation2 Goal might fail if the path to the Navigation2 Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation2 Goal.
* [Official ROS2 Navigation2 Wiki](https://navigation.ros.org/ "https://navigation.ros.org/")

1. Click the `Navigation2 Goal` button in the RViz2 menu.
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/ros2/tb3_navigation2_rviz_02.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation2**

* The robot will create a path to reach to the Navigation2 Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation2 will use local path planner to avoid the obstacle.
* Setting a Navigation2 Goal might fail if the path to the Navigation2 Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation2 Goal.
* [Official ROS2 Navigation2 Wiki](https://navigation.ros.org/ "https://navigation.ros.org/")

1. Click the `2D Nav Goal` button in the RViz menu.  

![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal_button.png)
2. Click on the map to set the destination of the robot and drag the green arrow toward the direction where the robot will be facing.
	* This green arrow is a marker that can specify the destination of the robot.
	* The root of the arrow is `x`, `y` coordinate of the destination, and the angle `θ` is determined by the orientation of the arrow.
	* As soon as x, y, θ are set, TurtleBot3 will start moving to the destination immediately.
	 ![](/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)

![](/assets/images/icon_unfold.png) Read more about **Navigation**

* The robot will create a path to reach to the Navigation Goal based on the global path planner. Then, the robot moves along the path. If an obstacle is placed in the path, the Navigation will use local path planner to avoid the obstacle.
* Setting a Navigation Goal might fail if the path to the Navigation Goal cannot be created. If you wish to stop the robot before it reaches to the goal position, set the current position of TurtleBot3 as a Navigation Goal.
* [Official ROS Navigation Wiki](http://wiki.ros.org/navigation "http://wiki.ros.org/navigation")

## [Tuning Guide](#tuning-guide "#tuning-guide")

Navigation stack has many parameters to change performances for different robots.

You can get more information about Navigation tuning from [Basic Navigation Tuning Guide](http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide "http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide"), [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf"), and the chapter 11 of [ROS Robot Programming](https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51 "https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51") book.

### inflation\_radius

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it don’t across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

### cost\_scaling\_factor

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This factor is multiplied by cost value. Because it is an reciprocal propotion, this parameter is increased, the cost is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The best path is for the robot to pass through a center of between obstacles. Set this factor to be smaller in order to far from obstacles.

### max\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

### min\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

### max\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum translational velocity. The robot can not be faster than this.

### min\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum translational velocity. The robot can not be slower than this.

### max\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

### min\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum rotational velocity. The robot can not be slower than this.

### acc\_lim\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the translational acceleration limit.

### acc\_lim\_theta

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the rotational acceleration limit.

### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

### sim\_time

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Too low value is in sufficient time to pass narrow area and too high value is not allowed rapidly rotates. You can watch defferences of length of the yellow line in below image.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

Navigation stack has many parameters to change performances for different robots.

You can get more information about Navigation tuning from [Basic Navigation Tuning Guide](http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide "http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide"), [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf"), and the chapter 11 of [ROS Robot Programming](https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51 "https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51") book.

### inflation\_radius

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it don’t across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

### cost\_scaling\_factor

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This factor is multiplied by cost value. Because it is an reciprocal propotion, this parameter is increased, the cost is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The best path is for the robot to pass through a center of between obstacles. Set this factor to be smaller in order to far from obstacles.

### max\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

### min\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

### max\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum translational velocity. The robot can not be faster than this.

### min\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum translational velocity. The robot can not be slower than this.

### max\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

### min\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum rotational velocity. The robot can not be slower than this.

### acc\_lim\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the translational acceleration limit.

### acc\_lim\_theta

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the rotational acceleration limit.

### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

### sim\_time

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Too low value is in sufficient time to pass narrow area and too high value is not allowed rapidly rotates. You can watch defferences of length of the yellow line in below image.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

Navigation stack has many parameters to change performances for different robots.

You can get more information about Navigation tuning from [Basic Navigation Tuning Guide](http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide "http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide"), [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf"), and the chapter 11 of [ROS Robot Programming](https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51 "https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51") book.

### inflation\_radius

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it don’t across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

### cost\_scaling\_factor

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This factor is multiplied by cost value. Because it is an reciprocal propotion, this parameter is increased, the cost is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The best path is for the robot to pass through a center of between obstacles. Set this factor to be smaller in order to far from obstacles.

### max\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

### min\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

### max\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum translational velocity. The robot can not be faster than this.

### min\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum translational velocity. The robot can not be slower than this.

### max\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

### min\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum rotational velocity. The robot can not be slower than this.

### acc\_lim\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the translational acceleration limit.

### acc\_lim\_theta

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the rotational acceleration limit.

### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

### sim\_time

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Too low value is in sufficient time to pass narrow area and too high value is not allowed rapidly rotates. You can watch defferences of length of the yellow line in below image.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

Navigation2 stack has many parameters to change performances for different robots. Although it’s similar to the ROS1 Navigation, please refer to the [Configuration Guide of Navigation2](https://navigation.ros.org/configuration/index.html "https://navigation.ros.org/configuration/index.html") or [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf") for more details.

### Costmap Parameters

#### inflation\_layer.inflation\_radius

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it does not across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

#### inflation\_layer.cost\_scaling\_factor

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This is an inverse proportional factor that is multiplied by the value of the costmap. If this parameter is increased, the value of the costmap is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The optimal path for the robot to pass through obstacles is to take a median path between them. Setting a smaller value for this parameter will create a farther path from the obstacles.

### dwb\_controller

#### max\_vel\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

#### min\_vel\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

#### max\_vel\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The maximum y velocity for the robot in m/s.

#### min\_vel\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The minimum y velocity for the robot in m/s.

#### max\_vel\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

#### min\_speed\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* Actual value of the minimum rotational speed. The robot can not be slower than this.

#### max\_speed\_xy

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The absolute value of the maximum translational velocity for the robot in m/s.

#### min\_speed\_xy

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The absolute value of the minimum translational velocity for the robot in m/s.

#### acc\_lim\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The x acceleration limit of the robot in meters/sec^2.

#### acc\_lim\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The y acceleration limit of the robot in meters/sec^2.

#### acc\_lim\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The rotational acceleration limit of the robot in radians/sec^2.

#### decel\_lim\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the x direction in m/s^2.

#### decel\_lim\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the y direction in m/s^2.

#### decel\_lim\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the theta direction in rad/s^2.

#### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

#### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

#### transform\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* It allows the latency for tf messages.

#### sim\_time

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Setting this too small makes robot difficult to pass a narrow space while large value limits dynamic turns. You can observe the defferences of length of the yellow line in below image that represents the simulation path.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

Navigation2 stack has many parameters to change performances for different robots. Although it’s similar to the ROS1 Navigation, please refer to the [Configuration Guide of Navigation2](https://navigation.ros.org/configuration/index.html "https://navigation.ros.org/configuration/index.html") or [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf") for more details.

### Costmap Parameters

#### inflation\_layer.inflation\_radius

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it does not across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

#### inflation\_layer.cost\_scaling\_factor

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This is an inverse proportional factor that is multiplied by the value of the costmap. If this parameter is increased, the value of the costmap is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The optimal path for the robot to pass through obstacles is to take a median path between them. Setting a smaller value for this parameter will create a farther path from the obstacles.

### dwb\_controller

#### max\_vel\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

#### min\_vel\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

#### max\_vel\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The maximum y velocity for the robot in m/s.

#### min\_vel\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The minimum y velocity for the robot in m/s.

#### max\_vel\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

#### min\_speed\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* Actual value of the minimum rotational speed. The robot can not be slower than this.

#### max\_speed\_xy

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The absolute value of the maximum translational velocity for the robot in m/s.

#### min\_speed\_xy

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The absolute value of the minimum translational velocity for the robot in m/s.

#### acc\_lim\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The x acceleration limit of the robot in meters/sec^2.

#### acc\_lim\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The y acceleration limit of the robot in meters/sec^2.

#### acc\_lim\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The rotational acceleration limit of the robot in radians/sec^2.

#### decel\_lim\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the x direction in m/s^2.

#### decel\_lim\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the y direction in m/s^2.

#### decel\_lim\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the theta direction in rad/s^2.

#### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

#### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

#### transform\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* It allows the latency for tf messages.

#### sim\_time

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Setting this too small makes robot difficult to pass a narrow space while large value limits dynamic turns. You can observe the defferences of length of the yellow line in below image that represents the simulation path.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

Navigation2 stack has many parameters to change performances for different robots. Although it’s similar to the ROS1 Navigation, please refer to the [Configuration Guide of Navigation2](https://navigation.ros.org/configuration/index.html "https://navigation.ros.org/configuration/index.html") or [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf") for more details.

### Costmap Parameters

#### inflation\_layer.inflation\_radius

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it does not across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

#### inflation\_layer.cost\_scaling\_factor

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This is an inverse proportional factor that is multiplied by the value of the costmap. If this parameter is increased, the value of the costmap is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The optimal path for the robot to pass through obstacles is to take a median path between them. Setting a smaller value for this parameter will create a farther path from the obstacles.

### dwb\_controller

#### max\_vel\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

#### min\_vel\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

#### max\_vel\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The maximum y velocity for the robot in m/s.

#### min\_vel\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The minimum y velocity for the robot in m/s.

#### max\_vel\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

#### min\_speed\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* Actual value of the minimum rotational speed. The robot can not be slower than this.

#### max\_speed\_xy

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The absolute value of the maximum translational velocity for the robot in m/s.

#### min\_speed\_xy

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The absolute value of the minimum translational velocity for the robot in m/s.

#### acc\_lim\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The x acceleration limit of the robot in meters/sec^2.

#### acc\_lim\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The y acceleration limit of the robot in meters/sec^2.

#### acc\_lim\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The rotational acceleration limit of the robot in radians/sec^2.

#### decel\_lim\_x

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the x direction in m/s^2.

#### decel\_lim\_y

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the y direction in m/s^2.

#### decel\_lim\_theta

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The deceleration limit of the robot in the theta direction in rad/s^2.

#### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

#### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

#### transform\_tolerance

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* It allows the latency for tf messages.

#### sim\_time

* Defined in `turtlebot3_navigation2/param/${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Setting this too small makes robot difficult to pass a narrow space while large value limits dynamic turns. You can observe the defferences of length of the yellow line in below image that represents the simulation path.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

Navigation stack has many parameters to change performances for different robots.

You can get more information about Navigation tuning from [Basic Navigation Tuning Guide](http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide "http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide"), [ROS Navigation Tuning Guide by Kaiyu Zheng](http://kaiyuzheng.me/documents/navguide.pdf "http://kaiyuzheng.me/documents/navguide.pdf"), and the chapter 11 of [ROS Robot Programming](https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51 "https://community.robotsource.org/t/download-the-ros-robot-programming-book-for-free/51") book.

### inflation\_radius

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This parameter makes inflation area from the obstacle. Path would be planned in order that it don’t across this area. It is safe that to set this to be bigger than robot radius. For more information, please refer to the [costmap\_2d wiki](http://wiki.ros.org/costmap_2d#Inflation "http://wiki.ros.org/costmap_2d#Inflation").

![](/assets/images/platform/turtlebot3/navigation/tuning_inflation_radius.png)

### cost\_scaling\_factor

* Defined in `turtlebot3_navigation/param/costmap_common_param_${TB3_MODEL}.yaml`
* This factor is multiplied by cost value. Because it is an reciprocal propotion, this parameter is increased, the cost is decreased.

![](/assets/images/platform/turtlebot3/navigation/tuning_cost_scaling_factor.png)

The best path is for the robot to pass through a center of between obstacles. Set this factor to be smaller in order to far from obstacles.

### max\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the maximum value of translational velocity.

### min\_vel\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set the minimum value of translational velocity. If set this negative, the robot can move backwards.

### max\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum translational velocity. The robot can not be faster than this.

### min\_trans\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum translational velocity. The robot can not be slower than this.

### max\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the maximum rotational velocity. The robot can not be faster than this.

### min\_rot\_vel

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the minimum rotational velocity. The robot can not be slower than this.

### acc\_lim\_x

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the translational acceleration limit.

### acc\_lim\_theta

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* Actual value of the rotational acceleration limit.

### xy\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The x,y distance allowed when the robot reaches its goal pose.

### yaw\_goal\_tolerance

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* The yaw angle allowed when the robot reaches its goal pose.

### sim\_time

* Defined in `turtlebot3_navigation/param/dwa_local_planner_params_${TB3_MODEL}.yaml`
* This factor is set forward simulation in seconds. Too low value is in sufficient time to pass narrow area and too high value is not allowed rapidly rotates. You can watch defferences of length of the yellow line in below image.

![](/assets/images/platform/turtlebot3/navigation/tuning_sim_time.png)

 Previous Page
Next Page 