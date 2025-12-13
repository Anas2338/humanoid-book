---
sidebar_position: 2
---

# Gazebo Environments

## Overview

Gazebo stands as one of the premier physics simulation environments for robotics research and development, offering realistic simulation of robot hardware, sensors, and environments. This chapter delves into the architecture, capabilities, and practical applications of Gazebo for creating digital twins of robotic systems. We explore the process of building realistic simulation environments, configuring sensors, and leveraging Gazebo's powerful physics engine for robot development and testing.

Gazebo provides a comprehensive platform for simulating robots in complex environments with accurate physics modeling, diverse sensor simulation, and realistic rendering capabilities. The simulation environment bridges the gap between pure software development and real-world deployment, enabling extensive testing and validation of robotic systems before physical implementation. This capability is especially valuable for humanoid robotics, where physical testing can be expensive and time-consuming.

The integration of Gazebo with the Robot Operating System (ROS) creates a powerful ecosystem for robotics development, allowing seamless transition between simulation and real robot deployment. Through Gazebo's plugin architecture, developers can customize sensor models, physics properties, and environmental conditions to closely match real-world scenarios. This chapter provides comprehensive coverage of Gazebo's features and their application to humanoid robotics.

Gazebo's modular architecture supports a wide range of robot models, from simple wheeled robots to complex multi-legged systems like humanoids. Its realistic physics engine, based on Open Dynamics Engine (ODE), Bullet, or DART, ensures that behaviors learned in simulation have a higher likelihood of transferring successfully to real robots. The environment also supports cloud-based simulation and large-scale experimentation, making it suitable for reinforcement learning and other data-intensive robotic applications.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Design and implement realistic Gazebo simulation environments for robotic applications
- Configure and calibrate sensors in Gazebo to match real hardware specifications
- Utilize Gazebo plugins for custom robot behaviors and environmental effects
- Integrate Gazebo with ROS/ROS 2 for seamless simulation-to-reality transfer
- Optimize simulation parameters for computational efficiency and accuracy
- Validate robot control algorithms in Gazebo before physical deployment
- Create custom world models and scenarios for specific robotic tasks

## Key Concepts

### Gazebo Architecture and Components

Gazebo's architecture consists of several interconnected components that enable realistic simulation:

- **Physics Engine**: Core simulation engine supporting ODE, Bullet, and DART
- **Rendering Engine**: Realistic graphics rendering using OGRE
- **Sensor Simulation**: Accurate modeling of various sensor types (cameras, LIDAR, IMU, etc.)
- **Plugin System**: Extensible architecture for custom behaviors and interfaces
- **Transport Layer**: Inter-process communication for distributed simulation

### World Modeling and Environment Creation

Creating realistic simulation environments requires careful attention to physical properties:

- **Terrain Generation**: Modeling of outdoor environments with varied terrain
- **Static Objects**: Placement of furniture, obstacles, and architectural elements
- **Dynamic Elements**: Moving objects and interactive environmental features
- **Lighting Conditions**: Realistic illumination modeling for sensor simulation
- **Weather Effects**: Simulation of atmospheric conditions affecting sensors

### Sensor Modeling and Calibration

Accurate sensor simulation is crucial for effective sim-to-real transfer:

- **Camera Models**: Pinhole and stereo camera simulation with distortion
- **Range Sensors**: LIDAR, sonar, and infrared sensor modeling
- **Inertial Measurement Units**: IMU simulation with noise and bias modeling
- **Force/Torque Sensors**: Simulation of tactile and force feedback
- **GPS and Compass**: Global positioning and orientation sensors

### ROS/ROS 2 Integration

Gazebo's integration with ROS/ROS 2 enables seamless simulation workflows:

- **Robot Description Format**: URDF/SDF for robot model definition
- **Control Interfaces**: ROS control integration for joint control
- **Sensor Data Publishing**: Realistic sensor data streams in ROS format
- **Simulation Control**: Programmatic control of simulation state
- **Clock Synchronization**: Simulation time management

### Performance Optimization

Efficient simulation requires balancing accuracy with computational demands:

- **Level of Detail**: Managing visual and physical complexity
- **Update Rates**: Configuring appropriate update frequencies for different components
- **Parallel Processing**: Utilizing multi-core systems for simulation
- **Cloud Simulation**: Leveraging distributed computing for large-scale experiments
- **Deterministic Simulation**: Ensuring reproducible results

## Diagrams and Code

### Gazebo Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gazebo Simulator                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Physics       │  │   Rendering     │  │   Sensors       │  │
│  │   Engine        │  │   Engine        │  │   Simulation    │  │
│  │   (ODE/Bullet)  │  │   (OGRE)        │  │   (Cameras,     │  │
│  └─────────────────┘  └─────────────────┘  │   LIDAR, etc.)  │  │
│         │                      │            └─────────────────┘  │
│         └──────────────────────┼────────────────────────────────┘
│                                │
│         ┌─────────────────────────────────────────────────┐
│         │           Plugin System                         │
│         │  ┌─────────────────┐  ┌─────────────────┐      │
│         │  │   Control       │  │   GUI           │      │
│         │  │   Plugins       │  │   Plugins       │      │
│         │  └─────────────────┘  └─────────────────┘      │
│         │  ┌─────────────────┐  ┌─────────────────┐      │
│         │  │   Sensor        │  │   World         │      │
│         │  │   Plugins       │  │   Plugins       │      │
│         │  └─────────────────┘  └─────────────────┘      │
│         └─────────────────────────────────────────────────┘
│                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│         ROS/ROS 2 Interface    │                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Robot State Publishing    │ Control Command Receiving │   │
│  │  Sensor Data Publishing    │ Joint Trajectory Control  │   │
│  │  TF Broadcasting          │ Gazebo Services           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### World Definition Example

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <!-- Indoor office environment with realistic furniture -->
  <world name="office_world">
    <!-- Include standard models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom office environment -->
    <model name="office_room">
      <!-- Room walls -->
      <pose>0 0 0 0 0 0</pose>

      <!-- Floor -->
      <link name="floor">
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Walls -->
      <link name="wall_north">
        <pose>0 5 2.5 0 0 0</pose>
        <collision name="wall_north_collision">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall_north_visual">
          <geometry>
            <box>
              <size>10 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.9 0.9 0.9 1</ambient>
            <diffuse>0.95 0.95 0.95 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Furniture -->
      <include>
        <uri>model://table</uri>
        <pose>2 0 0 0 0 0</pose>
      </include>

      <include>
        <uri>model://chair</uri>
        <pose>1.5 -0.8 0 0 0 1.57</pose>
      </include>

      <include>
        <uri>model://cylinder</uri>
        <pose>-2 1 0.5 0 0 0</pose>
        <box>
          <size>0.5 0.5 1.0</size>
        </box>
      </include>
    </model>

    <!-- Lighting -->
    <light name="ceiling_light_1" type="point">
      <pose>0 0 4 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <attenuation>
        <range>10</range>
        <constant>0.2</constant>
        <linear>0.5</linear>
        <quadratic>0.05</quadratic>
      </attenuation>
    </light>

    <!-- Physics properties -->
    <physics name="ode_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

### Robot Model with Sensors Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot_with_sensors" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
  </link>

  <!-- Chassis -->
  <link name="chassis">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.8 0.8 0.5"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 0.8"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.8 0.8 0.5"/>
      </geometry>
    </collision>
  </link>

  <joint name="base_to_chassis" type="fixed">
    <parent link="base_link"/>
    <child link="chassis"/>
    <origin xyz="0 0 0.75" rpy="0 0 0"/>
  </joint>

  <!-- Camera sensor -->
  <link name="camera_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="chassis"/>
    <child link="camera_link"/>
    <origin xyz="0.3 0 0.1" rpy="0 0 0"/>
  </joint>

  <!-- Camera sensor definition -->
  <gazebo reference="camera_link">
    <sensor name="camera1" type="camera">
      <update_rate>30.0</update_rate>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>0.0</updateRate>
        <cameraName>simple_camera</cameraName>
        <imageTopicName>image_raw</imageTopicName>
        <cameraInfoTopicName>camera_info</cameraInfoTopicName>
        <frameName>camera_link</frameName>
        <hackBaseline>0.07</hackBaseline>
        <distortionK1>0.0</distortionK1>
        <distortionK2>0.0</distortionK2>
        <distortionK3>0.0</distortionK3>
        <distortionT1>0.0</distortionT1>
        <distortionT2>0.0</distortionT2>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LIDAR sensor -->
  <link name="lidar_link">
    <inertial>
      <mass value="0.2"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.0002" ixy="0.0" ixz="0.0" iyy="0.0002" iyz="0.0" izz="0.0002"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="lidar_joint" type="fixed">
    <parent link="chassis"/>
    <child link="lidar_link"/>
    <origin xyz="0.3 0 0.2" rpy="0 0 0"/>
  </joint>

  <!-- LIDAR sensor definition -->
  <gazebo reference="lidar_link">
    <sensor name="lidar1" type="ray">
      <update_rate>40</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>720</samples>
            <resolution>1</resolution>
            <min_angle>-1.570796</min_angle>
            <max_angle>1.570796</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>30.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <topicName>scan</topicName>
        <frameName>lidar_link</frameName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU sensor -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1e-6" ixy="0.0" ixz="0.0" iyy="1e-6" iyz="0.0" izz="1e-6"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="chassis"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- IMU sensor definition -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <visualize>true</visualize>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <topicName>imu/data</topicName>
        <serviceName>imu/service</serviceName>
        <gaussianNoise>0.0</gaussianNoise>
        <frameName>imu_link</frameName>
        <initialOrientationAsReference>false</initialOrientationAsReference>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Python Script for Gazebo Control and Monitoring

```python
#!/usr/bin/env python3

"""
Gazebo simulation control and monitoring script.
Demonstrates programmatic interaction with Gazebo simulation.
"""

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import tf.transformations as tft

class GazeboSimulationController:
    """
    A controller for interacting with Gazebo simulation environment.
    """

    def __init__(self):
        rospy.init_node('gazebo_simulation_controller', anonymous=True)

        # Initialize CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Robot state tracking
        self.robot_pose = Pose()
        self.robot_twist = Twist()
        self.laser_scan = None
        self.camera_image = None
        self.imu_data = None

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/simple_robot/cmd_vel', Twist, queue_size=10)

        # Subscribers
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        self.laser_scan_sub = rospy.Subscriber('/scan', LaserScan, self.laser_scan_callback)
        self.camera_sub = rospy.Subscriber('/simple_camera/image_raw', Image, self.camera_callback)
        self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)

        # Services
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Simulation parameters
        self.robot_name = 'simple_robot'
        self.rate = rospy.Rate(30)  # 30 Hz update rate

        rospy.loginfo("Gazebo simulation controller initialized")

    def model_states_callback(self, data):
        """
        Callback for model states to track robot position and velocity.
        """
        try:
            idx = data.name.index(self.robot_name)
            self.robot_pose = data.pose[idx]
            self.robot_twist = data.twist[idx]
        except ValueError:
            # Robot not found in model states
            pass

    def laser_scan_callback(self, data):
        """
        Callback for laser scan data.
        """
        self.laser_scan = data

    def camera_callback(self, data):
        """
        Callback for camera image data.
        """
        try:
            self.camera_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {str(e)}")

    def imu_callback(self, data):
        """
        Callback for IMU data.
        """
        self.imu_data = data

    def get_robot_state(self):
        """
        Get current robot state from Gazebo.
        """
        try:
            resp = self.get_model_state(model_name=self.robot_name)
            return resp.pose, resp.twist
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")
            return None, None

    def set_robot_state(self, pose, twist):
        """
        Set robot state in Gazebo.
        """
        try:
            from gazebo_msgs.msg import ModelState
            model_state = ModelState()
            model_state.model_name = self.robot_name
            model_state.pose = pose
            model_state.twist = twist
            model_state.reference_frame = 'world'

            resp = self.set_model_state(model_state)
            return resp.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")
            return False

    def move_robot(self, linear_vel, angular_vel):
        """
        Send velocity commands to the robot.
        """
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)

    def analyze_laser_scan(self):
        """
        Analyze laser scan data to detect obstacles and free space.
        """
        if self.laser_scan is None:
            return None

        # Calculate minimum distance in front of robot (within 30 degrees)
        front_range_start = len(self.laser_scan.ranges) // 2 - 15
        front_range_end = len(self.laser_scan.ranges) // 2 + 15

        front_distances = self.laser_scan.ranges[front_range_start:front_range_end]
        min_front_dist = min(front_distances) if front_distances else float('inf')

        # Calculate average distance on left and right sides
        left_distances = self.laser_scan.ranges[:len(self.laser_scan.ranges)//4]
        right_distances = self.laser_scan.ranges[3*len(self.laser_scan.ranges)//4:]

        avg_left_dist = sum(left_distances) / len(left_distances) if left_distances else 0
        avg_right_dist = sum(right_distances) / len(right_distances) if right_distances else 0

        return {
            'min_front_distance': min_front_dist,
            'avg_left_distance': avg_left_dist,
            'avg_right_distance': avg_right_dist,
            'ranges': self.laser_scan.ranges,
            'angle_increment': self.laser_scan.angle_increment
        }

    def process_camera_image(self):
        """
        Process camera image for basic computer vision tasks.
        """
        if self.camera_image is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive threshold to detect objects
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        min_area = 500
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # Draw contours on original image
        result_image = self.camera_image.copy()
        cv2.drawContours(result_image, significant_contours, -1, (0, 255, 0), 2)

        # Calculate centroids of significant contours
        centroids = []
        for contour in significant_contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                centroids.append((cx, cy))

        return {
            'contours': significant_contours,
            'centroids': centroids,
            'processed_image': result_image
        }

    def simple_navigation_behavior(self):
        """
        Implement a simple navigation behavior using laser scan data.
        """
        scan_analysis = self.analyze_laser_scan()

        if scan_analysis is None:
            # No scan data available, stop robot
            self.move_robot(0.0, 0.0)
            return

        min_front_dist = scan_analysis['min_front_distance']
        avg_left_dist = scan_analysis['avg_left_distance']
        avg_right_dist = scan_analysis['avg_right_distance']

        # Simple obstacle avoidance behavior
        linear_vel = 0.5  # Default forward speed
        angular_vel = 0.0  # Default no turning

        if min_front_dist < 1.0:  # Obstacle detected in front
            if avg_left_dist > avg_right_dist:
                # Turn left (more space on left)
                linear_vel = 0.0
                angular_vel = 0.5
            else:
                # Turn right (more space on right)
                linear_vel = 0.0
                angular_vel = -0.5
        else:
            # No obstacle in front, move forward
            linear_vel = 0.5
            angular_vel = 0.0

        self.move_robot(linear_vel, angular_vel)

    def run_simulation_loop(self, duration=60):
        """
        Run the main simulation control loop for a specified duration.
        """
        start_time = rospy.Time.now()

        rospy.loginfo(f"Starting simulation control loop for {duration} seconds...")

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            elapsed_time = (current_time - start_time).to_sec()

            if elapsed_time > duration:
                rospy.loginfo("Simulation duration completed.")
                break

            # Execute navigation behavior
            self.simple_navigation_behavior()

            # Print status periodically
            if int(elapsed_time) % 10 == 0:
                pose, twist = self.get_robot_state()
                if pose:
                    rospy.loginfo(
                        f"Time: {elapsed_time:.1f}s, "
                        f"Position: ({pose.position.x:.2f}, {pose.position.y:.2f}), "
                        f"Orientation: {tft.euler_from_quaternion([
                            pose.orientation.x, pose.orientation.y,
                            pose.orientation.z, pose.orientation.w
                        ])[2]:.2f}"
                    )

            self.rate.sleep()

def main():
    """
    Main function to run the Gazebo simulation controller.
    """
    controller = GazeboSimulationController()

    try:
        # Run simulation for 60 seconds
        controller.run_simulation_loop(duration=60)
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down simulation controller...")
    finally:
        # Stop the robot before exiting
        controller.move_robot(0.0, 0.0)

if __name__ == '__main__':
    main()
```

## Labs and Exercises

### Exercise 1: Custom World Creation
Create a custom Gazebo world file that represents a specific environment for humanoid robotics (e.g., home environment, factory floor, or outdoor terrain). Include various obstacles, furniture, and surfaces with different friction properties. Test your world by spawning a robot model and verifying that physics interactions behave as expected.

### Exercise 2: Sensor Calibration
Implement a sensor calibration procedure for a simulated robot in Gazebo. Compare the simulated sensor data with expected values and adjust sensor parameters (noise, bias, scale factors) to better match real-world sensor characteristics. Document the differences between simulated and real sensors.

### Exercise 3: Navigation in Gazebo
Develop a navigation system for a robot in Gazebo that can successfully navigate through a complex environment with obstacles. Use the simulated sensors to detect obstacles and plan paths around them. Test the system with different environment configurations to ensure robustness.

### Exercise 4: Simulation-to-Reality Transfer
Design and conduct experiments to evaluate how well behaviors learned in Gazebo transfer to real robots. Identify key factors that affect transferability (sim-to-real gap) and propose methods to minimize this gap through domain randomization or other techniques.