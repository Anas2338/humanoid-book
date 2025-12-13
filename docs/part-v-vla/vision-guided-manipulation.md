---
sidebar_position: 4
---

# Vision-Guided Manipulation

## Overview

Vision-guided manipulation represents a critical capability in robotics, enabling robots to perceive their environment, identify objects, and execute precise manipulation tasks based on visual feedback. This chapter explores the integration of computer vision with robotic manipulation, covering the fundamental principles, architectures, and implementation strategies for vision-guided robotic systems. These systems combine visual perception, object recognition, pose estimation, and robotic control to perform complex manipulation tasks that require precise spatial reasoning and adaptive behavior.

The integration of vision with manipulation enables robots to operate in unstructured environments where objects are not precisely positioned and tasks require adaptation based on visual feedback. Modern vision-guided manipulation systems leverage deep learning, 3D computer vision, and real-time processing to achieve robust object detection, accurate pose estimation, and precise manipulation control. These capabilities are essential for applications such as warehouse automation, manufacturing, healthcare assistance, and domestic robotics where robots must interact with diverse objects in dynamic environments.

Vision-guided manipulation involves several key components working in harmony: perception systems for object detection and pose estimation, planning algorithms for grasp and manipulation planning, control systems for executing precise movements, and feedback mechanisms for adaptive behavior. The effectiveness of these systems depends on the accuracy of visual perception, the speed of processing, and the integration between vision and control components to achieve real-time performance with high precision.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Implement vision-guided manipulation systems using computer vision and robotics frameworks
- Integrate 3D perception with robotic manipulation control
- Design grasp planning algorithms based on visual object information
- Implement real-time visual feedback control for manipulation tasks
- Evaluate the accuracy and robustness of vision-guided manipulation systems
- Address challenges in object detection, pose estimation, and grasp execution
- Optimize vision-guided manipulation systems for specific applications

## Key Concepts

### 3D Perception for Manipulation

Core perception capabilities for vision-guided manipulation:

- **Object Detection**: Identifying and localizing objects in 3D space
- **Pose Estimation**: Determining 6D pose (position and orientation) of objects
- **3D Reconstruction**: Building 3D models of objects and environments
- **Depth Estimation**: Obtaining depth information from RGB or stereo cameras
- **Point Cloud Processing**: Working with 3D point cloud data from sensors
- **Multi-view Integration**: Combining information from multiple camera views

### Grasp Planning and Execution

Techniques for planning and executing robotic grasps:

- **Grasp Synthesis**: Generating potential grasp configurations for objects
- **Grasp Evaluation**: Assessing grasp quality and stability
- **Antipodal Grasps**: Planning stable grasps using geometric properties
- **Suction-based Grasping**: Planning for suction cup grippers
- **Multi-finger Grasping**: Coordinating multiple fingers for complex grasps
- **Adaptive Grasping**: Adjusting grasp strategy based on object properties

### Visual Servoing and Control

Real-time control systems using visual feedback:

- **Image-based Visual Servoing**: Controlling robot based on image features
- **Position-based Visual Servoing**: Controlling robot based on 3D pose
- **Hybrid Visual Servoing**: Combining image and position-based approaches
- **Feedback Control**: Real-time adjustment based on visual errors
- **Adaptive Control**: Modifying control parameters based on visual feedback
- **Robust Control**: Handling visual noise and uncertainty in control

### Multi-Modal Integration

Combining vision with other sensory modalities:

- **Vision-Tactile Integration**: Combining visual and tactile feedback
- **Vision-Force Integration**: Using visual and force feedback for precision
- **Audio-Visual Integration**: Incorporating audio cues for manipulation
- **Proprioceptive Feedback**: Using joint encoders with visual information
- **Sensor Fusion**: Combining multiple modalities for robust operation
- **Cross-Modal Learning**: Learning from multiple sensory inputs

## Diagrams and Code

### Vision-Guided Manipulation Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   3D Camera     │───▶│   Perception    │───▶│   Object        │
│   System        │    │   Processing    │    │   Pose          │
│   (RGB-D,       │    │   (Detection,   │    │   Estimation    │
│   Stereo, LIDAR)│    │   Segmentation) │    │   (6D Pose)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Grasp Planning &                             │
│                    Manipulation Reasoning                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Object      │  │ Grasp       │  │ Manipulation            │ │
│  │ Properties  │  │ Planning    │  │ Planning                │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────┬─────────────────────────────┬─────────────────┘
                  │                             │
                  ▼                             ▼
         ┌─────────────────┐           ┌─────────────────┐
         │   Robot         │           │   Control &     │
         │   Kinematics    │           │   Execution     │
         │   (IK, FK)      │           │   (Trajectory,  │
         │                 │           │   Force Control) │
         └─────────────────┘           └─────────────────┘
                  │                             │
                  └─────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Visual Feedback &     │
                    │   Adaptive Control      │
                    └─────────────────────────┘
```

### Vision-Guided Manipulation System Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Pose, Point, Vector3
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import time
from typing import Dict, List, Tuple, Optional

class VisionGuidedManipulationNode(Node):
    """
    Vision-guided manipulation system combining perception and control.
    Processes visual data to guide robotic manipulation tasks.
    """

    def __init__(self):
        super().__init__('vision_guided_manipulation_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.grasp_pose_pub = self.create_publisher(Pose, '/manipulation/grasp_pose', 10)
        self.manipulation_cmd_pub = self.create_publisher(String, '/manipulation/command', 10)
        self.debug_image_pub = self.create_publisher(Image, '/manipulation/debug_image', 10)

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # System components
        self.camera_intrinsics = None
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.object_detector = ObjectDetector()
        self.grasp_planner = GraspPlanner()
        self.manipulator_controller = ManipulatorController()

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('Vision-Guided Manipulation Node initialized')

    def camera_info_callback(self, msg):
        """
        Update camera intrinsic parameters
        """
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],  # Focal length x
                'fy': msg.k[4],  # Focal length y
                'cx': msg.k[2],  # Principal point x
                'cy': msg.k[5],  # Principal point y
                'width': msg.width,
                'height': msg.height
            }

    def rgb_callback(self, msg):
        """
        Process RGB image for object detection
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_image = cv_image

            # Process image if we have depth data
            if self.latest_depth_image is not None and self.camera_intrinsics is not None:
                self.process_manipulation_task(cv_image, self.latest_depth_image)

        except Exception as e:
            self.get_logger().error(f'RGB image processing error: {str(e)}')

    def depth_callback(self, msg):
        """
        Process depth image for 3D information
        """
        try:
            # Convert depth image to numpy array
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_image = depth_image

        except Exception as e:
            self.get_logger().error(f'Depth image processing error: {str(e)}')

    def process_manipulation_task(self, rgb_image, depth_image):
        """
        Process complete manipulation task from RGB-D data
        """
        start_time = time.time()

        try:
            # Step 1: Detect objects in the scene
            objects = self.object_detector.detect_objects(rgb_image)

            if not objects:
                self.get_logger().info('No objects detected')
                return

            # Select target object (first detected for this example)
            target_object = objects[0]

            # Step 2: Estimate object pose in 3D space
            object_pose = self.estimate_object_pose(
                target_object, rgb_image, depth_image
            )

            if object_pose is None:
                self.get_logger().warn('Could not estimate object pose')
                return

            # Step 3: Plan grasp for the object
            grasp_pose = self.grasp_planner.plan_grasp(
                object_pose, target_object['class']
            )

            if grasp_pose is not None:
                # Publish grasp pose for robot execution
                grasp_msg = Pose()
                grasp_msg.position.x = grasp_pose['position'][0]
                grasp_msg.position.y = grasp_pose['position'][1]
                grasp_msg.position.z = grasp_pose['position'][2]
                grasp_msg.orientation.x = grasp_pose['orientation'][0]
                grasp_msg.orientation.y = grasp_pose['orientation'][1]
                grasp_msg.orientation.z = grasp_pose['orientation'][2]
                grasp_msg.orientation.w = grasp_pose['orientation'][3]

                self.grasp_pose_pub.publish(grasp_msg)

                # Publish manipulation command
                cmd_msg = String()
                cmd_msg.data = f"grasp_object:{target_object['class']}:{object_pose['position']}"
                self.manipulation_cmd_pub.publish(cmd_msg)

                self.get_logger().info(f'Grasp planned for {target_object["class"]} at {object_pose["position"]}')

                # Create debug image showing results
                debug_image = self.create_debug_image(
                    rgb_image, target_object, grasp_pose
                )
                debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                self.debug_image_pub.publish(debug_msg)

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.frame_count += 1

            if self.frame_count % 20 == 0:
                avg_time = sum(self.processing_times[-20:]) / min(len(self.processing_times), 20)
                self.get_logger().info(f'Vision-guided manipulation: {avg_time:.4f}s average')

        except Exception as e:
            self.get_logger().error(f'Manipulation processing error: {str(e)}')

    def estimate_object_pose(self, object_info, rgb_image, depth_image):
        """
        Estimate 6D pose of object using RGB-D data
        """
        # Extract bounding box
        bbox = object_info['bbox']
        x, y, w, h = bbox

        # Crop region of interest
        roi_rgb = rgb_image[y:y+h, x:x+w]
        roi_depth = depth_image[y:y+h, x:x+w]

        # Get 3D point cloud from depth
        points_3d = self.depth_to_pointcloud(
            roi_depth, x, y, self.camera_intrinsics
        )

        if len(points_3d) < 10:  # Need sufficient points for reliable pose
            return None

        # Estimate object center from point cloud
        object_center = np.mean(points_3d, axis=0)

        # Estimate object orientation (simplified)
        # In practice, this would use more sophisticated pose estimation
        object_orientation = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion

        return {
            'position': object_center.tolist(),
            'orientation': object_orientation,
            'bbox': bbox,
            'confidence': object_info['confidence']
        }

    def depth_to_pointcloud(self, depth_image, offset_x, offset_y, intrinsics):
        """
        Convert depth image to 3D point cloud
        """
        height, width = depth_image.shape
        points = []

        # Generate coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]

        # Convert to camera coordinates
        x_coords = x_coords + offset_x
        y_coords = y_coords + offset_y

        # Convert to 3D coordinates
        z_coords = depth_image.astype(np.float32) / 1000.0  # Convert to meters

        x_3d = (x_coords - intrinsics['cx']) * z_coords / intrinsics['fx']
        y_3d = (y_coords - intrinsics['cy']) * z_coords / intrinsics['fy']

        # Stack coordinates
        points = np.stack([x_3d, y_3d, z_coords], axis=-1).reshape(-1, 3)

        # Filter out invalid points (where depth is 0 or invalid)
        valid_mask = points[:, 2] > 0.01  # Remove points too close or invalid
        points = points[valid_mask]

        return points

    def create_debug_image(self, rgb_image, object_info, grasp_pose):
        """
        Create debug image showing detection and grasp information
        """
        debug_image = rgb_image.copy()

        # Draw bounding box
        x, y, w, h = object_info['bbox']
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw object class label
        label = f"{object_info['class']}: {object_info['confidence']:.2f}"
        cv2.putText(debug_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw grasp point (project 3D grasp position to 2D)
        grasp_pos = grasp_pose['position']
        # This is a simplified projection - in practice, you'd use camera intrinsics
        # For this example, we'll just draw at the center of the bounding box
        grasp_x = x + w // 2
        grasp_y = y + h // 2

        cv2.circle(debug_image, (grasp_x, grasp_y), 10, (255, 0, 0), 3)
        cv2.line(debug_image, (grasp_x - 15, grasp_y), (grasp_x + 15, grasp_y), (255, 0, 0), 2)
        cv2.line(debug_image, (grasp_x, grasp_y - 15), (grasp_x, grasp_y + 15), (255, 0, 0), 2)

        return debug_image

class ObjectDetector:
    """
    Object detection module for manipulation system.
    """

    def __init__(self):
        # In real implementation, this would load a pre-trained model
        # For simulation, we'll use OpenCV's built-in methods
        self.class_names = ['cup', 'box', 'ball', 'bottle', 'phone', 'book']

    def detect_objects(self, image):
        """
        Detect objects in the image (simulated implementation).
        In real implementation, this would use a deep learning model.
        """
        height, width = image.shape[:2]

        # Simulate object detection by generating plausible object locations
        # In real implementation, this would run through an actual detector
        detected_objects = []

        # For simulation, let's assume we detect 1-3 random objects
        num_objects = np.random.randint(1, 4)

        for i in range(num_objects):
            # Generate random bounding box
            w = np.random.randint(width // 8, width // 4)
            h = np.random.randint(height // 8, height // 4)
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)

            # Random class
            class_idx = np.random.randint(0, len(self.class_names))
            class_name = self.class_names[class_idx]

            # Random confidence (0.6-0.99)
            confidence = np.random.uniform(0.6, 0.99)

            detected_objects.append({
                'bbox': [x, y, w, h],
                'class': class_name,
                'confidence': confidence
            })

        return detected_objects

class GraspPlanner:
    """
    Grasp planning module for determining grasp poses.
    """

    def __init__(self):
        # Grasp planning parameters
        self.grasp_height_offset = 0.05  # 5cm above object center
        self.gripper_width_range = (0.02, 0.1)  # 2-10cm gripper width

    def plan_grasp(self, object_pose, object_class):
        """
        Plan grasp pose for the given object.
        """
        obj_pos = np.array(object_pose['position'])
        obj_orient = object_pose['orientation']

        # Plan grasp position (above the object)
        grasp_position = obj_pos.copy()
        grasp_position[2] += self.grasp_height_offset  # Lift slightly above object

        # Plan grasp orientation (typically approach from above)
        # For this example, we'll use a simple top-down grasp
        grasp_orientation = self.plan_top_down_grasp_orientation(obj_orient)

        return {
            'position': grasp_position.tolist(),
            'orientation': grasp_orientation,
            'approach_direction': [0, 0, -1],  # Approach from above
            'grasp_width': self.estimate_grasp_width(object_class)
        }

    def plan_top_down_grasp_orientation(self, object_orientation):
        """
        Plan top-down grasp orientation based on object orientation.
        """
        # For a top-down grasp, we typically want the gripper aligned with the object
        # but oriented for a vertical approach
        # This is a simplified approach - in reality, grasp planning is more complex

        # For this example, return a quaternion representing a top-down approach
        # This represents a rotation of 180 degrees around X axis (to point gripper down)
        return [0.707, 0.0, 0.0, 0.707]  # 180 deg rotation around X

    def estimate_grasp_width(self, object_class):
        """
        Estimate appropriate grasp width based on object class.
        """
        # Define typical grasp widths for different object types
        class_grasp_widths = {
            'cup': 0.06,
            'box': 0.08,
            'ball': 0.05,
            'bottle': 0.04,
            'phone': 0.07,
            'book': 0.09
        }

        return class_grasp_widths.get(object_class, 0.06)  # Default width

class ManipulatorController:
    """
    Controller for manipulator robot.
    """

    def __init__(self):
        # Controller parameters
        self.approach_distance = 0.1  # 10cm approach distance
        self.retract_distance = 0.05  # 5cm retract distance

    def generate_manipulation_trajectory(self, grasp_pose, object_pose):
        """
        Generate manipulation trajectory from grasp planning results.
        """
        trajectory = []

        # 1. Approach the grasp point
        approach_pose = grasp_pose.copy()
        approach_pose['position'][2] += self.approach_distance  # Approach from above

        trajectory.append({
            'type': 'move_to',
            'pose': approach_pose,
            'description': 'Approach grasp point'
        })

        # 2. Descend to grasp position
        trajectory.append({
            'type': 'move_to',
            'pose': grasp_pose,
            'description': 'Move to grasp position'
        })

        # 3. Close gripper
        trajectory.append({
            'type': 'gripper_control',
            'action': 'close',
            'description': 'Close gripper'
        })

        # 4. Retract
        retract_pose = grasp_pose.copy()
        retract_pose['position'][2] += self.retract_distance

        trajectory.append({
            'type': 'move_to',
            'pose': retract_pose,
            'description': 'Retract from grasp'
        })

        return trajectory

def main(args=None):
    rclpy.init(args=args)

    manipulation_node = VisionGuidedManipulationNode()

    try:
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Grasp Planning with Deep Learning

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class GraspQualityNetwork(nn.Module):
    """
    Deep neural network for predicting grasp quality from point cloud data.
    """

    def __init__(self, input_dim: int = 3072):  # 1024 points * 3 coordinates
        super(GraspQualityNetwork, self).__init__()

        self.input_dim = input_dim

        # Point cloud processing layers
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        # Global feature extraction
        self.global_features = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Grasp quality prediction
        self.quality_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output quality score between 0 and 1
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for grasp quality prediction.

        Args:
            point_cloud: Tensor of shape (batch_size, num_points, 3)

        Returns:
            Quality scores of shape (batch_size, 1)
        """
        batch_size, num_points, _ = point_cloud.shape

        # Process each point
        encoded_points = self.point_encoder(point_cloud.view(-1, 3))
        encoded_points = encoded_points.view(batch_size, num_points, -1)

        # Max pooling to get global features
        global_features = torch.max(encoded_points, dim=1)[0]  # (batch_size, 256)

        # Extract global features
        global_features = self.global_features(global_features)

        # Predict grasp quality
        quality_scores = self.quality_head(global_features)

        return quality_scores

class DeepGraspPlanner:
    """
    Deep learning-based grasp planner using point cloud data.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize network
        self.network = GraspQualityNetwork().to(self.device)

        # Load pre-trained model if provided
        if model_path:
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained grasp quality model from {model_path}")

        # Grasp sampling parameters
        self.num_grasp_samples = 100
        self.min_grasp_quality = 0.5

    def plan_grasp(self, point_cloud: np.ndarray, object_centroid: np.ndarray) -> Optional[Dict]:
        """
        Plan grasp using deep learning on point cloud data.

        Args:
            point_cloud: Point cloud of the object (N, 3)
            object_centroid: Centroid of the object (3,)

        Returns:
            Best grasp pose if found, None otherwise
        """
        # Preprocess point cloud
        processed_pc = self.preprocess_point_cloud(point_cloud, object_centroid)

        # Sample grasp candidates
        grasp_candidates = self.sample_grasp_candidates(processed_pc, object_centroid)

        # Predict grasp qualities
        with torch.no_grad():
            pc_tensor = torch.FloatTensor(processed_pc).unsqueeze(0).to(self.device)
            pc_tensor = pc_tensor.repeat(len(grasp_candidates), 1, 1)

            grasp_poses_tensor = torch.FloatTensor(grasp_candidates).to(self.device)

            # For simplicity, we'll just predict based on point cloud
            # In a real implementation, you'd incorporate grasp pose information
            quality_scores = self.network(pc_tensor).squeeze().cpu().numpy()

        # Find best grasp
        if len(quality_scores) > 0:
            best_idx = np.argmax(quality_scores)
            best_quality = quality_scores[best_idx]

            if best_quality > self.min_grasp_quality:
                best_grasp = grasp_candidates[best_idx]
                return {
                    'position': best_grasp[:3].tolist(),
                    'orientation': best_grasp[3:].tolist(),
                    'quality': float(best_quality),
                    'approach_direction': [0, 0, -1]  # Default approach
                }

        return None

    def preprocess_point_cloud(self, point_cloud: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud for neural network input.
        """
        # Center the point cloud
        centered_pc = point_cloud - centroid

        # Sample or pad to fixed number of points
        if len(centered_pc) > 1024:
            # Randomly sample 1024 points
            indices = np.random.choice(len(centered_pc), 1024, replace=False)
            centered_pc = centered_pc[indices]
        elif len(centered_pc) < 1024:
            # Pad with zeros if less than 1024 points
            padding = np.zeros((1024 - len(centered_pc), 3))
            centered_pc = np.vstack([centered_pc, padding])

        return centered_pc

    def sample_grasp_candidates(self, point_cloud: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """
        Sample potential grasp poses around the object.
        """
        grasp_poses = []

        # Sample grasp positions near the object surface
        surface_points = self.select_surface_points(point_cloud)

        for _ in range(self.num_grasp_samples):
            # Randomly select a point as grasp position
            grasp_pos_idx = np.random.randint(0, len(surface_points))
            grasp_pos = surface_points[grasp_pos_idx] + centroid  # Add back centroid offset

            # Generate random orientation (for simplicity, just top-down grasps)
            # In reality, you'd want more diverse orientations
            grasp_rot = self.generate_grasp_orientation()

            grasp_pose = np.concatenate([grasp_pos, grasp_rot])
            grasp_poses.append(grasp_pose)

        return np.array(grasp_poses)

    def select_surface_points(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Select points that are likely to be on the object surface.
        """
        # For simplicity, just return random points
        # In practice, you'd use normal estimation or other surface detection methods
        if len(point_cloud) > 200:
            indices = np.random.choice(len(point_cloud), 200, replace=False)
            return point_cloud[indices]
        else:
            return point_cloud

    def generate_grasp_orientation(self) -> np.ndarray:
        """
        Generate a feasible grasp orientation.
        """
        # For this example, we'll use a random orientation near the identity
        # In practice, you'd want to consider the object's principal axes
        rotation = R.random().as_quat()  # Random rotation as quaternion
        return rotation

class VisionGuidedGraspingSystem:
    """
    Complete vision-guided grasping system combining perception and planning.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.grasp_planner = DeepGraspPlanner(model_path)
        self.object_detector = ObjectDetector()
        self.transform_manager = TransformManager()

    def process_scene(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                     camera_intrinsics: Dict) -> List[Dict]:
        """
        Process a complete scene to find graspable objects and their grasps.

        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera
            camera_intrinsics: Camera intrinsic parameters

        Returns:
            List of grasp plans for detected objects
        """
        # 1. Detect objects in the scene
        detected_objects = self.object_detector.detect_objects(rgb_image)

        grasp_plans = []

        for obj in detected_objects:
            # 2. Extract object point cloud from depth
            obj_point_cloud, obj_centroid = self.extract_object_pointcloud(
                obj['bbox'], depth_image, camera_intrinsics
            )

            if len(obj_point_cloud) > 10:  # Need sufficient points
                # 3. Plan grasp using deep learning
                grasp_plan = self.grasp_planner.plan_grasp(obj_point_cloud, obj_centroid)

                if grasp_plan:
                    grasp_plan['object_class'] = obj['class']
                    grasp_plan['object_bbox'] = obj['bbox']
                    grasp_plans.append(grasp_plan)

        return grasp_plans

    def extract_object_pointcloud(self, bbox: List[int], depth_image: np.ndarray,
                                 intrinsics: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract point cloud for object from depth image using bounding box.

        Args:
            bbox: Bounding box [x, y, w, h]
            depth_image: Depth image
            intrinsics: Camera intrinsic parameters

        Returns:
            Point cloud and centroid
        """
        x, y, w, h = bbox

        # Crop depth region
        crop_depth = depth_image[y:y+h, x:x+w]

        # Generate point cloud
        points = []
        for v in range(h):
            for u in range(w):
                z = crop_depth[v, u] / 1000.0  # Convert to meters
                if z > 0.01:  # Valid depth
                    x_3d = (u + x - intrinsics['cx']) * z / intrinsics['fx']
                    y_3d = (v + y - intrinsics['cy']) * z / intrinsics['fy']
                    points.append([x_3d, y_3d, z])

        if points:
            points = np.array(points)
            centroid = np.mean(points, axis=0)
            return points, centroid
        else:
            return np.array([]), np.array([0, 0, 0])

class TransformManager:
    """
    Manage coordinate transformations between different frames.
    """

    def __init__(self):
        self.transforms = {}

    def add_transform(self, from_frame: str, to_frame: str, transform: np.ndarray):
        """
        Add a transformation between two frames.
        """
        key = (from_frame, to_frame)
        self.transforms[key] = transform

    def transform_point(self, point: np.ndarray, from_frame: str, to_frame: str) -> np.ndarray:
        """
        Transform a point from one frame to another.
        """
        key = (from_frame, to_frame)
        if key in self.transforms:
            transform = self.transforms[key]
            # Apply transformation: [R|t] * [p; 1]
            homogeneous_point = np.append(point, 1)
            transformed_point = transform @ homogeneous_point
            return transformed_point[:3]
        else:
            # No transformation found, return original point
            return point

# Example usage
def example_usage():
    """
    Example of using the vision-guided grasping system.
    """
    # Initialize the grasping system (without pre-trained model)
    grasping_system = VisionGuidedGraspingSystem()

    # Simulate camera intrinsics
    camera_intrinsics = {
        'fx': 554.256,  # Focal length x
        'fy': 554.256,  # Focal length y
        'cx': 320.0,    # Principal point x
        'cy': 240.0,    # Principal point y
        'width': 640,
        'height': 480
    }

    # Simulate a point cloud for testing
    # In real implementation, this would come from depth sensor
    object_points = np.random.rand(500, 3) * 0.1  # Random points in 10cm cube
    object_points[:, 2] += 0.5  # Position 50cm in front of camera

    print(f"Generated object point cloud with {len(object_points)} points")

    # Plan grasp for the object
    centroid = np.mean(object_points, axis=0)
    grasp_plan = grasping_system.grasp_planner.plan_grasp(object_points, centroid)

    if grasp_plan:
        print(f"Found grasp with quality: {grasp_plan['quality']:.3f}")
        print(f"Grasp position: {grasp_plan['position']}")
        print(f"Grasp orientation: {grasp_plan['orientation']}")
    else:
        print("No good grasp found")

if __name__ == "__main__":
    example_usage()
```

### Real-time Visual Servoing Controller

```python
import numpy as np
import cv2
import time
from typing import Dict, Tuple, Optional
import threading
from collections import deque

class VisualServoingController:
    """
    Real-time visual servoing controller for vision-guided manipulation.
    Implements image-based and position-based visual servoing.
    """

    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        # Camera intrinsic parameters
        if camera_matrix is None:
            # Default camera matrix (for example purposes)
            self.camera_matrix = np.array([
                [554.256, 0.0, 320.0],
                [0.0, 554.256, 240.0],
                [0.0, 0.0, 1.0]
            ])
        else:
            self.camera_matrix = camera_matrix

        # Servoing parameters
        self.gain = 0.5  # Servoing gain
        self.max_velocity = 0.1  # Max linear velocity (m/s)
        self.max_angular_velocity = 0.5  # Max angular velocity (rad/s)

        # Control state
        self.current_error = np.zeros(2)  # Image plane error [u, v]
        self.previous_error = np.zeros(2)
        self.integral_error = np.zeros(2)

        # PID parameters
        self.kp = 1.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.05  # Derivative gain

        # Feature tracking
        self.feature_points = deque(maxlen=10)  # Track recent feature positions
        self.target_point = None

        # Control thread
        self.control_active = False
        self.control_thread = None
        self.robot_interface = None  # Will be set externally

    def set_target(self, target_point: Tuple[int, int]):
        """
        Set the target point in image coordinates for servoing.
        """
        self.target_point = np.array(target_point, dtype=np.float32)

    def update_current_feature(self, current_point: Tuple[int, int]):
        """
        Update the current feature position for servoing.
        """
        if current_point is not None:
            current_pos = np.array(current_point, dtype=np.float32)
            self.feature_points.append(current_pos)

    def compute_control_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute control velocity based on current visual error.
        Returns linear and angular velocity components.
        """
        if (self.target_point is None or
            len(self.feature_points) == 0):
            return np.zeros(3), np.zeros(3)  # No motion

        # Get current feature position
        current_pos = self.feature_points[-1]

        # Compute image plane error
        self.current_error = self.target_point - current_pos

        # PID control
        p_term = self.kp * self.current_error
        self.integral_error += self.current_error
        i_term = self.ki * self.integral_error
        d_term = self.kd * (self.current_error - self.previous_error)

        control_output = p_term + i_term + d_term

        # Update previous error
        self.previous_error = self.current_error.copy()

        # Convert image plane velocities to Cartesian velocities
        # Using interaction matrix for image-based visual servoing
        linear_vel = np.zeros(3)
        angular_vel = np.zeros(3)

        # Simple conversion: map image errors to Cartesian motion
        # This is a simplified version - in reality, you'd use the interaction matrix
        px = self.camera_matrix[0, 0]  # Focal length in pixels
        py = self.camera_matrix[1, 1]

        # Map image plane motion to Cartesian motion
        linear_vel[0] = -control_output[0] / px * self.gain  # X translation
        linear_vel[1] = -control_output[1] / py * self.gain  # Y translation
        # Z motion based on depth error (simplified)

        # Limit velocities
        linear_vel = np.clip(linear_vel, -self.max_velocity, self.max_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)

        return linear_vel, angular_vel

    def start_servoing(self, robot_interface):
        """
        Start the visual servoing control loop.
        """
        self.robot_interface = robot_interface
        self.control_active = True
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def stop_servoing(self):
        """
        Stop the visual servoing control loop.
        """
        self.control_active = False
        if self.control_thread:
            self.control_thread.join()

    def control_loop(self):
        """
        Main control loop for visual servoing.
        """
        rate = 30  # 30 Hz control rate
        dt = 1.0 / rate
        last_time = time.time()

        while self.control_active:
            current_time = time.time()
            if current_time - last_time >= dt:
                # Compute control command
                linear_vel, angular_vel = self.compute_control_velocity()

                # Send command to robot (in real implementation)
                if self.robot_interface:
                    self.robot_interface.send_velocity_command(linear_vel, angular_vel)

                last_time = current_time

            time.sleep(0.001)  # Small sleep to prevent busy waiting

    def is_converged(self, threshold: float = 2.0) -> bool:
        """
        Check if servoing has converged to target.
        """
        if self.target_point is None or len(self.feature_points) == 0:
            return False

        current_pos = self.feature_points[-1]
        error = np.linalg.norm(self.target_point - current_pos)
        return error < threshold

    def reset(self):
        """
        Reset servoing controller state.
        """
        self.current_error = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.integral_error = np.zeros(2)
        self.feature_points.clear()
        self.target_point = None

class FeatureTracker:
    """
    Feature tracking for visual servoing.
    """

    def __init__(self):
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.prev_frame = None
        self.prev_features = None
        self.feature_ids = 0

    def detect_features(self, frame: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Detect features in the current frame.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect corners using Shi-Tomasi
        features = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            **self.feature_params
        )

        self.prev_frame = gray
        self.prev_features = features

        return features

    def track_features(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Track features from previous frame to current frame using optical flow.
        """
        if self.prev_features is None or self.prev_frame is None:
            return None, None

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_features, None, **self.lk_params
        )

        # Select good points
        if status is not None:
            good_new = next_features[status == 1]
            good_old = self.prev_features[status == 1]
        else:
            good_new = None
            good_old = None

        # Update previous frame and features
        self.prev_frame = gray
        if good_new is not None and len(good_new) > 0:
            self.prev_features = good_new.reshape(-1, 1, 2)

        return good_new, good_old

    def select_target_feature(self, tracked_features: np.ndarray,
                            target_region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Select a feature within the target region for servoing.
        """
        if tracked_features is None:
            return None

        x, y, w, h = target_region
        target_center = np.array([x + w/2, y + h/2])

        # Find feature closest to target center
        distances = np.linalg.norm(tracked_features - target_center, axis=1)
        closest_idx = np.argmin(distances)

        return tracked_features[closest_idx].reshape(2)

class VisionServoingSystem:
    """
    Complete vision servoing system for manipulation.
    """

    def __init__(self):
        self.feature_tracker = FeatureTracker()
        self.servo_controller = VisualServoingController()
        self.active = False

    def start_servoing_to_object(self, initial_frame: np.ndarray,
                               target_bbox: Tuple[int, int, int, int],
                               robot_interface: object):
        """
        Start servoing to move robot toward a target object.
        """
        # Detect features in the initial frame
        features = self.feature_tracker.detect_features(initial_frame)

        # Select target feature
        target_feature = self.feature_tracker.select_target_feature(features, target_bbox)

        if target_feature is not None:
            # Set target in servo controller
            self.servo_controller.set_target((int(target_feature[0]), int(target_feature[1])))

            # Start control loop
            self.servo_controller.start_servoing(robot_interface)
            self.active = True

            print(f"Started servoing toward target at {target_feature}")
            return True
        else:
            print("Could not find target feature for servoing")
            return False

    def update_servoing(self, current_frame: np.ndarray):
        """
        Update servoing with new frame information.
        """
        if not self.active:
            return False

        # Track features in current frame
        tracked_features, _ = self.feature_tracker.track_features(current_frame)

        if tracked_features is not None:
            # Select the target feature from tracked features
            # For simplicity, we'll use the first tracked feature
            target_feature = tracked_features[0].reshape(2)

            # Update servo controller with current feature position
            self.servo_controller.update_current_feature((target_feature[0], target_feature[1]))

            # Check if converged
            if self.servo_controller.is_converged():
                print("Servoing converged to target")
                self.stop_servoing()
                return True

        return False

    def stop_servoing(self):
        """
        Stop the servoing process.
        """
        self.servo_controller.stop_servoing()
        self.active = False
        self.servo_controller.reset()

# Example usage
def example_servoing():
    """
    Example of vision servoing for manipulation.
    """
    # Create a simulated robot interface (in real implementation, this would be a real robot)
    class SimulatedRobotInterface:
        def send_velocity_command(self, linear_vel, angular_vel):
            print(f"Sending velocity command: linear={linear_vel}, angular={angular_vel}")

    # Initialize servoing system
    servo_system = VisionServoingSystem()
    robot_interface = SimulatedRobotInterface()

    # Simulate initial frame with a target object
    initial_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a target object (for simulation)
    cv2.rectangle(initial_frame, (200, 150), (300, 250), (0, 255, 0), 2)
    cv2.circle(initial_frame, (250, 200), 5, (0, 0, 255), -1)  # Target point

    # Define target bounding box
    target_bbox = (200, 150, 100, 100)  # x, y, width, height

    print("Starting vision servoing example...")

    # Start servoing
    success = servo_system.start_servoing_to_object(initial_frame, target_bbox, robot_interface)

    if success:
        # Simulate several frames of tracking
        for i in range(10):
            # Simulate frame with target moving (for demonstration)
            frame = initial_frame.copy()
            # Move target slightly each frame
            cv2.circle(frame, (250 - i*2, 200 - i), 5, (0, 0, 255), -1)

            converged = servo_system.update_servoing(frame)
            if converged:
                break
            time.sleep(0.1)  # Simulate frame rate

        servo_system.stop_servoing()
        print("Vision servoing example completed")

if __name__ == "__main__":
    example_servoing()
```

## Labs and Exercises

### Exercise 1: 3D Object Pose Estimation
Implement a 3D object pose estimation system using RGB-D data. Evaluate the accuracy of pose estimation for different object types and compare the performance of different pose estimation algorithms.

### Exercise 2: Grasp Planning with Physics Simulation
Create a grasp planning system that integrates with a physics simulator to validate grasp stability. Test the system's ability to predict successful grasps before physical execution.

### Exercise 3: Real-time Visual Servoing
Implement a real-time visual servoing system for precise manipulation tasks. Optimize the system for low latency and evaluate its performance in tracking and positioning tasks.

### Exercise 4: Multi-Object Manipulation Planning
Develop a system that can plan manipulation sequences for multiple objects in a scene. Implement collision avoidance and task sequencing for complex multi-object manipulation tasks.

## Summary

This chapter explored vision-guided manipulation systems, demonstrating how computer vision and robotic control can be integrated to enable robots to perceive and interact with their environment. We covered the fundamental components of vision-guided manipulation, implemented deep learning-based grasp planning, and created real-time visual servoing controllers. The examples showed how visual perception can be combined with robotic control to achieve precise manipulation tasks with adaptive behavior. Vision-guided manipulation is a critical capability for robots operating in unstructured environments, enabling them to handle diverse objects and adapt to changing conditions.