---
sidebar_position: 3
---

# Python Bridging with rclpy

## Overview

Python integration with ROS 2 through the `rclpy` client library provides a powerful and accessible way to develop robotic applications. Python's simplicity and extensive ecosystem of scientific computing libraries make it an ideal choice for rapid prototyping, AI algorithm development, and high-level robot control. The `rclpy` package provides Python bindings for the ROS 2 client library (rcl), enabling Python nodes to seamlessly integrate with the broader ROS 2 ecosystem.

The integration between Python and ROS 2 is facilitated by the Python Client Library for ROS 2 (rclpy), which abstracts the complexities of the underlying middleware while maintaining the performance and reliability of the ROS 2 architecture. This enables developers to leverage Python's rich ecosystem of libraries for machine learning, computer vision, data analysis, and scientific computing within the ROS 2 framework.

Python's interpreted nature and extensive package management system (pip) facilitate rapid development and prototyping, making it particularly valuable for research and development in robotics. The ability to integrate Python-based AI algorithms with ROS 2's distributed architecture enables sophisticated robotic systems that can leverage the latest advances in artificial intelligence and machine learning.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Create and manage ROS 2 nodes using the rclpy library
- Implement publishers, subscribers, services, and actions in Python
- Handle ROS 2 messages and parameters within Python nodes
- Integrate Python AI and machine learning libraries with ROS 2
- Design efficient message handling and processing in Python
- Debug and profile Python-based ROS 2 applications
- Optimize Python code for real-time robotic applications

## Key Concepts

### rclpy Architecture and Design

The rclpy library provides a Python interface to the ROS 2 client library:

- **Node abstraction**: Python classes that encapsulate ROS 2 node functionality
- **Message serialization**: Automatic conversion between Python objects and ROS messages
- **Event handling**: Asynchronous callback execution for message processing
- **Parameter management**: Configuration and runtime parameter handling
- **Time and logging**: Integration with ROS 2 time and logging systems

### Node Creation and Management in Python

Python nodes follow the same lifecycle as other ROS 2 nodes but with Python-specific interfaces:

- **Initialization**: Setting up the node with name and options
- **Entity creation**: Publishers, subscribers, services, and timers
- **Callback registration**: Associating functions with message events
- **Execution**: Running the node's event loop with `rclpy.spin()`
- **Cleanup**: Proper resource deallocation and shutdown

### Message Handling and Serialization

Python's dynamic typing integrates with ROS message definitions:

- **Message import**: Importing generated message types from ROS packages
- **Message creation**: Constructing message objects with Python syntax
- **Serialization**: Automatic conversion to/from network format
- **Type safety**: Runtime type checking for message fields
- **Custom messages**: Creating and using user-defined message types

### Parameter Management and Lifecycle

Parameters provide runtime configuration for Python nodes:

- **Declarative parameters**: Declaring parameters with types and constraints
- **Dynamic reconfiguration**: Changing parameters at runtime
- **Parameter validation**: Ensuring parameter values meet requirements
- **Parameter callbacks**: Reacting to parameter changes
- **Node lifecycle integration**: Parameter handling during state transitions

### Integration with Python AI Libraries

Python's rich ecosystem integrates with ROS 2 for AI applications:

- **NumPy/Pandas**: Scientific computing and data processing
- **TensorFlow/PyTorch**: Deep learning model inference
- **OpenCV**: Computer vision and image processing
- **SciPy**: Scientific algorithms and optimization
- **Scikit-learn**: Machine learning algorithms

## Diagrams and Code

### Python-ROS 2 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Application Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   AI/ML Apps    │  │   CV/Planning   │  │   Control Apps  │  │
│  │ (TensorFlow,    │  │ (OpenCV, etc.)  │  │ (PID, etc.)     │  │
│  │ PyTorch, etc.)  │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    rclpy Client Library                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │    Publishers   │  │   Subscribers   │  │   Services/     │  │
│  │                 │  │                 │  │   Actions       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   ROS 2 Client Library (rcl)                   │
├─────────────────────────────────────────────────────────────────┤
│                     DDS Middleware Layer                        │
└─────────────────────────────────────────────────────────────────┘
```

### Advanced Python Node with AI Integration

```python
#!/usr/bin/env python3

"""
Advanced Python ROS 2 node with AI integration.
Demonstrates integration of Python AI libraries with ROS 2.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np
from cv_bridge import CvBridge
import cv2
import time


class AIVisionNode(Node):
    """
    A Python ROS 2 node that integrates computer vision and AI.
    Processes camera images and publishes navigation commands.
    """

    def __init__(self):
        super().__init__('ai_vision_node')

        # Initialize OpenCV bridge for image conversion
        self.bridge = CvBridge()

        # Create subscription to camera image topic
        image_qos = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            image_qos
        )

        # Create publisher for robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create publisher for processed image info
        self.info_pub = self.create_publisher(String, '/vision_info', 10)

        # Variables for processing
        self.last_processed_time = time.time()
        self.processing_interval = 0.5  # Process every 0.5 seconds
        self.object_detected = False

        self.get_logger().info('AI Vision Node initialized')

    def image_callback(self, msg):
        """
        Callback function to process incoming camera images.
        """
        current_time = time.time()

        # Throttle processing to avoid overloading the system
        if current_time - self.last_processed_time < self.processing_interval:
            return

        self.last_processed_time = current_time

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process the image using computer vision
            processed_result = self.process_image(cv_image)

            # Publish information about the processed image
            info_msg = String()
            info_msg.data = f'Objects detected: {processed_result["object_count"]}, ' \
                           f'Largest contour area: {processed_result["largest_area"]:.2f}'
            self.info_pub.publish(info_msg)

            # Generate and publish velocity commands based on image analysis
            if processed_result['object_detected']:
                self.navigate_to_object(processed_result['centroid'])
            else:
                self.explore_randomly()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, cv_image):
        """
        Process the image using computer vision techniques.
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive threshold to detect objects
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area to find significant objects
        min_area = 500  # Minimum area to consider as an object
        significant_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        result = {
            'object_count': len(significant_contours),
            'object_detected': len(significant_contours) > 0,
            'largest_area': 0,
            'centroid': None
        }

        if significant_contours:
            # Find the largest contour
            largest_contour = max(significant_contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            result['largest_area'] = largest_area

            # Calculate centroid of the largest contour
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                result['centroid'] = (cx, cy)

        return result

    def navigate_to_object(self, centroid):
        """
        Generate velocity commands to navigate towards the detected object.
        """
        if centroid is None:
            return

        # Get image dimensions (assuming standard camera resolution)
        img_width = 640  # Standard width
        img_height = 480  # Standard height

        # Calculate the center of the image
        img_center_x = img_width / 2

        # Calculate offset from center (positive = object is to the right)
        offset_x = centroid[0] - img_center_x

        # Create Twist message for velocity commands
        cmd_msg = Twist()

        # Set linear velocity based on distance to object (simplified)
        cmd_msg.linear.x = 0.2  # Move forward at 0.2 m/s

        # Set angular velocity based on offset from center
        # Proportional controller to center the object
        cmd_msg.angular.z = -offset_x * 0.001  # Adjust gain as needed

        # Publish the command
        self.cmd_pub.publish(cmd_msg)

        self.get_logger().info(f'Navigating to object at ({centroid[0]}, {centroid[1]})')

    def explore_randomly(self):
        """
        Generate random exploration commands when no object is detected.
        """
        cmd_msg = Twist()

        # Random exploration: occasionally turn
        if np.random.random() < 0.1:  # 10% chance to turn
            cmd_msg.angular.z = np.random.uniform(-0.5, 0.5)  # Random turn
        else:
            cmd_msg.linear.x = 0.1  # Move forward slowly

        self.cmd_pub.publish(cmd_msg)

        self.get_logger().info('Exploring randomly - no object detected')


def main(args=None):
    """
    Main function to initialize and run the AI vision node.
    """
    rclpy.init(args=args)

    ai_vision_node = AIVisionNode()

    try:
        rclpy.spin(ai_vision_node)
    except KeyboardInterrupt:
        ai_vision_node.get_logger().info('Shutting down AI Vision Node...')
    finally:
        ai_vision_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Python Service with ML Model Integration

```python
#!/usr/bin/env python3

"""
Python service server with ML model integration.
Demonstrates integration of machine learning models with ROS 2 services.
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from example_interfaces.srv import SetBool
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os


class MLInferenceService(Node):
    """
    A Python service server that provides ML inference capabilities.
    """

    def __init__(self):
        super().__init__('ml_inference_service')

        # Create service for ML inference
        self.inference_service = self.create_service(
            SetBool, 'ml_inference', self.inference_callback
        )

        # Create service for model training
        self.train_service = self.create_service(
            Trigger, 'train_model', self.train_model_callback
        )

        # Initialize ML model
        self.model = LinearRegression()
        self.model_trained = False

        # Sample training data (in a real system, this would come from elsewhere)
        self.training_data = {
            'features': np.random.rand(100, 3),  # 100 samples, 3 features
            'targets': np.random.rand(100)       # 100 target values
        }

        self.get_logger().info('ML Inference Service initialized')

    def inference_callback(self, request, response):
        """
        Callback for ML inference service.
        """
        if not self.model_trained:
            response.success = False
            response.message = 'Model not trained yet. Call /train_model first.'
            return response

        try:
            # Parse input data from request (in a real system, you'd have a custom service)
            # For this example, we'll use the request data flag to determine input
            if request.data:  # If True, use sample data; if False, use different sample
                input_data = np.random.rand(1, 3)  # Single sample with 3 features
            else:
                input_data = np.random.rand(1, 3) * 2  # Different sample

            # Perform inference
            prediction = self.model.predict(input_data)

            response.success = True
            response.message = f'Prediction: {prediction[0]:.4f}'

            self.get_logger().info(f'ML Inference: input={input_data[0]}, prediction={prediction[0]:.4f}')

        except Exception as e:
            response.success = False
            response.message = f'Error during inference: {str(e)}'
            self.get_logger().error(f'ML inference error: {str(e)}')

        return response

    def train_model_callback(self, request, response):
        """
        Callback for model training service.
        """
        try:
            # Train the model with the stored training data
            self.model.fit(
                self.training_data['features'],
                self.training_data['targets']
            )

            self.model_trained = True

            response.success = True
            response.message = f'Model trained successfully with {len(self.training_data["features"])} samples'

            self.get_logger().info(f'Model trained: R² score = {self.model.score(self.training_data["features"], self.training_data["targets"]):.4f}')

        except Exception as e:
            response.success = False
            response.message = f'Error during training: {str(e)}'
            self.get_logger().error(f'Model training error: {str(e)}')

        return response


def main(args=None):
    """
    Main function to initialize and run the ML inference service.
    """
    rclpy.init(args=args)

    ml_service = MLInferenceService()

    try:
        rclpy.spin(ml_service)
    except KeyboardInterrupt:
        ml_service.get_logger().info('Shutting down ML Inference Service...')
    finally:
        ml_service.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Parameter Management in Python Nodes

```python
#!/usr/bin/env python3

"""
Python node demonstrating parameter management.
Shows how to declare, use, and respond to parameter changes.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64
import time


class ParameterDemoNode(Node):
    """
    A Python ROS 2 node demonstrating parameter management.
    """

    def __init__(self):
        super().__init__('parameter_demo_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('sensitivity', 0.5,
                              'Sensitivity factor for sensor processing (0.0-1.0)')
        self.declare_parameter('threshold', 10.0,
                              'Threshold value for detection')
        self.declare_parameter('debug_mode', False,
                              'Enable debug output')
        self.declare_parameter('max_velocity', 1.0,
                              'Maximum allowed velocity')

        # Create publisher for processed values
        self.value_pub = self.create_publisher(Float64, 'processed_value', 10)

        # Create timer for periodic processing
        self.timer = self.create_timer(1.0, self.process_data)

        # Store parameter values
        self.sensitivity = self.get_parameter('sensitivity').value
        self.threshold = self.get_parameter('threshold').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.max_velocity = self.get_parameter('max_velocity').value

        # Set callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

        self.get_logger().info('Parameter Demo Node initialized')
        self.log_current_parameters()

    def parameter_callback(self, params):
        """
        Callback function that handles parameter changes.
        """
        successful = True
        reason = ''

        for param in params:
            if param.name == 'sensitivity':
                if param.value < 0.0 or param.value > 1.0:
                    successful = False
                    reason = 'Sensitivity must be between 0.0 and 1.0'
                    break
            elif param.name == 'max_velocity':
                if param.value <= 0.0:
                    successful = False
                    reason = 'Max velocity must be positive'
                    break

        if successful:
            # Update stored values
            for param in params:
                if param.name == 'sensitivity':
                    self.sensitivity = param.value
                elif param.name == 'threshold':
                    self.threshold = param.value
                elif param.name == 'debug_mode':
                    self.debug_mode = param.value
                elif param.name == 'max_velocity':
                    self.max_velocity = param.value

            if self.debug_mode:
                self.log_current_parameters()

        return SetParametersResult(successful=successful, reason=reason)

    def log_current_parameters(self):
        """
        Log current parameter values for debugging.
        """
        self.get_logger().info(
            f'Current parameters - '
            f'sensitivity: {self.sensitivity}, '
            f'threshold: {self.threshold}, '
            f'debug_mode: {self.debug_mode}, '
            f'max_velocity: {self.max_velocity}'
        )

    def process_data(self):
        """
        Process simulated sensor data using current parameters.
        """
        # Simulate sensor reading
        raw_value = 5.0 + (time.time() % 10)  # Time-varying value

        # Apply sensitivity parameter
        processed_value = raw_value * self.sensitivity

        # Apply threshold
        if processed_value > self.threshold:
            processed_value = self.threshold  # Clamp to threshold

        # Apply maximum velocity limit
        if processed_value > self.max_velocity:
            processed_value = self.max_velocity

        # Publish processed value
        msg = Float64()
        msg.data = float(processed_value)
        self.value_pub.publish(msg)

        if self.debug_mode:
            self.get_logger().info(
                f'Raw: {raw_value:.2f}, Processed: {processed_value:.2f}, '
                f'Params - Sensitivity: {self.sensitivity}, Threshold: {self.threshold}'
            )


def main(args=None):
    """
    Main function to initialize and run the parameter demo node.
    """
    rclpy.init(args=args)

    param_node = ParameterDemoNode()

    try:
        rclpy.spin(param_node)
    except KeyboardInterrupt:
        param_node.get_logger().info('Shutting down Parameter Demo Node...')
    finally:
        param_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Labs and Exercises

### Exercise 1: Sensor Data Processing Node
Create a Python node that subscribes to sensor data (e.g., laser scan, IMU, or camera data) and processes it using Python libraries. Implement filtering, feature extraction, or other processing techniques appropriate for the sensor type. Publish the processed results to another topic.

### Exercise 2: AI Inference Service
Implement a Python service server that performs AI inference (e.g., object detection, classification, or regression) on request. Use a pre-trained model from a Python ML library (TensorFlow, PyTorch, or scikit-learn) and return the results to the client.

### Exercise 3: Dynamic Parameter Tuning
Create a Python node that uses ROS 2 parameters for configuration and implements a callback to respond to parameter changes at runtime. Demonstrate how the node behavior changes when parameters are updated using `ros2 param set` commands.

### Exercise 4: Computer Vision Integration
Build a Python ROS 2 node that integrates OpenCV for computer vision tasks. Subscribe to camera images, perform real-time processing (e.g., edge detection, color filtering, or object tracking), and publish the results. Include visualization capabilities to display the processed images.