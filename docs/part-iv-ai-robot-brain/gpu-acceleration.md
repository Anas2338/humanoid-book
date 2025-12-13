---
sidebar_position: 4
---

# GPU Acceleration in Robotics

## Overview

GPU (Graphics Processing Unit) acceleration has revolutionized robotics by enabling real-time processing of computationally intensive tasks that were previously infeasible on traditional CPU-based systems. This chapter explores the fundamental concepts, implementation strategies, and practical applications of GPU acceleration in robotics, with a particular focus on how NVIDIA's GPU computing platform enhances robotic capabilities through CUDA, TensorRT, and specialized robotics libraries.

The parallel processing capabilities of GPUs make them ideal for robotics applications that require high-throughput computation of similar operations across large datasets. This includes computer vision tasks such as image processing, object detection, and segmentation; sensor data processing for LIDAR, radar, and other sensors; and AI inference for machine learning models used in perception, planning, and control. GPU acceleration enables robots to process sensor data in real-time, make complex decisions quickly, and execute sophisticated behaviors that would be impossible with CPU-only processing.

Modern robotics applications increasingly rely on GPU acceleration to handle the computational demands of AI-powered systems. From autonomous vehicles processing multiple sensor streams simultaneously to warehouse robots navigating complex environments while manipulating objects, GPU acceleration provides the computational throughput necessary for these demanding applications. The integration of GPU acceleration with robotics frameworks like ROS 2 through specialized packages like Isaac ROS creates powerful platforms for developing next-generation robotic systems.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Understand the principles of GPU computing and parallel processing in robotics
- Implement GPU-accelerated algorithms for robotics applications using CUDA
- Optimize robotics algorithms for GPU execution and memory management
- Integrate GPU acceleration with ROS 2 and robotics frameworks
- Evaluate performance gains from GPU acceleration in robotics tasks
- Design GPU-accelerated perception and control pipelines
- Troubleshoot common GPU acceleration issues in robotics systems

## Key Concepts

### GPU Computing Fundamentals for Robotics

GPU computing concepts essential for robotics applications:

- **Parallel Processing**: SIMD (Single Instruction, Multiple Data) architecture for processing similar operations on multiple data points
- **CUDA Programming**: NVIDIA's parallel computing platform and programming model
- **Memory Hierarchy**: Global, shared, and constant memory for optimal GPU performance
- **Thread Organization**: Blocks, grids, and warps for organizing parallel computations
- **Stream Processing**: Concurrent execution of multiple operations for pipeline efficiency
- **Cooperative Processing**: CPU-GPU collaboration for optimal system performance

### Robotics-Specific GPU Acceleration

GPU applications in robotics systems:

- **Computer Vision**: Real-time image processing, feature extraction, and object detection
- **Sensor Processing**: LIDAR point cloud processing, radar signal processing, and sensor fusion
- **AI Inference**: Neural network inference for perception, planning, and control
- **Path Planning**: Parallel path optimization and collision checking algorithms
- **Physics Simulation**: Parallel physics calculations for simulation and control
- **SLAM**: Simultaneous Localization and Mapping with GPU acceleration

### GPU Memory Management in Robotics

Critical aspects of GPU memory management for robotics:

- **Memory Transfers**: Efficient CPU-GPU data movement to minimize bottlenecks
- **Memory Allocation**: Proper allocation strategies for dynamic robotics applications
- **Memory Reuse**: Techniques to minimize allocation overhead in real-time systems
- **Unified Memory**: Simplified memory management with unified CPU-GPU memory space
- **Memory Bandwidth**: Optimizing data access patterns for maximum throughput
- **Memory Footprint**: Managing GPU memory usage within hardware constraints

### Performance Optimization Strategies

Techniques for maximizing GPU acceleration benefits:

- **Kernel Optimization**: Optimizing GPU kernels for robotics-specific computations
- **Occupancy Maximization**: Ensuring sufficient parallel threads for GPU utilization
- **Memory Coalescing**: Optimizing memory access patterns for bandwidth efficiency
- **Asynchronous Execution**: Overlapping computation and data transfer
- **Multi-GPU Scaling**: Utilizing multiple GPUs for increased computational power
- **Power Management**: Balancing performance with power consumption constraints

## Diagrams and Code

### GPU Acceleration Architecture for Robotics

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Data   │───▶│   CPU (ROS 2)   │───▶│   GPU Compute   │
│   (Camera,      │    │   (Control,     │    │   (CUDA,        │
│   LIDAR, IMU)   │    │   Planning)     │    │   TensorRT)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data          │    │   Task          │    │   Parallel      │
│   Acquisition   │    │   Scheduling    │    │   Processing    │
│   & Preprocessing│   │   & Message     │    │   (Vision,      │
│                 │    │   Handling      │    │   AI, Control)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### CUDA-Accelerated Image Processing Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import cupy as cp  # Use CuPy for GPU operations
import time

class GPUImageProcessor(Node):
    """
    GPU-accelerated image processing node using CuPy.
    Demonstrates how to leverage GPU for real-time image processing.
    """

    def __init__(self):
        super().__init__('gpu_image_processor')

        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.gpu_image_callback,
            10
        )

        self.performance_pub = self.create_publisher(
            Float32,
            '/gpu_processing_time',
            10
        )

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('GPU Image Processor initialized')

    def gpu_image_callback(self, msg):
        """
        Process image using GPU acceleration
        """
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Transfer image to GPU memory
            gpu_image = cp.asarray(cv_image)

            # Perform GPU-accelerated image processing
            processed_gpu = self.gpu_image_processing(gpu_image)

            # Transfer result back to CPU memory
            processed_cpu = cp.asnumpy(processed_gpu)

            # Calculate and publish processing time
            end_time = time.time()
            processing_time = end_time - start_time

            perf_msg = Float32()
            perf_msg.data = processing_time
            self.performance_pub.publish(perf_msg)

            # Track performance statistics
            self.processing_times.append(processing_time)
            self.frame_count += 1

            # Log performance every 100 frames
            if self.frame_count % 100 == 0:
                avg_time = sum(self.processing_times[-100:]) / min(len(self.processing_times), 100)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(f'GPU processing: {avg_time:.4f}s ({fps:.2f} FPS)')

        except Exception as e:
            self.get_logger().error(f'GPU image processing error: {str(e)}')

    def gpu_image_processing(self, gpu_image):
        """
        Perform GPU-accelerated image processing operations
        """
        # Convert to grayscale using GPU
        if gpu_image.ndim == 3:
            gray = 0.299 * gpu_image[:, :, 0] + 0.587 * gpu_image[:, :, 1] + 0.114 * gpu_image[:, :, 2]
        else:
            gray = gpu_image

        # Apply Gaussian blur using GPU
        blurred = self.gpu_gaussian_blur(gray, kernel_size=5)

        # Apply edge detection using GPU
        edges = self.gpu_canny_edge_detection(blurred)

        # Combine results
        result = cp.stack([edges, blurred, cp.zeros_like(edges)], axis=2).astype(cp.uint8)

        return result

    def gpu_gaussian_blur(self, image, kernel_size=5):
        """
        Apply Gaussian blur using GPU operations
        """
        # Create Gaussian kernel
        sigma = kernel_size / 3.0
        kernel_1d = cp.exp(-cp.arange(-kernel_size//2, kernel_size//2 + 1)**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / cp.sum(kernel_1d)

        # Apply separable convolution
        # Horizontal pass
        padded_image = cp.pad(image, ((0, 0), (kernel_size//2, kernel_size//2)), mode='edge')
        blurred_horizontal = cp.zeros_like(image)

        for i in range(kernel_size):
            offset = i - kernel_size//2
            blurred_horizontal += kernel_1d[i] * padded_image[:, kernel_size//2 + offset:padded_image.shape[1] - kernel_size//2 + offset]

        # Vertical pass
        padded_image = cp.pad(blurred_horizontal, ((kernel_size//2, kernel_size//2), (0, 0)), mode='edge')
        blurred_vertical = cp.zeros_like(image)

        for i in range(kernel_size):
            offset = i - kernel_size//2
            blurred_vertical += kernel_1d[i] * padded_image[kernel_size//2 + offset:padded_image.shape[0] - kernel_size//2 + offset, :]

        return blurred_vertical

    def gpu_canny_edge_detection(self, image):
        """
        Simplified Canny edge detection using GPU
        """
        # Compute gradients using Sobel operators
        sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Pad image for convolution
        padded = cp.pad(image, 1, mode='edge')

        # Compute gradients
        grad_x = cp.zeros_like(image)
        grad_y = cp.zeros_like(image)

        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                grad_x[i-1, j-1] = cp.sum(padded[i-1:i+2, j-1:j+2] * sobel_x)
                grad_y[i-1, j-1] = cp.sum(padded[i-1:i+2, j-1:j+2] * sobel_y)

        # Compute gradient magnitude
        magnitude = cp.sqrt(grad_x**2 + grad_y**2)

        # Apply threshold
        threshold = cp.percentile(magnitude, 80)
        edges = cp.where(magnitude > threshold, 255, 0).astype(cp.uint8)

        return edges

def main(args=None):
    rclpy.init(args=args)

    gpu_processor = GPUImageProcessor()

    try:
        rclpy.spin(gpu_processor)
    except KeyboardInterrupt:
        pass
    finally:
        gpu_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### GPU-Accelerated Point Cloud Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32
import numpy as np
import cupy as cp
import struct
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

class GPUPointCloudProcessor(Node):
    """
    GPU-accelerated point cloud processing node.
    Demonstrates how to leverage GPU for LIDAR point cloud operations.
    """

    def __init__(self):
        super().__init__('gpu_pointcloud_processor')

        # Publishers and subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/lidar/points',
            self.gpu_pointcloud_callback,
            10
        )

        self.performance_pub = self.create_publisher(
            Float32,
            '/gpu_pointcloud_processing_time',
            10
        )

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        self.get_logger().info('GPU Point Cloud Processor initialized')

    def gpu_pointcloud_callback(self, msg):
        """
        Process point cloud using GPU acceleration
        """
        start_time = time.time()

        try:
            # Convert PointCloud2 to structured numpy array
            points_list = []
            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])

            if not points_list:
                return

            # Convert to numpy array and transfer to GPU
            cpu_points = np.array(points_list, dtype=np.float32)
            gpu_points = cp.asarray(cpu_points)

            # Perform GPU-accelerated point cloud processing
            processed_points = self.gpu_pointcloud_processing(gpu_points)

            # Calculate and publish processing time
            end_time = time.time()
            processing_time = end_time - start_time

            perf_msg = Float32()
            perf_msg.data = processing_time
            self.performance_pub.publish(perf_msg)

            # Track performance statistics
            self.processing_times.append(processing_time)
            self.frame_count += 1

            # Log performance every 50 frames
            if self.frame_count % 50 == 0:
                avg_time = sum(self.processing_times[-50:]) / min(len(self.processing_times), 50)
                self.get_logger().info(f'GPU point cloud processing: {avg_time:.4f}s for {len(cpu_points)} points')

        except Exception as e:
            self.get_logger().error(f'GPU point cloud processing error: {str(e)}')

    def gpu_pointcloud_processing(self, gpu_points):
        """
        Perform GPU-accelerated point cloud processing operations
        """
        # Example: Ground plane removal using RANSAC-like approach
        processed_points = self.gpu_ground_removal(gpu_points)

        # Example: Point cloud downsampling
        downsampled_points = self.gpu_voxel_grid_filter(processed_points, voxel_size=0.1)

        # Example: Outlier removal
        filtered_points = self.gpu_statistical_outlier_removal(downsampled_points)

        return filtered_points

    def gpu_ground_removal(self, gpu_points, ground_height_threshold=-0.5, max_iterations=100):
        """
        Remove ground plane using GPU-accelerated RANSAC-like approach
        """
        # Filter out points that are likely ground (z < threshold)
        non_ground_mask = gpu_points[:, 2] > ground_height_threshold
        non_ground_points = gpu_points[non_ground_mask]

        return non_ground_points

    def gpu_voxel_grid_filter(self, gpu_points, voxel_size=0.1):
        """
        Apply voxel grid filter using GPU
        """
        if gpu_points.size == 0:
            return gpu_points

        # Calculate voxel indices
        min_vals = cp.min(gpu_points, axis=0)
        max_vals = cp.max(gpu_points, axis=0)

        # Calculate voxel grid dimensions
        dims = cp.ceil((max_vals - min_vals) / voxel_size).astype(cp.int32)

        # Calculate voxel indices for each point
        voxel_indices = cp.floor((gpu_points - min_vals) / voxel_size).astype(cp.int32)

        # Convert 3D indices to 1D hash
        hash_vals = (voxel_indices[:, 0] * dims[1] * dims[2] +
                    voxel_indices[:, 1] * dims[2] +
                    voxel_indices[:, 2])

        # Get unique voxel indices and corresponding points
        unique_hashes, indices = cp.unique(hash_vals, return_index=True)

        return gpu_points[indices]

    def gpu_statistical_outlier_removal(self, gpu_points, k=20, std_dev_multiplier=2.0):
        """
        Remove statistical outliers using GPU
        """
        if len(gpu_points) < k + 1:
            return gpu_points

        # Calculate distances to k nearest neighbors for each point
        n_points = len(gpu_points)
        distances = cp.zeros((n_points, k))

        for i in range(n_points):
            # Calculate distances to all other points
            diff = gpu_points - gpu_points[i]
            all_distances = cp.sqrt(cp.sum(diff**2, axis=1))

            # Get k nearest neighbors (excluding self)
            sorted_indices = cp.argsort(all_distances)
            neighbor_indices = sorted_indices[1:k+1]  # Exclude self (index 0)
            distances[i, :] = all_distances[neighbor_indices]

        # Calculate mean distance for each point
        mean_distances = cp.mean(distances, axis=1)

        # Calculate global mean and std of mean distances
        global_mean = cp.mean(mean_distances)
        global_std = cp.std(mean_distances)

        # Filter points based on statistical criteria
        threshold = global_mean + std_dev_multiplier * global_std
        inlier_mask = mean_distances <= threshold

        return gpu_points[inlier_mask]

def main(args=None):
    import time  # Import time module

    rclpy.init(args=args)

    gpu_processor = GPUPointCloudProcessor()

    try:
        rclpy.spin(gpu_processor)
    except KeyboardInterrupt:
        pass
    finally:
        gpu_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### GPU-Accelerated AI Inference for Robotics

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import cupy as cp
import time

class GPUAIInferenceNode(Node):
    """
    GPU-accelerated AI inference node for robotics applications.
    Demonstrates how to use GPU for neural network inference in robotics.
    """

    def __init__(self):
        super().__init__('gpu_ai_inference')

        self.cv_bridge = CvBridge()

        # Publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/gpu_detections', 10)
        self.performance_pub = self.create_publisher(Float32, '/gpu_inference_time', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.gpu_inference_callback,
            10
        )

        # Simulated AI model parameters (in real implementation, this would be a loaded model)
        self.model_params = {
            'input_shape': (3, 224, 224),
            'num_classes': 80,
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4
        }

        # Initialize GPU model simulator
        self.gpu_model = self.initialize_gpu_model()

        # Performance tracking
        self.inference_times = []
        self.frame_count = 0

        self.get_logger().info('GPU AI Inference Node initialized')

    def initialize_gpu_model(self):
        """
        Initialize GPU model simulator
        In real implementation, this would load a TensorRT model
        """
        self.get_logger().info('Initializing GPU-accelerated AI model')
        return {
            'initialized': True,
            'gpu_accelerated': True,
            'model_type': 'YOLOv8',
            'input_resolution': (640, 640)
        }

    def gpu_inference_callback(self, msg):
        """
        Perform GPU-accelerated AI inference on image
        """
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image for model input
            input_tensor = self.preprocess_image(cv_image)

            # Perform GPU-accelerated inference
            detections = self.gpu_inference(input_tensor)

            # Post-process detections
            filtered_detections = self.postprocess_detections(detections)

            # Publish results
            detection_msg = self.create_detection_message(filtered_detections, msg.header)
            self.detection_pub.publish(detection_msg)

            # Calculate and publish performance metrics
            end_time = time.time()
            inference_time = end_time - start_time

            perf_msg = Float32()
            perf_msg.data = inference_time
            self.performance_pub.publish(perf_msg)

            # Track performance
            self.inference_times.append(inference_time)
            self.frame_count += 1

            # Log performance statistics
            if self.frame_count % 25 == 0:
                avg_time = sum(self.inference_times[-25:]) / min(len(self.inference_times), 25)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(f'GPU inference: {avg_time:.4f}s ({fps:.2f} FPS), {len(filtered_detections)} detections')

        except Exception as e:
            self.get_logger().error(f'GPU inference error: {str(e)}')

    def preprocess_image(self, image):
        """
        Preprocess image for model input using GPU
        """
        # Resize image to model input size
        target_h, target_w = self.gpu_model['input_resolution']
        resized_image = self.gpu_resize(image, target_h, target_w)

        # Normalize image
        normalized_image = (resized_image / 255.0).astype(cp.float32)

        # Convert to NCHW format (batch, channels, height, width)
        if normalized_image.ndim == 3:
            normalized_image = cp.transpose(normalized_image, (2, 0, 1))

        return normalized_image

    def gpu_resize(self, image, target_h, target_w):
        """
        Resize image using GPU operations (simplified implementation)
        """
        # In real implementation, this would use optimized GPU resize
        # For this example, we'll use CuPy operations
        orig_h, orig_w = image.shape[:2]

        # Calculate scaling factors
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w

        # Create coordinate grids
        y_coords = cp.linspace(0, orig_h - 1, target_h)
        x_coords = cp.linspace(0, orig_w - 1, target_w)

        # Perform bilinear interpolation using GPU
        resized = cp.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

        for c in range(image.shape[2]):
            # Simplified nearest neighbor for demonstration
            y_indices = cp.clip(cp.round(y_coords).astype(cp.int32), 0, orig_h - 1)
            x_indices = cp.clip(cp.round(x_coords).astype(cp.int32), 0, orig_w - 1)

            for i, y_idx in enumerate(y_indices):
                for j, x_idx in enumerate(x_indices):
                    resized[i, j, c] = image[y_idx, x_idx, c]

        return resized

    def gpu_inference(self, input_tensor):
        """
        Perform GPU-accelerated inference
        In real implementation, this would call a TensorRT model
        """
        # Simulate GPU inference with realistic timing
        # Transfer input to GPU if not already there
        if not isinstance(input_tensor, cp.ndarray):
            input_tensor = cp.asarray(input_tensor)

        # Simulate inference by creating realistic detection outputs
        # In real implementation, this would be the actual model forward pass
        batch_size, channels, height, width = input_tensor.shape

        # Simulate detection outputs (bounding boxes, scores, class predictions)
        # This would normally come from the neural network
        num_detections = 50  # Simulate 50 potential detections
        detections = {
            'boxes': cp.random.uniform(0, 1, (num_detections, 4)).astype(cp.float32),  # x1, y1, x2, y2
            'scores': cp.random.uniform(0.1, 1.0, (num_detections,)).astype(cp.float32),
            'classes': cp.random.randint(0, self.model_params['num_classes'], (num_detections,)).astype(cp.int32)
        }

        # Add some realistic constraints to simulated detections
        # Ensure x2 > x1 and y2 > y1
        detections['boxes'][:, 2] = cp.maximum(detections['boxes'][:, 0] + 0.05, detections['boxes'][:, 2])
        detections['boxes'][:, 3] = cp.maximum(detections['boxes'][:, 1] + 0.05, detections['boxes'][:, 3])

        # Simulate processing time
        time.sleep(0.01)  # Simulate 10ms inference time

        return detections

    def postprocess_detections(self, detections):
        """
        Post-process detections with GPU acceleration
        """
        # Filter by confidence threshold
        conf_mask = detections['scores'] > self.model_params['confidence_threshold']
        filtered_boxes = detections['boxes'][conf_mask]
        filtered_scores = detections['scores'][conf_mask]
        filtered_classes = detections['classes'][conf_mask]

        # Apply Non-Maximum Suppression (simplified GPU implementation)
        if len(filtered_boxes) > 0:
            nms_indices = self.gpu_non_max_suppression(
                filtered_boxes,
                filtered_scores,
                self.model_params['nms_threshold']
            )

            final_boxes = filtered_boxes[nms_indices]
            final_scores = filtered_scores[nms_indices]
            final_classes = filtered_classes[nms_indices]
        else:
            final_boxes = cp.array([], dtype=cp.float32).reshape(0, 4)
            final_scores = cp.array([], dtype=cp.float32)
            final_classes = cp.array([], dtype=cp.int32)

        # Convert to list of dictionaries for easier handling
        result = []
        for i in range(len(final_boxes)):
            result.append({
                'bbox': final_boxes[i].get(),  # Transfer to CPU
                'score': float(final_scores[i]),
                'class_id': int(final_classes[i])
            })

        return result

    def gpu_non_max_suppression(self, boxes, scores, threshold):
        """
        GPU-accelerated Non-Maximum Suppression
        """
        if len(boxes) == 0:
            return cp.array([], dtype=cp.int32)

        # Sort by scores in descending order
        sorted_indices = cp.argsort(scores)[::-1]
        boxes_sorted = boxes[sorted_indices]
        scores_sorted = scores[sorted_indices]

        # Calculate areas
        areas = (boxes_sorted[:, 2] - boxes_sorted[:, 0]) * (boxes_sorted[:, 3] - boxes_sorted[:, 1])

        keep = []
        while len(boxes_sorted) > 0:
            # Keep the box with highest score
            keep.append(sorted_indices[len(keep)])

            if len(boxes_sorted) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = cp.maximum(boxes_sorted[0, 0], boxes_sorted[1:, 0])
            yy1 = cp.maximum(boxes_sorted[0, 1], boxes_sorted[1:, 1])
            xx2 = cp.minimum(boxes_sorted[0, 2], boxes_sorted[1:, 2])
            yy2 = cp.minimum(boxes_sorted[0, 3], boxes_sorted[1:, 3])

            w = cp.maximum(0.0, xx2 - xx1)
            h = cp.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[0] + areas[1:] - intersection)

            # Keep boxes with IoU less than threshold
            keep_indices = cp.where(iou < threshold)[0] + 1
            boxes_sorted = boxes_sorted[keep_indices]
            scores_sorted = scores_sorted[keep_indices]
            sorted_indices = sorted_indices[keep_indices]

        return cp.array(keep, dtype=cp.int32)

    def create_detection_message(self, detections, header):
        """
        Create Detection2DArray message from detection results
        """
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            detection_msg = Detection2D()
            detection_msg.header = header

            # Set bounding box
            x1, y1, x2, y2 = detection['bbox']
            detection_msg.bbox.size_x = float(x2 - x1)
            detection_msg.bbox.size_y = float(y2 - y1)

            # Set center point
            detection_msg.bbox.center.x = float((x1 + x2) / 2)
            detection_msg.bbox.center.y = float((y1 + y2) / 2)

            # Add classification result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(detection['class_id'])
            hypothesis.hypothesis.score = detection['score']
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        return detection_array

def main(args=None):
    rclpy.init(args=args)

    ai_inference_node = GPUAIInferenceNode()

    try:
        rclpy.spin(ai_inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        ai_inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Labs and Exercises

### Exercise 1: GPU Memory Management Optimization
Implement a GPU-accelerated robotics application that efficiently manages GPU memory for real-time processing. Focus on minimizing memory allocation overhead and optimizing data transfer between CPU and GPU. Measure the performance improvements achieved through proper memory management.

### Exercise 2: Parallel Path Planning Algorithm
Develop a GPU-accelerated path planning algorithm that can evaluate multiple potential paths simultaneously. Compare the performance of the GPU-accelerated version with a CPU-only implementation and analyze the scalability with increasing problem complexity.

### Exercise 3: Multi-Sensor Fusion with GPU Acceleration
Create a multi-sensor fusion pipeline that processes data from cameras, LIDAR, and IMU sensors using GPU acceleration. Implement sensor data synchronization and fusion algorithms optimized for GPU execution.

### Exercise 4: GPU-Accelerated SLAM Implementation
Implement a GPU-accelerated SLAM (Simultaneous Localization and Mapping) system that can process sensor data in real-time. Focus on GPU optimization of key SLAM components such as feature extraction, matching, and map building.

## Summary

This chapter explored GPU acceleration in robotics, demonstrating how NVIDIA's GPU computing platform can significantly enhance robotic capabilities through parallel processing. We covered the fundamentals of GPU computing for robotics, implemented GPU-accelerated image processing and point cloud operations, and showed how to leverage GPU acceleration for AI inference in robotics applications. The examples illustrated how GPU acceleration enables robots to process large amounts of sensor data in real-time, making complex AI-powered behaviors possible. As robotics applications become more sophisticated and data-intensive, GPU acceleration will continue to play a critical role in enabling real-time performance for perception, planning, and control tasks.