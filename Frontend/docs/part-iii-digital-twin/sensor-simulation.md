---
sidebar_position: 4
---

# Sensor Simulation

## Overview

Sensor simulation plays a critical role in digital twin environments for robotics, providing realistic virtual sensory data that closely matches real-world sensor characteristics. This chapter explores the principles and implementation of sensor simulation in digital twin environments, covering various sensor modalities including cameras, LIDAR, radar, IMU, and other perception systems. Accurate sensor simulation is essential for effective sim-to-real transfer, enabling robotic systems to be trained and validated in virtual environments before deployment on real hardware.

Modern robotics systems rely heavily on sensor data for perception, localization, mapping, and navigation. The quality of sensor simulation directly impacts the effectiveness of robot learning, testing, and validation in digital twin environments. Realistic sensor simulation must account for various physical phenomena including noise, distortion, latency, and environmental effects that influence sensor performance in real-world scenarios.

Sensor simulation in digital twins involves modeling both the physical sensing process and the electronic signal processing that occurs in real sensors. This includes modeling of sensor physics, electronics, and environmental factors that affect sensor performance. For humanoid robots, sensor simulation must account for the complex interactions between multiple sensors and the robot's dynamic movement, which can introduce additional challenges like motion blur, vibration effects, and occlusion.

The ultimate goal of sensor simulation is to minimize the "reality gap" between virtual and real sensors, making it possible to transfer learned behaviors and algorithms from simulation to real robots with minimal adaptation. This requires careful calibration of simulation parameters to match real sensor characteristics and validation of simulation output against real sensor data under various environmental conditions.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Model and simulate various sensor types used in robotics applications
- Characterize and replicate real sensor noise, distortion, and other imperfections
- Implement realistic sensor simulation in physics engines like Gazebo and Unity
- Validate sensor simulation accuracy against real sensor data
- Account for environmental factors affecting sensor performance in simulation
- Optimize sensor simulation for computational efficiency while maintaining accuracy
- Design sensor fusion algorithms that work effectively in both simulation and reality

## Key Concepts

### Camera and Vision Sensor Simulation

Camera simulation involves modeling optical properties and image formation processes:

- **Intrinsic Parameters**: Focal length, principal point, and lens distortion coefficients
- **Extrinsic Parameters**: Camera position, orientation, and mounting configuration
- **Image Formation**: Modeling of pinhole projection, lens effects, and image sampling
- **Noise Models**: Thermal noise, quantization noise, and photon shot noise
- **Dynamic Effects**: Motion blur, rolling shutter, and temporal artifacts

### Range Sensor Simulation (LIDAR, Sonar, Radar)

Range sensors measure distances to objects in the environment:

- **Ray Casting**: Physics-based simulation of laser beams and reflections
- **Point Cloud Generation**: Creating realistic 3D point cloud data
- **Multi-return Processing**: Modeling of beam divergence and partial returns
- **Environmental Factors**: Weather effects, surface reflectance, and interference
- **Temporal Characteristics**: Scanning patterns and measurement timing

### Inertial Measurement Unit (IMU) Simulation

IMUs provide information about robot motion and orientation:

- **Accelerometer Modeling**: Linear acceleration with gravity and noise
- **Gyroscope Modeling**: Angular velocity with drift and bias
- **Magnetometer Modeling**: Magnetic field sensing with disturbances
- **Bias and Drift**: Time-dependent sensor inaccuracies
- **Temperature Effects**: Performance variation with temperature

### Force and Tactile Sensor Simulation

Force and tactile sensors provide information about physical interactions:

- **Force/Torque Sensors**: Six-axis force/torque measurements
- **Tactile Sensors**: Contact detection and pressure distribution
- **Contact Modeling**: Accurate simulation of physical interactions
- **Haptic Feedback**: Modeling of touch and force sensations
- **Calibration**: Accounting for sensor offsets and scaling factors

### Environmental and Cross-Modal Effects

Sensors are affected by environmental conditions and other system components:

- **Weather Simulation**: Rain, fog, dust, and atmospheric effects
- **Illumination Modeling**: Day/night cycles, shadows, and glare
- **Multipath Interference**: Signal reflection and interference effects
- **Electromagnetic Interference**: Cross-talk between sensors and systems
- **Vibration and Shock**: Mechanical effects on sensor performance

### Sensor Fusion and Integration

Combining data from multiple sensors to improve perception:

- **Kalman Filtering**: Optimal estimation from multiple sensor sources
- **Particle Filtering**: Non-linear estimation for complex sensor models
- **Data Association**: Matching sensor observations across modalities
- **Temporal Synchronization**: Aligning sensor data in time
- **Uncertainty Propagation**: Tracking confidence in fused estimates

## Diagrams and Code

### Sensor Simulation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Digital Twin Environment                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Camera        │  │   LIDAR         │  │   IMU           │  │
│  │   Simulation    │  │   Simulation    │  │   Simulation    │  │
│  │   (Visual)      │  │   (Range)       │  │   (Inertial)    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                      │                       │        │
│         ▼                      ▼                       ▼        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Noise &       │  │   Noise &       │  │   Noise &       │  │
│  │   Distortion    │  │   Distortion    │  │   Drift Models  │  │
│  │   Models        │  │   Models        │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│         │                      │                       │        │
│         └──────────────────────┼───────────────────────┘        │
│                                │                                │
│         ┌─────────────────────────────────────────────────┐      │
│         │           Sensor Fusion Module                  │      │
│         │  ┌─────────────────┐  ┌─────────────────┐      │      │
│         │  │   Estimation    │  │   Validation    │      │      │
│         │  │   Algorithms    │  │   Components    │      │      │
│         │  └─────────────────┘  └─────────────────┘      │      │
│         └─────────────────────────────────────────────────┘      │
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────┐
│        Realistic Sensor Data Output                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Image Data    │  │   Point Cloud   │  │   IMU Data      │  │
│  │   (ROS msgs)    │  │   (ROS msgs)    │  │   (ROS msgs)    │  │
│  │   (sensor_msgs/ │  │   (sensor_msgs/ │  │   (sensor_msgs/ │  │
│  │    Image)       │  │    PointCloud)  │  │    Imu)         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Camera Sensor Simulation Example

```python
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional
import random

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2

class CameraSimulator:
    """
    Advanced camera simulator with realistic noise and distortion models.
    """

    def __init__(self, width: int, height: int, intrinsics: CameraIntrinsics):
        self.width = width
        self.height = height
        self.intrinsics = intrinsics

        # Noise parameters
        self.readout_noise = 2.0  # electrons
        self.dark_current = 0.01  # electrons/pixel/sec
        self.quantization_noise = 0.2  # AD conversion noise
        self.photo_response_nonuniformity = 0.02  # PRNU (2%)

        # Create camera matrix
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ])

        # Create distortion coefficients
        self.dist_coeffs = np.array([
            intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2, 0.0
        ])

    def add_photon_noise(self, image: np.ndarray, exposure_time: float = 0.033) -> np.ndarray:
        """
        Add photon shot noise to the image based on exposure time.
        """
        # Convert to photons (simplified model)
        photons = image * 1000  # Scale to reasonable photon count

        # Add Poisson noise (photon shot noise)
        noisy_photons = np.random.poisson(photons)

        # Convert back to image units
        noisy_image = noisy_photons / 1000.0

        return noisy_image

    def add_readout_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add readout noise (thermal noise from electronics).
        """
        readout_noise = np.random.normal(0, self.readout_noise, image.shape)
        return image + readout_noise

    def add_dark_current_noise(self, image: np.ndarray, exposure_time: float) -> np.ndarray:
        """
        Add dark current noise based on exposure time.
        """
        dark_noise = np.random.poisson(self.dark_current * exposure_time, image.shape)
        return image + dark_noise

    def add_quantization_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add quantization noise from analog-to-digital conversion.
        """
        # Quantize to 8-bit then add noise
        quantized = np.round(image * 255) / 255.0
        noise = np.random.uniform(-self.quantization_noise, self.quantization_noise, image.shape)
        return np.clip(quantized + noise, 0, 1)

    def apply_lens_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lens distortion using OpenCV.
        """
        h, w = image.shape[:2]

        # Create map for undistortion (but we'll use it to distort)
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, None, self.K, (w, h), cv2.CV_32FC1
        )

        # Apply distortion
        distorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return distorted

    def add_photo_response_nonuniformity(self, image: np.ndarray) -> np.ndarray:
        """
        Add Photo Response Non-Uniformity (PRNU) noise.
        """
        prnu = np.random.normal(1.0, self.photo_response_nonuniformity, image.shape)
        return image * prnu

    def simulate_frame(self, scene_image: np.ndarray, exposure_time: float = 0.033) -> np.ndarray:
        """
        Simulate a complete camera frame with all noise sources.
        """
        # Apply lens distortion first
        distorted_image = self.apply_lens_distortion(scene_image)

        # Add various noise sources
        noisy_image = self.add_photon_noise(distorted_image, exposure_time)
        noisy_image = self.add_dark_current_noise(noisy_image, exposure_time)
        noisy_image = self.add_readout_noise(noisy_image)
        noisy_image = self.add_photo_response_nonuniformity(noisy_image)
        noisy_image = self.add_quantization_noise(noisy_image)

        # Ensure valid range
        return np.clip(noisy_image, 0, 1)

    def simulate_stereo_pair(self, left_scene: np.ndarray, right_scene: np.ndarray,
                            baseline: float, exposure_time: float = 0.033) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate stereo camera pair with parallax effect.
        """
        left_image = self.simulate_frame(left_scene, exposure_time)

        # Create right camera simulator with offset
        right_intrinsics = CameraIntrinsics(
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx - baseline * self.intrinsics.fx,  # Adjust for baseline
            cy=self.intrinsics.cy,
            k1=self.intrinsics.k1,
            k2=self.intrinsics.k2,
            p1=self.intrinsics.p1,
            p2=self.intrinsics.p2
        )

        right_simulator = CameraSimulator(self.width, self.height, right_intrinsics)
        right_image = right_simulator.simulate_frame(right_scene, exposure_time)

        return left_image, right_image

# Example usage and demonstration
def demonstrate_camera_simulation():
    """
    Demonstrate the camera simulation with a synthetic scene.
    """
    print("=== Camera Sensor Simulation Demonstration ===\n")

    # Create synthetic scene (a simple gradient with some shapes)
    width, height = 640, 480
    scene = np.zeros((height, width, 3), dtype=np.float32)

    # Create a gradient background
    for i in range(height):
        for j in range(width):
            scene[i, j, 0] = j / width  # Red gradient
            scene[i, j, 1] = i / height  # Green gradient
            scene[i, j, 2] = 0.5  # Blue constant

    # Add some shapes
    cv2.circle(scene, (width//2, height//2), 50, (1, 1, 1), -1)  # White circle
    cv2.rectangle(scene, (100, 100), (200, 200), (0, 1, 1), -1)  # Cyan square

    # Create camera simulator
    intrinsics = CameraIntrinsics(
        fx=500, fy=500, cx=width//2, cy=height//2,
        k1=-0.1, k2=0.05  # Some distortion
    )

    camera_sim = CameraSimulator(width, height, intrinsics)

    # Simulate clean vs noisy images
    clean_image = scene
    noisy_image = camera_sim.simulate_frame(clean_image)

    print(f"Original scene shape: {clean_image.shape}")
    print(f"Simulated noisy image shape: {noisy_image.shape}")
    print(f"Mean pixel value - Original: {np.mean(clean_image):.3f}, Noisy: {np.mean(noisy_image):.3f}")
    print(f"Std dev - Original: {np.std(clean_image):.3f}, Noisy: {np.std(noisy_image):.3f}")

    return clean_image, noisy_image

if __name__ == "__main__":
    clean_img, noisy_img = demonstrate_camera_simulation()
```

### LIDAR Sensor Simulation Example

```python
import numpy as np
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class LIDARParams:
    """LIDAR sensor parameters"""
    fov_horizontal: float = 2 * math.pi  # Horizontal field of view in radians
    fov_vertical: float = 0.1  # Vertical field of view in radians (for 2D lidar, use 0.1)
    resolution_horizontal: float = 0.01  # Angular resolution in radians
    resolution_vertical: float = 0.01  # Vertical angular resolution
    range_min: float = 0.1  # Minimum detectable range (m)
    range_max: float = 30.0  # Maximum detectable range (m)
    noise_std: float = 0.02  # Range measurement noise standard deviation (m)
    beam_divergence: float = 0.003  # Beam divergence (radians)

class LIDARSimulator:
    """
    Advanced LIDAR simulator with realistic physics and noise models.
    """

    def __init__(self, params: LIDARParams):
        self.params = params

        # Calculate number of rays
        self.num_rays_h = int(self.params.fov_horizontal / self.params.resolution_horizontal)
        self.num_rays_v = int(self.params.fov_vertical / self.params.resolution_vertical)

        if self.params.fov_vertical < 0.05:  # 2D LIDAR
            self.num_rays_v = 1
            self.params.fov_vertical = 0.01  # Small value to avoid division by zero

    def cast_ray(self, origin: np.ndarray, direction: np.ndarray,
                 obstacles: List[Tuple[np.ndarray, float]], max_range: float) -> float:
        """
        Cast a single ray and return the distance to the nearest obstacle.
        """
        min_distance = max_range

        for center, radius in obstacles:
            # Ray-sphere intersection
            oc = origin - center
            a = np.dot(direction, direction)
            b = 2.0 * np.dot(oc, direction)
            c = np.dot(oc, oc) - radius * radius

            discriminant = b * b - 4 * a * c

            if discriminant >= 0:
                t1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
                t2 = (-b + math.sqrt(discriminant)) / (2.0 * a)

                if t1 > 0 and t1 < min_distance:
                    min_distance = t1
                elif t2 > 0 and t2 < min_distance:
                    min_distance = t2

        return min_distance if min_distance < max_range else float('inf')

    def simulate_scan(self, robot_pose: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Simulate a complete LIDAR scan.
        """
        ranges = np.full(self.num_rays_h, self.params.range_max, dtype=np.float32)

        # Robot position and orientation
        robot_pos = robot_pose[:2]  # x, y
        robot_yaw = robot_pose[2]   # theta

        # Generate ray directions
        angles = np.linspace(
            -self.params.fov_horizontal/2,
            self.params.fov_horizontal/2,
            self.num_rays_h
        )

        for i, angle in enumerate(angles):
            # Rotate ray by robot orientation
            world_angle = angle + robot_yaw

            # Ray direction in world coordinates
            direction = np.array([math.cos(world_angle), math.sin(world_angle)])

            # Cast ray
            distance = self.cast_ray(robot_pos, direction, obstacles, self.params.range_max)

            # Add noise
            if distance < self.params.range_max:
                distance += np.random.normal(0, self.params.noise_std)
                distance = max(self.params.range_min, min(distance, self.params.range_max))

            ranges[i] = distance

        return ranges

    def add_cosine_error(self, ranges: np.ndarray) -> np.ndarray:
        """
        Add cosine error based on surface angle relative to beam.
        """
        # This is a simplified model - in reality, this would depend on surface normals
        # and beam characteristics
        corrected_ranges = ranges.copy()

        for i in range(len(ranges)):
            if ranges[i] < self.params.range_max:
                # Cosine error is more significant at larger angles
                angle_factor = 1.0 + 0.01 * (i / len(ranges) - 0.5)  # Simplified model
                corrected_ranges[i] *= angle_factor

        return corrected_ranges

    def add_multipath_error(self, ranges: np.ndarray) -> np.ndarray:
        """
        Add multipath error that can occur with reflective surfaces.
        """
        perturbed_ranges = ranges.copy()

        for i in range(len(ranges)):
            if ranges[i] < self.params.range_max and np.random.random() < 0.05:  # 5% chance of multipath
                # Multipath can cause early or late returns
                multipath_effect = np.random.uniform(-0.5, 0.5)
                perturbed_ranges[i] = max(self.params.range_min,
                                         min(perturbed_ranges[i] + multipath_effect, self.params.range_max))

        return perturbed_ranges

    def simulate_noisy_scan(self, robot_pose: np.ndarray, obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Simulate a LIDAR scan with all realistic effects.
        """
        # Base scan
        ranges = self.simulate_scan(robot_pose, obstacles)

        # Add various error sources
        ranges = self.add_cosine_error(ranges)
        ranges = self.add_multipath_error(ranges)

        return ranges

def visualize_lidar_scan(ranges: np.ndarray, title: str = "LIDAR Scan"):
    """
    Visualize LIDAR scan data.
    """
    angles = np.linspace(-math.pi, math.pi, len(ranges))

    # Convert to Cartesian coordinates
    x_coords = ranges * np.cos(angles)
    y_coords = ranges * np.sin(angles)

    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, s=1, c='blue', alpha=0.6)
    plt.scatter(0, 0, c='red', s=100, label='Robot', zorder=5)  # Robot at origin
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def demonstrate_lidar_simulation():
    """
    Demonstrate the LIDAR simulation with a synthetic environment.
    """
    print("=== LIDAR Sensor Simulation Demonstration ===\n")

    # Create LIDAR parameters
    lidar_params = LIDARParams(
        fov_horizontal=2 * math.pi,  # 360 degree
        resolution_horizontal=0.01745,  # ~1 degree resolution
        range_max=20.0,
        noise_std=0.02
    )

    lidar_sim = LIDARSimulator(lidar_params)

    # Define some obstacles (center position, radius)
    obstacles = [
        (np.array([5.0, 0.0]), 1.0),   # Cylinder at (5,0) with radius 1
        (np.array([-3.0, 4.0]), 0.8),  # Cylinder at (-3,4) with radius 0.8
        (np.array([2.0, -3.0]), 1.2),  # Cylinder at (2,-3) with radius 1.2
        (np.array([0.0, 7.0]), 1.5),   # Cylinder at (0,7) with radius 1.5
    ]

    # Robot pose (x, y, theta)
    robot_pose = np.array([0.0, 0.0, 0.0])

    # Simulate clean and noisy scans
    clean_scan = lidar_sim.simulate_scan(robot_pose, obstacles)
    noisy_scan = lidar_sim.simulate_noisy_scan(robot_pose, obstacles)

    print(f"Number of LIDAR rays: {len(clean_scan)}")
    print(f"Clean scan - Min: {np.min(clean_scan):.2f}m, Max: {np.max(clean_scan):.2f}m")
    print(f"Noisy scan - Min: {np.min(noisy_scan):.2f}m, Max: {np.max(noisy_scan):.2f}m")
    print(f"Mean difference: {np.mean(np.abs(clean_scan - noisy_scan)):.3f}m")

    # Visualize the scans
    try:
        visualize_lidar_scan(clean_scan, "Clean LIDAR Scan")
        visualize_lidar_scan(noisy_scan, "Noisy LIDAR Scan (with realistic effects)")
    except ImportError:
        print("Matplotlib not available, skipping visualization.")

    return clean_scan, noisy_scan

if __name__ == "__main__":
    clean_scan, noisy_scan = demonstrate_lidar_simulation()
```

### IMU Sensor Simulation Example

```python
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple

@dataclass
class IMUParams:
    """IMU sensor parameters"""
    # Accelerometer parameters
    acc_bias: np.ndarray = None  # Bias vector [x, y, z]
    acc_noise_density: float = 80e-6  # m/s^2 / sqrt(Hz)
    acc_random_walk: float = 2.0e-4  # m/s^2 / sqrt(s)
    acc_scale_factor_error: float = 0.001  # Scale factor error (unitless)

    # Gyroscope parameters
    gyro_bias: np.ndarray = None  # Bias vector [x, y, z]
    gyro_noise_density: float = 0.5e-3  # rad/s / sqrt(Hz)
    gyro_random_walk: float = 0.001  # rad/s / sqrt(s)
    gyro_scale_factor_error: float = 0.001  # Scale factor error (unitless)

    # Magnetometer parameters (if present)
    mag_bias: np.ndarray = None  # Bias vector [x, y, z]
    mag_noise_density: float = 100e-9  # Tesla / sqrt(Hz)

    # Sampling parameters
    sampling_rate: float = 100.0  # Hz

class IMUSimulator:
    """
    Advanced IMU simulator with realistic noise and bias models.
    """

    def __init__(self, params: IMUParams):
        self.params = params

        # Initialize bias values if not provided
        if self.params.acc_bias is None:
            self.params.acc_bias = np.random.normal(0, 0.01, 3)  # Small initial bias
        if self.params.gyro_bias is None:
            self.params.gyro_bias = np.random.normal(0, 0.001, 3)  # Small initial bias
        if self.params.mag_bias is None:
            self.params.mag_bias = np.random.normal(0, 0.1, 3)  # Small initial bias

        # Initialize random walk values
        self.acc_bias_drift = np.zeros(3)
        self.gyro_bias_drift = np.zeros(3)
        self.mag_bias_drift = np.zeros(3)

        # Time step
        self.dt = 1.0 / self.params.sampling_rate

    def simulate_accelerometer(self, true_acc: np.ndarray, temperature: float = 25.0) -> np.ndarray:
        """
        Simulate accelerometer measurements with all error sources.
        """
        # Start with true acceleration
        measurement = true_acc.copy()

        # Add scale factor error
        scale_errors = np.array([1.0 + self.params.acc_scale_factor_error * np.random.normal(0, 0.1) for _ in range(3)])
        measurement *= scale_errors

        # Add bias (including temperature effects)
        temperature_effect = 0.0001 * (temperature - 25.0)  # Temperature coefficient
        bias = self.params.acc_bias + self.acc_bias_drift + temperature_effect
        measurement += bias

        # Add noise (white noise based on noise density)
        noise_std = self.params.acc_noise_density / np.sqrt(2 * self.dt)
        noise = np.random.normal(0, noise_std, 3)
        measurement += noise

        return measurement

    def simulate_gyroscope(self, true_omega: np.ndarray, temperature: float = 25.0) -> np.ndarray:
        """
        Simulate gyroscope measurements with all error sources.
        """
        # Start with true angular velocity
        measurement = true_omega.copy()

        # Add scale factor error
        scale_errors = np.array([1.0 + self.params.gyro_scale_factor_error * np.random.normal(0, 0.1) for _ in range(3)])
        measurement *= scale_errors

        # Add bias (including temperature effects)
        temperature_effect = 0.00001 * (temperature - 25.0)  # Temperature coefficient
        bias = self.params.gyro_bias + self.gyro_bias_drift + temperature_effect
        measurement += bias

        # Add noise (white noise based on noise density)
        noise_std = self.params.gyro_noise_density / np.sqrt(2 * self.dt)
        noise = np.random.normal(0, noise_std, 3)
        measurement += noise

        return measurement

    def simulate_magnetometer(self, true_mag: np.ndarray, temperature: float = 25.0) -> np.ndarray:
        """
        Simulate magnetometer measurements with all error sources.
        """
        # Start with true magnetic field
        measurement = true_mag.copy()

        # Add bias (including temperature effects)
        temperature_effect = 0.001 * (temperature - 25.0)  # Temperature coefficient
        bias = self.params.mag_bias + self.mag_bias_drift + temperature_effect
        measurement += bias

        # Add noise (white noise based on noise density)
        noise_std = self.params.mag_noise_density / np.sqrt(2 * self.dt)
        noise = np.random.normal(0, noise_std, 3)
        measurement += noise

        return measurement

    def update_bias_drift(self):
        """
        Update bias drift using random walk model.
        """
        # Accelerometer bias drift (random walk)
        drift_step = np.random.normal(0, self.params.acc_random_walk * np.sqrt(self.dt), 3)
        self.acc_bias_drift += drift_step

        # Gyroscope bias drift (random walk)
        drift_step = np.random.normal(0, self.params.gyro_random_walk * np.sqrt(self.dt), 3)
        self.gyro_bias_drift += drift_step

        # Magnetometer bias drift (random walk)
        drift_step = np.random.normal(0, 0.001 * np.sqrt(self.dt), 3)  # Typical mag RW
        self.mag_bias_drift += drift_step

    def simulate_sample(self, true_acc: np.ndarray, true_omega: np.ndarray,
                       true_mag: np.ndarray = None, temperature: float = 25.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a complete IMU sample with all sensors.
        """
        # Update bias drift
        self.update_bias_drift()

        # Simulate each sensor
        acc_measurement = self.simulate_accelerometer(true_acc, temperature)
        gyro_measurement = self.simulate_gyroscope(true_omega, temperature)

        if true_mag is not None:
            mag_measurement = self.simulate_magnetometer(true_mag, temperature)
        else:
            mag_measurement = np.array([0.0, 0.0, 0.0])  # Zero if not provided

        return acc_measurement, gyro_measurement, mag_measurement

def demonstrate_imu_simulation():
    """
    Demonstrate the IMU simulation with realistic motion.
    """
    print("=== IMU Sensor Simulation Demonstration ===\n")

    # Create IMU parameters
    imu_params = IMUParams(
        acc_noise_density=80e-6,
        acc_random_walk=2.0e-4,
        gyro_noise_density=0.5e-3,
        gyro_random_walk=0.001,
        sampling_rate=100.0
    )

    imu_sim = IMUSimulator(imu_params)

    # Simulate some motion (e.g., circular motion with rotation)
    duration = 10.0  # seconds
    num_samples = int(duration * imu_params.sampling_rate)

    # True motion parameters
    radius = 5.0  # meters
    omega = 0.5   # rad/s (period = 2*pi/omega ~ 12.6s)

    # Arrays to store measurements
    true_accs = []
    true_omegas = []
    measured_accs = []
    measured_gyros = []
    measured_mags = []

    for i in range(num_samples):
        t = i / imu_params.sampling_rate

        # True motion (circular path)
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)

        # True velocities
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)

        # True accelerations (centripetal + tangential)
        ax = -radius * omega**2 * np.cos(omega * t)  # centripetal
        ay = -radius * omega**2 * np.sin(omega * t)

        # True angular velocity (constant rotation about z-axis)
        true_omega = np.array([0.0, 0.0, omega])

        # True acceleration (including gravity in body frame)
        # Assuming body frame rotates with motion
        theta = omega * t
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])

        # Gravity vector in inertial frame
        g_inertial = np.array([0.0, 0.0, -9.81])
        # Transform to body frame
        g_body = R.T @ g_inertial
        # Add centripetal acceleration
        acc_body = np.array([ax, ay, 0.0])
        true_acc = acc_body + g_body

        # True magnetic field (Earth's field in North-East-Down frame)
        true_mag = np.array([25000e-9, 5000e-9, -45000e-9])  # Roughly Earth's field in nT

        # Simulate IMU measurement
        acc_meas, gyro_meas, mag_meas = imu_sim.simulate_sample(
            true_acc, true_omega, true_mag, temperature=25.0
        )

        # Store for analysis
        true_accs.append(true_acc.copy())
        true_omegas.append(true_omega.copy())
        measured_accs.append(acc_meas.copy())
        measured_gyros.append(gyro_meas.copy())
        measured_mags.append(mag_meas.copy())

    # Convert to arrays
    true_accs = np.array(true_accs)
    true_omegas = np.array(omegas)
    measured_accs = np.array(measured_accs)
    measured_gyros = np.array(measured_gyros)
    measured_mags = np.array(measured_mags)

    print(f"Simulated {num_samples} IMU samples over {duration}s")
    print(f"True acceleration - Mean: {np.mean(true_accs, axis=0)}, Std: {np.std(true_accs, axis=0)}")
    print(f"Measured acceleration - Mean: {np.mean(measured_accs, axis=0)}, Std: {np.std(measured_accs, axis=0)}")
    print(f"Acceleration error (RMS): {np.sqrt(np.mean((true_accs - measured_accs)**2, axis=0))}")

    print(f"\nTrue angular velocity - Mean: {np.mean(true_omegas, axis=0)}, Std: {np.std(true_omegas, axis=0)}")
    print(f"Measured angular velocity - Mean: {np.mean(measured_gyros, axis=0)}, Std: {np.std(measured_gyros, axis=0)}")
    print(f"Angular velocity error (RMS): {np.sqrt(np.mean((true_omegas - measured_gyros[:,:2])[:2])**2, axis=0)}")  # Only compare x,y

    # Visualize results
    try:
        time_axis = np.arange(num_samples) / imu_params.sampling_rate

        # Plot accelerometer data
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 2, 1)
        plt.plot(time_axis, true_accs[:, 0], label='True X', alpha=0.7)
        plt.plot(time_axis, measured_accs[:, 0], label='Measured X', alpha=0.7)
        plt.title('Accelerometer X-axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 2)
        plt.plot(time_axis, true_accs[:, 1], label='True Y', alpha=0.7)
        plt.plot(time_axis, measured_accs[:, 1], label='Measured Y', alpha=0.7)
        plt.title('Accelerometer Y-axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 3)
        plt.plot(time_axis, true_omegas[:, 2], label='True Z (gyro)', alpha=0.7)
        plt.plot(time_axis, measured_gyros[:, 2], label='Measured Z (gyro)', alpha=0.7)
        plt.title('Gyroscope Z-axis')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 4)
        plt.plot(time_axis, measured_mags[:, 0], label='Mag X', alpha=0.7)
        plt.plot(time_axis, measured_mags[:, 1], label='Mag Y', alpha=0.7)
        plt.plot(time_axis, measured_mags[:, 2], label='Mag Z', alpha=0.7)
        plt.title('Magnetometer (simulated)')
        plt.xlabel('Time (s)')
        plt.ylabel('Magnetic Field (T)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 5)
        plt.plot(time_axis, measured_accs[:, 2], label='Measured Z (acc)', alpha=0.7)
        plt.title('Accelerometer Z-axis (includes gravity)')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 2, 6)
        error_acc = np.linalg.norm(true_accs - measured_accs, axis=1)
        plt.plot(time_axis, error_acc)
        plt.title('Total Accelerometer Error Magnitude')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m/s²)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping visualization.")

    return true_accs, measured_accs, true_omegas, measured_gyros

if __name__ == "__main__":
    true_accs, meas_accs, true_omegas, meas_gyros = demonstrate_imu_simulation()
```

## Labs and Exercises

### Exercise 1: Camera Calibration Simulation
Simulate a camera calibration process in a digital twin environment. Generate synthetic checkerboard images with known poses, add realistic noise and distortion, then implement a calibration algorithm to recover the camera parameters. Compare the recovered parameters with the ground truth values.

### Exercise 2: Multi-Sensor Fusion Implementation
Implement a sensor fusion algorithm that combines data from multiple simulated sensors (e.g., camera, LIDAR, and IMU) to improve state estimation accuracy. Use techniques like Extended Kalman Filtering or Particle Filtering to combine the sensor data effectively.

### Exercise 3: Environmental Effects Modeling
Extend the sensor simulation to include environmental effects like rain, fog, or varying lighting conditions. Implement models that modify sensor performance based on environmental parameters and validate the effects against real-world sensor behavior.

### Exercise 4: Sensor Failure Simulation
Implement realistic sensor failure models that simulate various types of sensor malfunctions (bias drift, noise increase, complete failure, intermittent operation). Design fault detection algorithms that can identify and handle sensor failures in real-time.