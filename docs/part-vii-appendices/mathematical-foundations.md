---
sidebar_position: 2
---

# Appendix B: Mathematical Foundations for Robotics

## Overview

This appendix provides the essential mathematical foundations required for understanding and implementing humanoid robotics systems. The content covers the mathematical concepts, tools, and techniques that form the backbone of robotics, including linear algebra, calculus, probability theory, and their applications in robotics. Understanding these mathematical foundations is crucial for developing perception algorithms, control systems, planning methods, and machine learning applications in robotics.

The mathematical concepts presented here are specifically tailored to robotics applications, with emphasis on practical implementation and computational methods. The appendix bridges the gap between theoretical mathematics and applied robotics, showing how mathematical principles translate into algorithms and systems. Each concept is presented with clear explanations, examples, and code implementations that demonstrate their application in real robotics scenarios.

The mathematical foundations covered in this appendix include coordinate systems and transformations, which are fundamental for representing and manipulating spatial relationships in robotics. These concepts are essential for tasks such as robot localization, mapping, navigation, and manipulation. The appendix also covers optimization techniques that are crucial for trajectory planning, parameter estimation, and control system design.

## Learning Outcomes

By the end of this appendix, you should be able to:

- Apply linear algebra concepts to robotics problems including transformations and kinematics
- Understand and implement coordinate system transformations and representations
- Use calculus for motion planning, control system analysis, and optimization
- Apply probability theory to sensor fusion, localization, and uncertainty management
- Implement mathematical algorithms for robotics applications using computational tools
- Analyze the mathematical properties of robotic systems and algorithms
- Solve robotics problems using appropriate mathematical methods and techniques

## Key Concepts

### Linear Algebra in Robotics

Fundamental linear algebra concepts for robotics applications:

- **Vectors and Matrices**: Representation of positions, orientations, and transformations
- **Vector Spaces**: Mathematical structures for representing robot configurations
- **Linear Transformations**: Mathematical operations for coordinate system changes
- **Eigenvalues and Eigenvectors**: Analysis of system dynamics and stability
- **Matrix Decompositions**: LU, QR, and SVD for solving robotics problems
- **Norms and Metrics**: Distance measures and optimization criteria

### Coordinate Systems and Transformations

Mathematical representation of spatial relationships:

- **Cartesian Coordinates**: Standard 3D coordinate system for position representation
- **Spherical and Cylindrical Coordinates**: Alternative coordinate representations
- **Rotation Matrices**: 3x3 matrices for representing orientations
- **Quaternions**: 4D representations for rotations without gimbal lock
- **Homogeneous Transformations**: 4x4 matrices for combined rotation and translation
- **Denavit-Hartenberg Parameters**: Method for describing robot kinematics

### Calculus Applications in Robotics

Calculus concepts applied to robotics systems:

- **Derivatives**: Velocity and acceleration computation from position data
- **Integrals**: Position computation from velocity data and trajectory planning
- **Partial Derivatives**: Multivariable optimization and Jacobian computation
- **Differential Equations**: Modeling robot dynamics and control systems
- **Vector Calculus**: Analysis of force and motion fields
- **Optimization**: Finding minimum energy paths and optimal controls

### Probability and Statistics in Robotics

Probabilistic methods for handling uncertainty in robotics:

- **Probability Distributions**: Modeling sensor noise and uncertainty
- **Bayesian Inference**: Updating beliefs based on sensor observations
- **Kalman Filtering**: Estimating robot state from noisy sensor data
- **Monte Carlo Methods**: Sampling-based approaches for complex problems
- **Statistical Estimation**: Parameter estimation from sensor measurements
- **Information Theory**: Quantifying information content and uncertainty

## Mathematical Foundations Implementation

### Linear Algebra Operations for Robotics

```python
import numpy as np
import math
from typing import Tuple, List, Optional

class LinearAlgebraRobotics:
    """
    Mathematical operations for robotics using linear algebra.
    """

    @staticmethod
    def rotation_matrix_x(angle: float) -> np.ndarray:
        """
        Generate rotation matrix around X-axis.

        Args:
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])

    @staticmethod
    def rotation_matrix_y(angle: float) -> np.ndarray:
        """
        Generate rotation matrix around Y-axis.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])

    @staticmethod
    def rotation_matrix_z(angle: float) -> np.ndarray:
        """
        Generate rotation matrix around Z-axis.
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrix.

        Args:
            roll: Rotation around X-axis
            pitch: Rotation around Y-axis
            yaw: Rotation around Z-axis

        Returns:
            3x3 rotation matrix
        """
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_x = LinearAlgebraRobotics.rotation_matrix_x(roll)
        R_y = LinearAlgebraRobotics.rotation_matrix_y(pitch)
        R_z = LinearAlgebraRobotics.rotation_matrix_z(yaw)

        return R_z @ R_y @ R_x

    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw).

        Args:
            R: 3x3 rotation matrix

        Returns:
            Tuple of (roll, pitch, yaw) angles in radians
        """
        # Handle gimbal lock case
        if abs(R[2, 0]) > 0.999:
            # Gimbal lock: pitch near ±90 degrees
            yaw = 0  # Convention
            pitch = math.atan2(-R[2, 0], 0)  # This creates a singularity
            roll = math.atan2(-R[0, 1], R[0, 2])
        else:
            # Standard case
            pitch = math.asin(-R[2, 0])
            yaw = math.atan2(R[1, 0], R[0, 0])
            roll = math.atan2(R[2, 1], R[2, 2])

        return roll, pitch, yaw

    @staticmethod
    def quaternion_to_rotation_matrix(q: List[float]) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q

        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> List[float]:
        """
        Convert rotation matrix to quaternion.

        Args:
            R: 3x3 rotation matrix

        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        return [w, x, y, z]

    @staticmethod
    def homogeneous_transform(translation: List[float],
                            rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Create homogeneous transformation matrix.

        Args:
            translation: [x, y, z] translation vector
            rotation_matrix: 3x3 rotation matrix

        Returns:
            4x4 homogeneous transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        return T

    @staticmethod
    def transform_point(point: List[float],
                       transform_matrix: np.ndarray) -> List[float]:
        """
        Transform a 3D point using homogeneous transformation matrix.

        Args:
            point: [x, y, z] point to transform
            transform_matrix: 4x4 homogeneous transformation matrix

        Returns:
            Transformed [x, y, z] point
        """
        # Convert to homogeneous coordinates
        homogeneous_point = np.array([point[0], point[1], point[2], 1.0])

        # Apply transformation
        transformed = transform_matrix @ homogeneous_point

        # Convert back to Cartesian coordinates
        return [transformed[0], transformed[1], transformed[2]]

    @staticmethod
    def skew_symmetric_matrix(vector: List[float]) -> np.ndarray:
        """
        Create skew-symmetric matrix from vector (for cross product).

        Args:
            vector: [x, y, z] vector

        Returns:
            3x3 skew-symmetric matrix
        """
        x, y, z = vector
        return np.array([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

class VectorOperations:
    """
    Vector operations commonly used in robotics.
    """

    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.

        Args:
            vector: Input vector [x, y, z]

        Returns:
            Normalized vector [x, y, z]
        """
        magnitude = math.sqrt(sum(v**2 for v in vector))
        if magnitude == 0:
            return [0.0, 0.0, 0.0]

        return [v / magnitude for v in vector]

    @staticmethod
    def cross_product(a: List[float], b: List[float]) -> List[float]:
        """
        Compute cross product of two 3D vectors.

        Args:
            a: First vector [x, y, z]
            b: Second vector [x, y, z]

        Returns:
            Cross product vector [x, y, z]
        """
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]

    @staticmethod
    def dot_product(a: List[float], b: List[float]) -> float:
        """
        Compute dot product of two vectors.

        Args:
            a: First vector [x, y, z]
            b: Second vector [x, y, z]

        Returns:
            Dot product (scalar)
        """
        return sum(a[i] * b[i] for i in range(len(a)))

    @staticmethod
    def vector_magnitude(vector: List[float]) -> float:
        """
        Compute magnitude of a vector.

        Args:
            vector: Input vector [x, y, z]

        Returns:
            Magnitude (scalar)
        """
        return math.sqrt(sum(v**2 for v in vector))

    @staticmethod
    def vector_distance(a: List[float], b: List[float]) -> float:
        """
        Compute Euclidean distance between two points.

        Args:
            a: First point [x, y, z]
            b: Second point [x, y, z]

        Returns:
            Distance (scalar)
        """
        return VectorOperations.vector_magnitude([
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2]
        ])

# Example usage and testing
def test_linear_algebra_operations():
    """
    Test the linear algebra operations with examples.
    """
    print("Testing Linear Algebra Operations for Robotics")
    print("=" * 50)

    # Test rotation matrices
    print("\n1. Testing Rotation Matrices:")

    # Test X-axis rotation
    Rx = LinearAlgebraRobotics.rotation_matrix_x(math.pi/4)  # 45 degrees
    print(f"X-axis rotation by 45°:\n{Rx}")

    # Test Y-axis rotation
    Ry = LinearAlgebraRobotics.rotation_matrix_y(math.pi/6)  # 30 degrees
    print(f"Y-axis rotation by 30°:\n{Ry}")

    # Test Z-axis rotation
    Rz = LinearAlgebraRobotics.rotation_matrix_z(math.pi/3)  # 60 degrees
    print(f"Z-axis rotation by 60°:\n{Rz}")

    # Test Euler to rotation matrix conversion
    print("\n2. Testing Euler to Rotation Matrix:")
    R_euler = LinearAlgebraRobotics.euler_to_rotation_matrix(
        math.pi/4, math.pi/6, math.pi/3  # roll=45°, pitch=30°, yaw=60°
    )
    print(f"Rotation matrix from Euler angles:\n{R_euler}")

    # Convert back to Euler angles
    roll, pitch, yaw = LinearAlgebraRobotics.rotation_matrix_to_euler(R_euler)
    print(f"Converted back - Roll: {math.degrees(roll):.2f}°, "
          f"Pitch: {math.degrees(pitch):.2f}°, "
          f"Yaw: {math.degrees(yaw):.2f}°")

    # Test quaternion operations
    print("\n3. Testing Quaternion Operations:")
    q = [0.707, 0.707, 0, 0]  # 90° rotation around X-axis
    R_quat = LinearAlgebraRobotics.quaternion_to_rotation_matrix(q)
    print(f"Rotation matrix from quaternion:\n{R_quat}")

    q_back = LinearAlgebraRobotics.rotation_matrix_to_quaternion(R_quat)
    print(f"Quaternion back from rotation matrix: {q_back}")

    # Test homogeneous transformations
    print("\n4. Testing Homogeneous Transformations:")
    translation = [1.0, 2.0, 3.0]
    rotation = LinearAlgebraRobotics.rotation_matrix_z(math.pi/4)
    T = LinearAlgebraRobotics.homogeneous_transform(translation, rotation)
    print(f"Homogeneous transformation matrix:\n{T}")

    # Transform a point
    point = [1.0, 0.0, 0.0]
    transformed_point = LinearAlgebraRobotics.transform_point(point, T)
    print(f"Original point: {point}")
    print(f"Transformed point: {transformed_point}")

    # Test vector operations
    print("\n5. Testing Vector Operations:")
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]

    print(f"Vector a: {a}")
    print(f"Vector b: {b}")
    print(f"Dot product: {VectorOperations.dot_product(a, b):.3f}")
    print(f"Cross product: {VectorOperations.cross_product(a, b)}")
    print(f"Magnitude of a: {VectorOperations.vector_magnitude(a):.3f}")
    print(f"Distance between a and b: {VectorOperations.vector_distance(a, b):.3f}")

    normalized_a = VectorOperations.normalize_vector(a)
    print(f"Normalized vector a: {normalized_a}")
    print(f"Magnitude of normalized a: {VectorOperations.vector_magnitude(normalized_a):.3f}")

if __name__ == "__main__":
    test_linear_algebra_operations()
```

### Calculus Applications in Robotics

```python
import numpy as np
import math
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

class CalculusRobotics:
    """
    Calculus applications for robotics including differentiation, integration, and optimization.
    """

    @staticmethod
    def numerical_derivative(func: Callable[[float], float],
                           x: float,
                           h: float = 1e-7) -> float:
        """
        Compute numerical derivative using central difference method.

        Args:
            func: Function to differentiate
            x: Point at which to compute derivative
            h: Step size for numerical differentiation

        Returns:
            Numerical derivative at point x
        """
        return (func(x + h) - func(x - h)) / (2 * h)

    @staticmethod
    def numerical_jacobian(func: Callable[[List[float]], List[float]],
                          x: List[float],
                          h: float = 1e-7) -> np.ndarray:
        """
        Compute numerical Jacobian matrix of a vector function.

        Args:
            func: Vector function R^n -> R^m
            x: Input vector at which to compute Jacobian
            h: Step size for numerical differentiation

        Returns:
            Jacobian matrix (m x n)
        """
        x = np.array(x)
        n = len(x)

        # Evaluate function at x
        fx = np.array(func(x.tolist()))
        m = len(fx)

        # Initialize Jacobian
        J = np.zeros((m, n))

        # Compute each column of Jacobian
        for j in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[j] += h
            x_minus[j] -= h

            f_plus = np.array(func(x_plus.tolist()))
            f_minus = np.array(func(x_minus.tolist()))

            J[:, j] = (f_plus - f_minus) / (2 * h)

        return J

    @staticmethod
    def integrate_trapezoidal(func: Callable[[float], float],
                            a: float,
                            b: float,
                            n: int = 1000) -> float:
        """
        Numerical integration using trapezoidal rule.

        Args:
            func: Function to integrate
            a: Lower limit of integration
            b: Upper limit of integration
            n: Number of intervals

        Returns:
            Approximate integral value
        """
        h = (b - a) / n
        x_values = np.linspace(a, b, n + 1)
        y_values = [func(x) for x in x_values]

        # Trapezoidal rule: h/2 * [f(x0) + 2*f(x1) + 2*f(x2) + ... + f(xn)]
        integral = h * (0.5 * y_values[0] + sum(y_values[1:-1]) + 0.5 * y_values[-1])
        return integral

    @staticmethod
    def velocity_from_position(position_func: Callable[[float], List[float]],
                            t: float,
                            h: float = 1e-5) -> List[float]:
        """
        Compute velocity from position function using numerical differentiation.

        Args:
            position_func: Function returning [x, y, z] position at time t
            t: Time at which to compute velocity
            h: Step size for numerical differentiation

        Returns:
            Velocity vector [vx, vy, vz]
        """
        pos_plus = np.array(position_func(t + h))
        pos_minus = np.array(position_func(t - h))

        velocity = (pos_plus - pos_minus) / (2 * h)
        return velocity.tolist()

    @staticmethod
    def acceleration_from_position(position_func: Callable[[float], List[float]],
                                t: float,
                                h: float = 1e-5) -> List[float]:
        """
        Compute acceleration from position function using numerical differentiation.

        Args:
            position_func: Function returning [x, y, z] position at time t
            t: Time at which to compute acceleration
            h: Step size for numerical differentiation

        Returns:
            Acceleration vector [ax, ay, az]
        """
        pos_plus = np.array(position_func(t + h))
        pos_current = np.array(position_func(t))
        pos_minus = np.array(position_func(t - h))

        acceleration = (pos_plus - 2 * pos_current + pos_minus) / (h ** 2)
        return acceleration.tolist()

    @staticmethod
    def trajectory_planning_polynomial(start_pos: float,
                                     start_vel: float,
                                     end_pos: float,
                                     end_vel: float,
                                     duration: float) -> Callable[[float], float]:
        """
        Plan trajectory using 5th order polynomial (minimum jerk).

        Args:
            start_pos: Starting position
            start_vel: Starting velocity (should be 0 for smooth start)
            end_pos: Ending position
            end_vel: Ending velocity (should be 0 for smooth stop)
            duration: Total duration of motion

        Returns:
            Function that computes position at any time t (0 <= t <= duration)
        """
        # For minimum jerk trajectory, we use a 5th order polynomial:
        # s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5

        # Boundary conditions:
        # s(0) = start_pos, s'(0) = start_vel, s''(0) = 0
        # s(T) = end_pos, s'(T) = end_vel, s''(T) = 0

        T = duration
        delta = end_pos - start_pos

        a0 = start_pos
        a1 = start_vel
        a2 = 0  # start acceleration = 0
        a3 = (20*delta - (8*end_vel + 12*start_vel)*T) / (2 * T**3)
        a4 = (30*start_vel + 20*end_vel - 30*delta) / (2 * T**4)
        a5 = (12*delta - 6*start_vel - 6*end_vel) / (2 * T**5)

        def trajectory_func(t: float) -> float:
            if t < 0:
                return start_pos
            elif t > T:
                return end_pos
            else:
                t_rel = t
                return a0 + a1*t_rel + a2*t_rel**2 + a3*t_rel**3 + a4*t_rel**4 + a5*t_rel**5

        return trajectory_func

    @staticmethod
    def gradient_descent(func: Callable[[List[float]], float],
                        initial_params: List[float],
                        learning_rate: float = 0.01,
                        max_iterations: int = 1000,
                        tolerance: float = 1e-6) -> Tuple[List[float], List[float]]:
        """
        Gradient descent optimization algorithm.

        Args:
            func: Function to minimize
            initial_params: Initial parameter values
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (optimal parameters, function values over iterations)
        """
        params = initial_params.copy()
        func_values = []

        for i in range(max_iterations):
            # Compute gradient numerically
            gradient = []
            for j in range(len(params)):
                params_plus = params.copy()
                params_minus = params.copy()
                h = 1e-7
                params_plus[j] += h
                params_minus[j] -= h

                grad_j = (func(params_plus) - func(params_minus)) / (2 * h)
                gradient.append(grad_j)

            # Compute function value
            func_val = func(params)
            func_values.append(func_val)

            # Update parameters
            for j in range(len(params)):
                params[j] -= learning_rate * gradient[j]

            # Check convergence
            if i > 0 and abs(func_values[-1] - func_values[-2]) < tolerance:
                break

        return params, func_values

class KinematicsCalculus:
    """
    Calculus applications specifically for robot kinematics.
    """

    @staticmethod
    def jacobian_2dof_link_lengths(theta1: float, theta2: float,
                                 l1: float, l2: float) -> np.ndarray:
        """
        Compute Jacobian matrix for 2-DOF planar manipulator.

        Args:
            theta1: Joint angle 1
            theta2: Joint angle 2
            l1: Length of first link
            l2: Length of second link

        Returns:
            2x2 Jacobian matrix [dx/dtheta1, dx/dtheta2; dy/dtheta1, dy/dtheta2]
        """
        # Forward kinematics:
        # x = l1*cos(theta1) + l2*cos(theta1 + theta2)
        # y = l1*sin(theta1) + l2*sin(theta1 + theta2)

        # Jacobian:
        # dx/dtheta1 = -l1*sin(theta1) - l2*sin(theta1 + theta2)
        # dx/dtheta2 = -l2*sin(theta1 + theta2)
        # dy/dtheta1 = l1*cos(theta1) + l2*cos(theta1 + theta2)
        # dy/dtheta2 = l2*cos(theta1 + theta2)

        c1 = math.cos(theta1)
        s1 = math.sin(theta1)
        c12 = math.cos(theta1 + theta2)
        s12 = math.sin(theta1 + theta2)

        J = np.array([
            [-l1*s1 - l2*s12, -l2*s12],
            [l1*c1 + l2*c12, l2*c12]
        ])

        return J

    @staticmethod
    def inverse_kinematics_2dof(x: float, y: float, l1: float, l2: float) -> Tuple[float, float]:
        """
        Compute inverse kinematics for 2-DOF planar manipulator.

        Args:
            x: Target x-coordinate
            y: Target y-coordinate
            l1: Length of first link
            l2: Length of second link

        Returns:
            Tuple of (theta1, theta2) joint angles
        """
        # Distance from origin to target
        r = math.sqrt(x**2 + y**2)

        # Check if target is reachable
        if r > l1 + l2:
            # Target is outside workspace - return closest point
            scale = (l1 + l2) / r
            x = x * scale
            y = y * scale
            r = l1 + l2
        elif r < abs(l1 - l2):
            # Target is inside inner workspace - return extended position
            pass

        # Compute theta2 using law of cosines
        cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = max(-1, min(1, cos_theta2))  # Clamp to [-1, 1]
        theta2 = math.acos(cos_theta2)

        # Compute theta1
        k1 = l1 + l2 * math.cos(theta2)
        k2 = l2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        return theta1, theta2

    @staticmethod
    def forward_kinematics_2dof(theta1: float, theta2: float,
                              l1: float, l2: float) -> Tuple[float, float]:
        """
        Compute forward kinematics for 2-DOF planar manipulator.

        Args:
            theta1: Joint angle 1
            theta2: Joint angle 2
            l1: Length of first link
            l2: Length of second link

        Returns:
            Tuple of (x, y) end-effector position
        """
        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)

        return x, y

# Example usage and applications
def demonstrate_calculus_applications():
    """
    Demonstrate calculus applications in robotics with examples.
    """
    print("Calculus Applications in Robotics")
    print("=" * 40)

    # Example 1: Numerical differentiation
    print("\n1. Numerical Differentiation Example:")

    # Define a trajectory function (e.g., position over time)
    def position_trajectory(t):
        return math.sin(t) + 0.5 * math.sin(3 * t)  # Oscillating trajectory

    # Compute velocity at specific time
    time = 1.0
    velocity = CalculusRobotics.numerical_derivative(position_trajectory, time)
    acceleration = CalculusRobotics.numerical_derivative(
        lambda t: CalculusRobotics.numerical_derivative(position_trajectory, t),
        time
    )

    print(f"Position at t={time}: {position_trajectory(time):.3f}")
    print(f"Velocity at t={time}: {velocity:.3f}")
    print(f"Acceleration at t={time}: {acceleration:.3f}")

    # Example 2: 3D motion analysis
    print("\n2. 3D Motion Analysis:")

    def position_3d(t):
        """Example 3D position function"""
        x = math.cos(t)
        y = math.sin(t)
        z = 0.1 * t  # Helical motion
        return [x, y, z]

    time = 2.0
    velocity_3d = CalculusRobotics.velocity_from_position(position_3d, time)
    acceleration_3d = CalculusRobotics.acceleration_from_position(position_3d, time)

    print(f"3D Position at t={time}: {position_3d(time)}")
    print(f"3D Velocity at t={time}: {velocity_3d}")
    print(f"3D Acceleration at t={time}: {acceleration_3d}")

    # Example 3: Trajectory planning
    print("\n3. Minimum Jerk Trajectory Planning:")

    # Plan trajectory from 0 to 1 in 2 seconds
    trajectory_func = CalculusRobotics.trajectory_planning_polynomial(
        start_pos=0.0, start_vel=0.0,
        end_pos=1.0, end_vel=0.0,
        duration=2.0
    )

    # Sample trajectory
    times = np.linspace(0, 2, 100)
    positions = [trajectory_func(t) for t in times]
    velocities = [CalculusRobotics.numerical_derivative(trajectory_func, t) for t in times]
    accelerations = [CalculusRobotics.numerical_derivative(
        lambda t_prime: CalculusRobotics.numerical_derivative(trajectory_func, t_prime),
        t) for t in times]

    print(f"Trajectory starts at {trajectory_func(0):.3f} and ends at {trajectory_func(2):.3f}")
    print(f"Peak velocity: {max(abs(v) for v in velocities):.3f}")
    print(f"Peak acceleration: {max(abs(a) for a in accelerations):.3f}")

    # Example 4: Kinematics
    print("\n4. Robot Kinematics Example:")

    # 2-DOF manipulator with link lengths
    l1, l2 = 1.0, 0.8

    # Target position
    target_x, target_y = 1.2, 0.8

    # Compute inverse kinematics
    theta1, theta2 = KinematicsCalculus.inverse_kinematics_2dof(
        target_x, target_y, l1, l2
    )

    # Verify with forward kinematics
    x_calc, y_calc = KinematicsCalculus.forward_kinematics_2dof(
        theta1, theta2, l1, l2
    )

    # Compute Jacobian
    J = KinematicsCalculus.jacobian_2dof_link_lengths(theta1, theta2, l1, l2)

    print(f"Target: ({target_x}, {target_y})")
    print(f"Inverse kinematics: theta1={math.degrees(theta1):.1f}°, theta2={math.degrees(theta2):.1f}°")
    print(f"Forward kinematics verification: ({x_calc:.3f}, {y_calc:.3f})")
    print(f"Jacobian matrix:\n{J}")

    # Example 5: Optimization
    print("\n5. Optimization Example:")

    # Define a simple function to minimize: f(x,y) = (x-2)^2 + (y-3)^2
    def simple_function(params):
        x, y = params
        return (x - 2)**2 + (y - 3)**2

    # Run gradient descent
    optimal_params, func_values = CalculusRobotics.gradient_descent(
        simple_function,
        initial_params=[0.0, 0.0],
        learning_rate=0.1,
        max_iterations=100
    )

    print(f"Optimization result: {optimal_params}")
    print(f"True minimum: [2.0, 3.0]")
    print(f"Function value at optimum: {simple_function(optimal_params):.6f}")
    print(f"Number of iterations: {len(func_values)}")

if __name__ == "__main__":
    demonstrate_calculus_applications()
```

### Probability and Statistics for Robotics

```python
import numpy as np
import math
from typing import List, Tuple, Dict, Any
import random

class ProbabilityRobotics:
    """
    Probability theory applications for robotics including sensor fusion, localization, and uncertainty management.
    """

    @staticmethod
    def gaussian_probability(x: float, mean: float, variance: float) -> float:
        """
        Compute probability density for Gaussian distribution.

        Args:
            x: Value at which to compute probability
            mean: Mean of Gaussian distribution
            variance: Variance of Gaussian distribution

        Returns:
            Probability density at x
        """
        std_dev = math.sqrt(variance)
        coefficient = 1.0 / (std_dev * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - mean) / std_dev) ** 2
        return coefficient * math.exp(exponent)

    @staticmethod
    def multivariate_gaussian_probability(x: List[float],
                                        mean: List[float],
                                        covariance: List[List[float]]) -> float:
        """
        Compute probability density for multivariate Gaussian distribution.

        Args:
            x: Input vector
            mean: Mean vector
            covariance: Covariance matrix

        Returns:
            Probability density at x
        """
        x = np.array(x)
        mean = np.array(mean)
        cov = np.array(covariance)

        n = len(x)

        # Compute determinant and inverse of covariance matrix
        det = np.linalg.det(cov)
        if det <= 0:
            return 0.0  # Invalid covariance matrix

        inv_cov = np.linalg.inv(cov)

        # Compute the exponential term
        diff = x - mean
        exponent = -0.5 * diff.T @ inv_cov @ diff

        # Compute normalization constant
        norm_const = 1.0 / math.sqrt((2 * math.pi) ** n * det)

        return norm_const * math.exp(exponent)

    @staticmethod
    def kalman_predict(x_prior: List[float],
                      P_prior: List[List[float]],
                      F: List[List[float]],
                      Q: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        Kalman filter prediction step.

        Args:
            x_prior: Prior state estimate
            P_prior: Prior state covariance
            F: State transition model
            Q: Process noise covariance

        Returns:
            Tuple of (predicted state, predicted covariance)
        """
        x_prior = np.array(x_prior)
        P_prior = np.array(P_prior)
        F = np.array(F)
        Q = np.array(Q)

        # Predict state: x_pred = F * x_prior
        x_pred = F @ x_prior

        # Predict covariance: P_pred = F * P_prior * F.T + Q
        P_pred = F @ P_prior @ F.T + Q

        return x_pred.tolist(), P_pred.tolist()

    @staticmethod
    def kalman_update(x_pred: List[float],
                     P_pred: List[List[float]],
                     z: List[float],
                     H: List[List[float]],
                     R: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        Kalman filter update step.

        Args:
            x_pred: Predicted state estimate
            P_pred: Predicted state covariance
            z: Measurement vector
            H: Measurement model
            R: Measurement noise covariance

        Returns:
            Tuple of (updated state, updated covariance)
        """
        x_pred = np.array(x_pred)
        P_pred = np.array(P_pred)
        z = np.array(z)
        H = np.array(H)
        R = np.array(R)

        # Compute innovation: y = z - H * x_pred
        y = z - H @ x_pred

        # Compute innovation covariance: S = H * P_pred * H.T + R
        S = H @ P_pred @ H.T + R

        # Compute Kalman gain: K = P_pred * H.T * S.inv
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update state: x_update = x_pred + K * y
        x_update = x_pred + K @ y

        # Update covariance: P_update = (I - K * H) * P_pred
        I = np.eye(len(x_pred))
        P_update = (I - K @ H) @ P_pred

        return x_update.tolist(), P_update.tolist()

    @staticmethod
    def particle_filter_step(particles: List[Tuple[List[float], float]],
                           control: List[float],
                           measurement: List[float],
                           motion_model: Callable,
                           measurement_model: Callable) -> List[Tuple[List[float], float]]:
        """
        Single step of particle filter algorithm.

        Args:
            particles: List of (state, weight) tuples
            control: Control input applied to system
            measurement: Measurement received from sensors
            motion_model: Function for motion prediction
            measurement_model: Function for measurement likelihood

        Returns:
            Updated list of particles with new weights
        """
        new_particles = []

        for state, weight in particles:
            # Predict new state based on motion model
            new_state = motion_model(state, control)

            # Calculate likelihood of measurement given new state
            likelihood = measurement_model(new_state, measurement)

            # Update weight
            new_weight = weight * likelihood

            new_particles.append((new_state, new_weight))

        # Normalize weights
        total_weight = sum(weight for _, weight in new_particles)
        if total_weight > 0:
            normalized_particles = [
                (state, weight / total_weight)
                for state, weight in new_particles
            ]
        else:
            # If all weights are zero, reset uniform weights
            normalized_particles = [
                (state, 1.0 / len(new_particles))
                for state, _ in new_particles
            ]

        return normalized_particles

    @staticmethod
    def resample_particles(particles: List[Tuple[List[float], float]],
                          num_particles: int) -> List[Tuple[List[float], float]]:
        """
        Resample particles based on their weights.

        Args:
            particles: List of (state, weight) tuples
            num_particles: Number of particles to maintain

        Returns:
            Resampled list of particles
        """
        # Extract states and weights
        states = [state for state, _ in particles]
        weights = [weight for _, weight in particles]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # Uniform resampling if all weights are zero
            return [(states[i % len(states)], 1.0/num_particles)
                   for i in range(num_particles)]

        normalized_weights = [w / total_weight for w in weights]

        # Systematic resampling
        new_particles = []
        step = 1.0 / num_particles
        start = random.random() * step

        cumulative_weight = 0.0
        weight_idx = 0

        for i in range(num_particles):
            threshold = start + i * step

            while cumulative_weight < threshold and weight_idx < len(normalized_weights):
                cumulative_weight += normalized_weights[weight_idx]
                weight_idx += 1

            if weight_idx > 0:
                selected_idx = weight_idx - 1
            else:
                selected_idx = 0

            new_particles.append((states[selected_idx], 1.0/num_particles))

        return new_particles

class SensorFusion:
    """
    Sensor fusion techniques for robotics applications.
    """

    @staticmethod
    def weighted_fusion(sensor_readings: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Fuse multiple sensor readings using weighted averaging based on precision.

        Args:
            sensor_readings: List of (measurement, variance) tuples

        Returns:
            Tuple of (fused_mean, fused_variance)
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for measurement, variance in sensor_readings:
            if variance <= 0:
                continue  # Skip invalid measurements

            weight = 1.0 / variance  # Weight by precision (inverse of variance)
            total_weight += weight
            weighted_sum += weight * measurement

        if total_weight == 0:
            return sensor_readings[0][0], sensor_readings[0][1]  # Return first reading

        fused_mean = weighted_sum / total_weight
        fused_variance = 1.0 / total_weight

        return fused_mean, fused_variance

    @staticmethod
    def complementary_filter(prev_estimate: float,
                           measurement: float,
                           alpha: float = 0.8) -> float:
        """
        Complementary filter combining prediction and measurement.

        Args:
            prev_estimate: Previous estimate
            measurement: New measurement
            alpha: Weight for prediction (0-1, higher = more trust in prediction)

        Returns:
            Fused estimate
        """
        return alpha * prev_estimate + (1 - alpha) * measurement

    @staticmethod
    def kalman_filter_1d(observed_values: List[float],
                        process_variance: float = 1e-5,
                        measurement_variance: float = 1e-1) -> List[float]:
        """
        1D Kalman filter implementation.

        Args:
            observed_values: List of observed measurements
            process_variance: Process noise variance
            measurement_variance: Measurement noise variance

        Returns:
            List of filtered estimates
        """
        estimates = []

        # Initial state estimate and covariance
        x = 0.0  # Initial estimate
        P = 1.0  # Initial uncertainty

        for z in observed_values:
            # Prediction step (identity model: x = x)
            x_pred = x  # A = 1 for identity model
            P_pred = P + process_variance  # P_pred = P + Q

            # Update step
            K = P_pred / (P_pred + measurement_variance)  # Kalman gain
            x = x_pred + K * (z - x_pred)  # Update state
            P = (1 - K) * P_pred  # Update uncertainty

            estimates.append(x)

        return estimates

class LocalizationAlgorithms:
    """
    Localization algorithms using probability and statistics.
    """

    @staticmethod
    def monte_carlo_localization(initial_belief: List[Tuple[float, float, float, float]],
                               motion_commands: List[Tuple[float, float]],
                               measurements: List[List[Tuple[float, float, float]]],
                               motion_noise: float = 0.1,
                               measurement_noise: float = 0.1) -> List[Tuple[float, float]]:
        """
        Monte Carlo (particle filter) localization algorithm.

        Args:
            initial_belief: Initial particles (x, y, theta, weight)
            motion_commands: List of motion commands (linear_vel, angular_vel)
            measurements: List of measurements for each time step
            motion_noise: Noise level for motion model
            measurement_noise: Noise level for measurement model

        Returns:
            List of estimated positions at each time step
        """
        particles = initial_belief
        estimates = []

        for i, (motion_cmd, meas) in enumerate(zip(motion_commands, measurements)):
            # Prediction step: move particles according to motion command
            new_particles = []
            for x, y, theta, weight in particles:
                # Add motion noise
                noisy_linear = motion_cmd[0] + random.gauss(0, motion_noise)
                noisy_angular = motion_cmd[1] + random.gauss(0, motion_noise * 0.5)

                # Update particle position
                new_theta = theta + noisy_angular
                new_x = x + noisy_linear * math.cos(new_theta)
                new_y = y + noisy_linear * math.sin(new_theta)

                new_particles.append((new_x, new_y, new_theta, weight))

            particles = new_particles

            # Update step: adjust weights based on measurements
            new_particles = []
            total_weight = 0.0

            for x, y, theta, old_weight in particles:
                # Calculate likelihood of measurements given particle pose
                likelihood = 1.0
                for range_meas, bearing_meas in meas:
                    # Simulate expected measurement from this particle's position
                    # (simplified model - in reality, would use map and sensor model)
                    expected_range = math.sqrt(x**2 + y**2)  # Example
                    expected_bearing = math.atan2(y, x)  # Example

                    # Calculate likelihood
                    range_likelihood = ProbabilityRobotics.gaussian_probability(
                        range_meas, expected_range, measurement_noise
                    )
                    bearing_likelihood = ProbabilityRobotics.gaussian_probability(
                        bearing_meas, expected_bearing, measurement_noise
                    )

                    likelihood *= range_likelihood * bearing_likelihood

                new_weight = old_weight * likelihood
                new_particles.append((x, y, theta, new_weight))
                total_weight += new_weight

            # Normalize weights
            if total_weight > 0:
                normalized_particles = [
                    (x, y, theta, weight / total_weight)
                    for x, y, theta, weight in new_particles
                ]
            else:
                # Reset to uniform weights if all likelihoods are zero
                normalized_particles = [
                    (x, y, theta, 1.0 / len(new_particles))
                    for x, y, theta, _ in new_particles
                ]

            particles = normalized_particles

            # Calculate estimate (weighted average of particles)
            est_x = sum(x * w for x, y, theta, w in particles)
            est_y = sum(y * w for x, y, theta, w in particles)
            estimates.append((est_x, est_y))

        return estimates

# Example usage and demonstration
def demonstrate_probability_applications():
    """
    Demonstrate probability and statistics applications in robotics.
    """
    print("Probability and Statistics in Robotics")
    print("=" * 45)

    # Example 1: Gaussian distributions
    print("\n1. Gaussian Distribution Example:")

    # Calculate probability of measurement given model
    measurement = 2.5
    model_mean = 2.0
    model_variance = 0.25  # std dev = 0.5

    prob = ProbabilityRobotics.gaussian_probability(
        measurement, model_mean, model_variance
    )
    print(f"P(x={measurement} | μ={model_mean}, σ²={model_variance}) = {prob:.6f}")

    # Example 2: Sensor fusion
    print("\n2. Sensor Fusion Example:")

    # Three sensor readings with different uncertainties
    sensors = [
        (2.1, 0.1),   # Very precise
        (1.9, 0.4),   # Less precise
        (2.3, 0.2)    # Medium precision
    ]

    fused_mean, fused_var = SensorFusion.weighted_fusion(sensors)
    print(f"Individual readings: {[s[0] for s in sensors]}")
    print(f"Fused estimate: {fused_mean:.3f} ± {math.sqrt(fused_var):.3f}")

    # Example 3: Kalman filter
    print("\n3. Kalman Filter Example:")

    # Simulate noisy position measurements
    true_positions = [i * 0.5 for i in range(20)]  # True positions: 0, 0.5, 1.0, ...
    noisy_measurements = [pos + random.gauss(0, 0.3) for pos in true_positions]  # Add noise

    kalman_estimates = SensorFusion.kalman_filter_1d(
        noisy_measurements,
        process_variance=0.01,
        measurement_variance=0.09  # Variance of noise (0.3^2)
    )

    print(f"Initial error (first measurement): {abs(noisy_measurements[0] - true_positions[0]):.3f}")
    print(f"Final error (last estimate): {abs(kalman_estimates[-1] - true_positions[-1]):.3f}")
    print(f"Kalman filter reduces uncertainty over time")

    # Example 4: Monte Carlo localization
    print("\n4. Monte Carlo Localization Example:")

    # Initial particles (x, y, theta, weight)
    initial_particles = [
        (random.uniform(-1, 1), random.uniform(-1, 1),
         random.uniform(-math.pi, math.pi), 1.0/100)
        for _ in range(100)
    ]

    # Simulate motion and measurements
    motions = [(0.1, 0.05) for _ in range(5)]  # Move forward with slight turn
    measurements = [
        [(random.uniform(0.5, 1.5), random.uniform(-0.5, 0.5)) for _ in range(3)]
        for _ in range(5)
    ]  # 3 measurements per time step

    estimates = LocalizationAlgorithms.monte_carlo_localization(
        initial_particles, motions, measurements
    )

    print(f"Final estimated position: ({estimates[-1][0]:.3f}, {estimates[-1][1]:.3f})")
    print(f"Particle filter converges to consistent estimate over time")

    # Example 5: Multivariate Gaussian
    print("\n5. Multivariate Gaussian Example:")

    # 2D position with correlation
    x = [1.0, 2.0]  # x, y coordinates
    mean = [0.9, 2.1]  # Expected position
    cov = [[0.2, 0.05], [0.05, 0.3]]  # Covariance matrix (with correlation)

    prob_2d = ProbabilityRobotics.multivariate_gaussian_probability(x, mean, cov)
    print(f"2D Gaussian probability at {x} with mean {mean}: {prob_2d:.8f}")

if __name__ == "__main__":
    demonstrate_probability_applications()
```

## Troubleshooting Mathematical Implementations

### Common Issues and Solutions

When implementing mathematical concepts in robotics, several common issues may arise:

1. **Numerical Stability**: Matrix operations and iterative algorithms can suffer from numerical errors
2. **Coordinate System Confusion**: Different conventions (left-handed vs right-handed) can cause errors
3. **Unit Conversions**: Mixing degrees and radians is a frequent source of bugs
4. **Singularity Handling**: Gimbal lock and other singularities need special handling
5. **Performance**: Computationally expensive operations need optimization

### Validation Techniques

To ensure mathematical implementations are correct:

1. **Unit Testing**: Test with known inputs and expected outputs
2. **Symmetry Checks**: Verify that transformations are reversible
3. **Conservation Laws**: Check that energy, momentum, or other quantities are preserved where expected
4. **Limit Behavior**: Test extreme cases and boundary conditions
5. **Visual Verification**: Plot results when possible to verify reasonableness

## Summary

This appendix provided comprehensive coverage of the mathematical foundations essential for humanoid robotics, including linear algebra for transformations and kinematics, calculus for motion planning and control, and probability theory for sensor fusion and uncertainty management. The mathematical concepts were presented with practical implementations and examples relevant to robotics applications. Mastery of these mathematical foundations is crucial for developing sophisticated humanoid robot systems capable of perception, planning, control, and learning in complex environments.