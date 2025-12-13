---
sidebar_position: 2
---

# Robotics Fundamentals

## Overview

Robotics is the interdisciplinary field that combines mechanical engineering, electrical engineering, computer science, and control theory to design, construct, operate, and apply robots. In the context of Physical AI, robotics provides the essential framework for understanding how intelligent systems can interact with the physical world through mechanical components, sensors, and actuators.

This chapter establishes the fundamental principles that underpin all robotic systems: kinematics (the study of motion without considering forces), dynamics (the study of motion with forces), control theory (methods for governing system behavior), and sensor integration (techniques for perceiving the environment). These concepts form the backbone of any robotic system and are essential for developing intelligent, autonomous robots that can effectively operate in the real world.

Understanding these fundamentals is crucial for Physical AI because they define the constraints and capabilities within which intelligent behavior must operate. A robot's physical embodiment—its mechanical structure, actuators, and sensors—fundamentally shapes what it can perceive, how it can act, and the types of intelligent behaviors it can exhibit.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Define and apply fundamental robotics concepts including kinematics, dynamics, and control
- Distinguish between forward and inverse kinematics problems and solve them for simple mechanisms
- Understand the relationship between robot dynamics and motion planning
- Analyze the role of feedback control in robotic systems
- Evaluate different sensor modalities and their applications in robotics
- Apply basic control theory to simple robotic systems

## Key Concepts

### Forward and Inverse Kinematics

**Forward Kinematics** is the process of determining the position and orientation of a robot's end-effector (or any other point of interest) given the joint angles or actuator positions. This is a straightforward calculation that involves multiplying transformation matrices corresponding to each joint.

**Inverse Kinematics** is the reverse problem: determining the joint angles required to achieve a desired end-effector position and orientation. This is often more challenging than forward kinematics and may have multiple solutions, no solutions, or require numerical methods to solve.

### Robot Dynamics and Motion Planning

Robot dynamics encompasses the forces and torques that cause motion in robotic systems. It includes:
- **Rigid body dynamics**: The motion of bodies under applied forces
- **Lagrangian mechanics**: A method for deriving equations of motion
- **Actuator dynamics**: The behavior of motors and other actuators
- **Friction and contact models**: How robots interact with their environment

Motion planning involves determining appropriate trajectories for robot movement while considering:
- Kinematic and dynamic constraints
- Obstacle avoidance
- Energy efficiency
- Smoothness and continuity of motion

### Control Systems and Feedback Loops

Robotic control systems use feedback to regulate robot behavior and achieve desired performance. Key concepts include:

- **Open-loop control**: Commands are sent without considering the actual state
- **Closed-loop (feedback) control**: System output is measured and used to adjust inputs
- **PID control**: Proportional-Integral-Derivative control for error minimization
- **State estimation**: Determining system state from sensor measurements

### Sensor Fusion and Perception

Robots must integrate information from multiple sensors to understand their environment:
- **Proprioceptive sensors**: Measure robot's internal state (encoders, IMUs)
- **Exteroceptive sensors**: Measure external environment (cameras, LIDAR, sonar)
- **Sensor fusion**: Combining multiple sensor readings to improve accuracy
- **State estimation**: Using sensor data to estimate robot and environment state

## Diagrams and Code

### Coordinate Systems and Transformations

```
     Z
     |   / Y
     |  /
     | /
     O -------- X

For a robotic manipulator, each joint has its own coordinate frame.
Transformation matrices map between these frames:

T = [R  p]
    [0  1]

Where R is a 3x3 rotation matrix and p is a 3x1 position vector.
```

### Forward Kinematics Example: 2-DOF Planar Manipulator

```python
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

class PlanarManipulator:
    """
    A 2-DOF planar manipulator with link lengths l1 and l2.
    Joint angles q1 and q2 are measured from the horizontal.
    """

    def __init__(self, l1=1.0, l2=1.0):
        self.l1 = l1  # Length of first link
        self.l2 = l2  # Length of second link

    def forward_kinematics(self, q1, q2):
        """
        Calculate end-effector position given joint angles.

        Args:
            q1: First joint angle (radians)
            q2: Second joint angle (radians)

        Returns:
            tuple: (x, y) end-effector position
        """
        x = self.l1 * cos(q1) + self.l2 * cos(q1 + q2)
        y = self.l1 * sin(q1) + self.l2 * sin(q1 + q2)
        return x, y

    def inverse_kinematics(self, x, y):
        """
        Calculate joint angles to reach desired end-effector position.
        Returns the "elbow up" solution.

        Args:
            x: Desired x position
            y: Desired y position

        Returns:
            tuple: (q1, q2) joint angles in radians, or None if no solution
        """
        # Distance from origin to end-effector
        r = np.sqrt(x**2 + y**2)

        # Check if position is reachable
        if r > (self.l1 + self.l2):
            print("Position is outside workspace")
            return None

        if r < abs(self.l1 - self.l2):
            print("Position is inside inner workspace boundary")
            return None

        # Calculate angle of second link relative to first
        cos_q2 = (r**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        sin_q2 = np.sqrt(1 - cos_q2**2)  # Take positive square root for "elbow up"
        q2 = np.arctan2(sin_q2, cos_q2)

        # Calculate angle of first link
        k1 = self.l1 + self.l2 * cos_q2
        k2 = self.l2 * sin_q2
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        return q1, q2

    def plot_configuration(self, q1, q2, title="Robot Configuration"):
        """
        Plot the manipulator in its current configuration.
        """
        # Calculate joint positions
        x1 = self.l1 * cos(q1)
        y1 = self.l1 * sin(q1)

        x2, y2 = self.forward_kinematics(q1, q2)

        # Plot the manipulator
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, x1, x2], [0, y1, y2], 'o-', linewidth=3, markersize=8,
                label='Manipulator Links')
        ax.plot(0, 0, 'ro', markersize=10, label='Base')
        ax.plot(x1, y1, 'go', markersize=8, label='Joint 2')
        ax.plot(x2, y2, 'bo', markersize=10, label='End-Effector')

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title(title)
        plt.show()

# Example usage
if __name__ == "__main__":
    manipulator = PlanarManipulator(l1=1.0, l2=0.8)

    # Example 1: Forward kinematics
    q1, q2 = np.pi/4, np.pi/3  # 45° and 60°
    x, y = manipulator.forward_kinematics(q1, q2)
    print(f"Forward Kinematics: Joint angles ({q1:.3f}, {q2:.3f}) -> End-effector ({x:.3f}, {y:.3f})")

    # Example 2: Inverse kinematics
    target_x, target_y = 1.2, 1.0
    solution = manipulator.inverse_kinematics(target_x, target_y)

    if solution:
        sol_q1, sol_q2 = solution
        print(f"Inverse Kinematics: Target ({target_x}, {target_y}) -> Joint angles ({sol_q1:.3f}, {sol_q2:.3f})")

        # Verify solution
        ver_x, ver_y = manipulator.forward_kinematics(sol_q1, sol_q2)
        print(f"Verification: Forward kinematics gives ({ver_x:.3f}, {ver_y:.3f})")
        print(f"Error: {np.sqrt((ver_x-target_x)**2 + (ver_y-target_y)**2):.6f}")

    # Plot the configuration
    manipulator.plot_configuration(q1, q2, "2-DOF Planar Manipulator Example")

### PID Controller Example for Robot Joint Control

```python
import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    """
    A PID controller for robot joint position control.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, dt=0.01):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.dt = dt  # Time step

        self.error_sum = 0.0
        self.prev_error = 0.0

    def update(self, setpoint, measurement):
        """
        Calculate control output based on setpoint and measurement.

        Args:
            setpoint: Desired value
            measurement: Current measured value

        Returns:
            float: Control output
        """
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_sum += error * self.dt
        i_term = self.ki * self.error_sum

        # Derivative term
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error

        output = p_term + i_term + d_term
        return output

class SimpleRobotJoint:
    """
    A simple model of a robot joint with inertia, damping, and gravity.
    """

    def __init__(self, inertia=1.0, damping=0.1, gravity_compensation=0.0):
        self.inertia = inertia
        self.damping = damping
        self.gravity_compensation = gravity_compensation

        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

    def update(self, torque, dt):
        """
        Update joint state based on applied torque.
        """
        # Calculate acceleration (tau = I*alpha, so alpha = tau/I)
        # Include damping and gravity compensation
        self.acceleration = (torque - self.damping * self.velocity + self.gravity_compensation) / self.inertia

        # Update velocity and position using basic physics
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

def simulate_joint_control():
    """
    Simulate PID control of a robot joint.
    """
    # Create controller and joint
    controller = PIDController(kp=10.0, ki=2.0, kd=0.5, dt=0.01)
    joint = SimpleRobotJoint(inertia=0.5, damping=0.1)

    # Simulation parameters
    dt = 0.01
    time_steps = 1000  # 10 seconds of simulation
    setpoint = np.pi/2  # Target position: 90 degrees

    # Storage for plotting
    times = []
    positions = []
    velocities = []
    errors = []

    # Simulation loop
    for i in range(time_steps):
        time = i * dt
        times.append(time)
        positions.append(joint.position)
        velocities.append(joint.velocity)
        errors.append(setpoint - joint.position)

        # Calculate control effort
        control_output = controller.update(setpoint, joint.position)

        # Apply control effort to joint (as torque)
        joint.update(control_output, dt)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    ax1.plot(times, positions, label='Actual Position')
    ax1.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Joint Position Control with PID')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(times, velocities)
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Joint Velocity')
    ax2.grid(True)

    ax3.plot(times, errors)
    ax3.set_ylabel('Error (rad)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Position Error')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Final position: {joint.position:.3f} rad")
    print(f"Final velocity: {joint.velocity:.3f} rad/s")
    print(f"Final error: {setpoint - joint.position:.3f} rad")

# Example usage
if __name__ == "__main__":
    simulate_joint_control()
```

## Labs and Exercises

### Exercise 1: Forward Kinematics Calculation
Implement forward kinematics for a 3-DOF planar manipulator with link lengths l1=1.0, l2=0.8, l3=0.5. Derive the equations for the end-effector position and test with several different joint angle configurations.

### Exercise 2: PID Controller Tuning
Implement the PID controller example and experiment with different gain values (kp, ki, kd). Observe how each parameter affects the system response and stability. Document your findings about how to tune PID controllers for robotic applications.

### Exercise 3: Workspace Analysis
For the 2-DOF planar manipulator, analyze its workspace by plotting the reachable area. Identify the boundary of the workspace and explain why certain positions are unreachable.

### Exercise 4: Sensor Integration
Design a sensor fusion algorithm that combines encoder readings with IMU data to estimate the position of a mobile robot. Implement a simple Kalman filter or complementary filter to combine the measurements.