---
sidebar_position: 2
---

# System Architecture: Autonomous Humanoid Design

## Overview

The system architecture of an autonomous humanoid robot represents one of the most complex engineering challenges in robotics, requiring the seamless integration of mechanical, electrical, computational, and software components into a unified, real-time system. This chapter explores the architectural principles, design patterns, and implementation strategies for creating robust, efficient, and safe humanoid robot systems that can operate autonomously in dynamic human environments. The architecture must balance competing requirements including real-time performance, power efficiency, safety, reliability, and adaptability while supporting the complex multimodal processing required for autonomous operation.

A successful humanoid architecture must address the fundamental challenges of embodied AI: the tight coupling between perception, cognition, and action in a physical system that must maintain balance and safety while performing complex tasks. The architecture encompasses multiple layers of abstraction, from low-level motor control and sensor processing to high-level cognitive planning and human interaction. Each layer must operate with precise timing constraints while communicating effectively with other layers to enable coordinated behavior.

The design of humanoid robot architecture has evolved significantly with advances in computing power, sensor technology, and AI algorithms. Modern architectures leverage distributed computing, real-time operating systems, and sophisticated middleware to achieve the performance and reliability required for autonomous operation. The architecture must also accommodate the unique challenges of humanoid form factors, including the need for dynamic balance, complex kinematics, and safe human interaction capabilities.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Design system architectures for autonomous humanoid robots with real-time performance requirements
- Implement distributed computing architectures that balance computational load across multiple processing units
- Create safe and reliable communication protocols between humanoid subsystems
- Design fault-tolerant architectures that maintain operation despite component failures
- Implement real-time scheduling and resource management for humanoid systems
- Evaluate architectural trade-offs between performance, power consumption, and safety
- Design modular architectures that support iterative development and testing

## Key Concepts

### Distributed Computing Architecture

Architectural approaches for distributing computation across humanoid systems:

- **Edge Computing**: Local processing on robot-mounted computers for low-latency responses
- **Cloud Integration**: Offloading complex computations to remote servers when appropriate
- **Hierarchical Processing**: Multi-level processing from raw sensors to high-level cognition
- **Parallel Processing**: Simultaneous processing of multiple data streams
- **Load Balancing**: Dynamic distribution of computational tasks across available resources
- **Resource Management**: Efficient allocation of computational and power resources

### Real-Time System Design

Principles for designing systems with strict timing requirements:

- **Deterministic Execution**: Predictable timing behavior for safety-critical functions
- **Priority-Based Scheduling**: Ensuring critical tasks receive necessary resources
- **Deadline Management**: Meeting timing constraints for different system components
- **Interrupt Handling**: Managing high-priority events without disrupting ongoing processes
- **Buffer Management**: Efficient handling of data streams with minimal latency
- **Timing Analysis**: Verification that timing requirements are met under all conditions

### Safety and Reliability Architecture

Design principles for ensuring safe and reliable operation:

- **Fail-Safe Mechanisms**: Default safe states when systems fail
- **Redundancy**: Backup systems for critical functions
- **Fault Detection**: Continuous monitoring for system failures
- **Graceful Degradation**: Maintaining partial functionality when components fail
- **Safety Monitoring**: Continuous assessment of system safety status
- **Emergency Protocols**: Rapid response to safety-critical situations

### Communication and Middleware

Architectural patterns for system communication:

- **Message Passing**: Asynchronous communication between components
- **Service-Based Architecture**: Request-response patterns for system services
- **Event-Driven Systems**: Reactive architectures responding to system events
- **Real-Time Communication**: Low-latency communication for time-critical functions
- **Network Protocols**: Efficient communication between distributed components
- **Data Serialization**: Efficient encoding and transmission of complex data structures

## Diagrams and Code

### Autonomous Humanoid System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Humanoid Robot Architecture                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                           Computing Layer                                 │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │  │
│  │  │   Perception    │ │   Cognition     │ │     Action      │              │  │
│  │  │   Computer      │ │   Computer      │ │   Computer      │              │  │
│  │  │   (Vision,      │ │   (Planning,    │ │   (Control,     │              │  │
│  │  │   LIDAR, Audio) │ │   Reasoning)    │ │   Locomotion)   │              │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘              │  │
│  │           │                       │                       │               │  │
│  │           └───────────────────────┼───────────────────────┘               │  │
│  │                                   │                                       │  │
│  │                    ┌─────────────────────────┐                            │  │
│  │                    │    Central Hub          │                            │  │
│  │                    │    (ROS 2, Messaging,  │                            │  │
│  │                    │     Coordination)       │                            │  │
│  │                    └─────────────────────────┘                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                            │
│  ┌─────────────────────────────────┼───────────────────────────────────────────┘
│  │                                 │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐
│  │  │                           Hardware Layer                                │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            │
│  │  │  │   Sensor        │ │   Actuator      │ │   Power &       │            │
│  │  │  │   Systems       │ │   Systems       │ │   Communication │            │
│  │  │  │   (Cameras,     │ │   (Motors,      │ │   (WiFi,        │            │
│  │  │  │   LIDAR, IMU)   │ │   Servos,       │ │   Bluetooth,    │            │
│  │  │  └─────────────────┘ │   Hydraulics)   │ │   Ethernet)     │            │
│  │  │                        └─────────────────┘ └─────────────────┘            │
│  │  │                                 │                       │                 │
│  │  │                                 └───────────────────────┘                 │
│  │  │                                        │                                  │
│  │  │                    ┌─────────────────────────────────────────────────┐    │
│  │  │                    │            Mechanical Platform                  │    │
│  │  │                    │      (Frame, Joints, Balance System)            │    │
│  │  │                    └─────────────────────────────────────────────────┘    │
│  │  └───────────────────────────────────────────────────────────────────────────┘
│  │
│  └─────────────────────────────────────────────────────────────────────────────────┘
│                                    │
│                    ┌───────────────────────────────────────────────────────────────┐
│                    │                     Safety Layer                              │
│                    │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│                    │  │   Safety        │ │   Monitoring    │ │   Emergency     │  │
│                    │  │   Controller    │ │   Systems       │ │   Systems       │  │
│                    │  │   (Balance,     │ │   (IMU, Joint   │ │   (E-Stop,     │  │
│                    │  │   Collision)    │ │   Monitoring)   │ │   Recovery)     │  │
│                    │  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
│                    └───────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Real-Time System Architecture Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState, Imu, Image, PointCloud2
from std_msgs.msg import String, Bool, Float32
from geometry_msgs.msg import Twist, Pose
from builtin_interfaces.msg import Time
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import asyncio
from dataclasses import dataclass
from enum import Enum

class TaskPriority(Enum):
    """Task priority levels for real-time scheduling."""
    EMERGENCY = 0
    SAFETY_CRITICAL = 1
    TIME_CRITICAL = 2
    NORMAL = 3
    BACKGROUND = 4

@dataclass
class TaskDefinition:
    """Definition of a real-time task."""
    name: str
    function: Callable
    period: float  # seconds
    priority: TaskPriority
    deadline: float  # seconds
    resources: Dict[str, float]  # resource requirements

class RealTimeScheduler:
    """
    Real-time scheduler for humanoid robot tasks.
    Implements priority-based scheduling with deadline management.
    """

    def __init__(self):
        self.tasks: List[TaskDefinition] = []
        self.task_queue = []
        self.scheduling_lock = threading.Lock()
        self.system_time = time.time()
        self.performance_metrics = {}

    def add_task(self, task: TaskDefinition):
        """Add a task to the scheduler."""
        with self.scheduling_lock:
            self.tasks.append(task)
            # Sort tasks by priority (lower number = higher priority)
            self.tasks.sort(key=lambda t: t.priority.value)

    def schedule_tasks(self):
        """Generate execution schedule based on priorities and deadlines."""
        current_time = time.time()
        ready_tasks = []

        with self.scheduling_lock:
            for task in self.tasks:
                # Check if task is ready to execute based on its period
                if current_time % task.period < 0.001:  # Task ready
                    ready_tasks.append(task)

        # Sort ready tasks by priority
        ready_tasks.sort(key=lambda t: t.priority.value)

        return ready_tasks

    def execute_task(self, task: TaskDefinition):
        """Execute a single task with deadline monitoring."""
        start_time = time.time()

        try:
            # Execute the task
            result = task.function()

            execution_time = time.time() - start_time

            # Check deadline compliance
            deadline_met = execution_time <= task.deadline

            # Update performance metrics
            if task.name not in self.performance_metrics:
                self.performance_metrics[task.name] = []

            self.performance_metrics[task.name].append({
                'execution_time': execution_time,
                'deadline_met': deadline_met,
                'timestamp': start_time
            })

            return result

        except Exception as e:
            print(f"Task {task.name} execution failed: {str(e)}")
            return None

    def get_performance_report(self) -> Dict:
        """Get performance metrics for all tasks."""
        report = {}

        for task_name, metrics in self.performance_metrics.items():
            if metrics:
                execution_times = [m['execution_time'] for m in metrics]
                deadline_compliance = [m['deadline_met'] for m in metrics]

                report[task_name] = {
                    'avg_execution_time': np.mean(execution_times),
                    'max_execution_time': max(execution_times),
                    'deadline_compliance_rate': sum(deadline_compliance) / len(deadline_compliance),
                    'total_executions': len(metrics)
                }

        return report

class ResourceManager:
    """
    Resource manager for computational and power resources.
    """

    def __init__(self):
        self.computational_resources = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0,
            'available_threads': 8
        }
        self.power_resources = {
            'battery_level': 100.0,
            'current_draw': 0.0,
            'estimated_runtime': 3600.0  # seconds
        }
        self.resource_lock = threading.Lock()

    def allocate_resources(self, task: TaskDefinition) -> bool:
        """Allocate resources for a task."""
        with self.resource_lock:
            # Check if sufficient resources are available
            required_cpu = task.resources.get('cpu', 0.1)
            required_memory = task.resources.get('memory', 100.0)  # MB

            if (self.computational_resources['cpu_usage'] + required_cpu <= 1.0 and
                self.computational_resources['available_threads'] > 0):

                # Allocate resources
                self.computational_resources['cpu_usage'] += required_cpu
                self.computational_resources['available_threads'] -= 1

                return True
            else:
                return False

    def release_resources(self, task: TaskDefinition):
        """Release resources after task completion."""
        with self.resource_lock:
            required_cpu = task.resources.get('cpu', 0.1)
            self.computational_resources['cpu_usage'] = max(
                0, self.computational_resources['cpu_usage'] - required_cpu
            )
            self.computational_resources['available_threads'] += 1

    def get_resource_status(self) -> Dict:
        """Get current resource status."""
        with self.resource_lock:
            return {
                'computational': self.computational_resources.copy(),
                'power': self.power_resources.copy()
            }

class HumanoidSystemArchitecture(Node):
    """
    Main system architecture node for the autonomous humanoid.
    Coordinates all subsystems with real-time scheduling and resource management.
    """

    def __init__(self):
        super().__init__('humanoid_system_architecture')

        # Initialize real-time scheduler
        self.scheduler = RealTimeScheduler()
        self.resource_manager = ResourceManager()

        # System state
        self.system_state = {
            'operational': True,
            'safety_mode': False,
            'current_behavior': 'idle',
            'task_execution_time': 0.0
        }

        # Initialize subsystems
        self.perception_subsystem = PerceptionSubsystem()
        self.cognition_subsystem = CognitionSubsystem()
        self.action_subsystem = ActionSubsystem()
        self.safety_subsystem = SafetySubsystem()

        # Publishers for system coordination
        self.system_status_pub = self.create_publisher(String, '/system/status', 10)
        self.safety_alert_pub = self.create_publisher(String, '/safety/alert', 10)
        self.resource_status_pub = self.create_publisher(String, '/system/resources', 10)

        # Subscribers for all subsystems
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.vision_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.vision_callback, 10
        )

        # Initialize real-time tasks
        self.initialize_real_time_tasks()

        # Start main control loop
        self.main_loop_rate = self.create_rate(100)  # 100 Hz
        self.main_loop_thread = threading.Thread(target=self.main_control_loop, daemon=True)
        self.main_loop_thread.start()

        self.get_logger().info('Humanoid System Architecture initialized')

    def initialize_real_time_tasks(self):
        """Initialize all real-time tasks with their scheduling parameters."""

        # Safety-critical tasks (highest priority)
        balance_task = TaskDefinition(
            name='balance_control',
            function=self.balance_control_task,
            period=0.01,  # 100 Hz
            priority=TaskPriority.SAFETY_CRITICAL,
            deadline=0.008,  # 8ms deadline
            resources={'cpu': 0.15, 'memory': 50.0}
        )

        emergency_task = TaskDefinition(
            name='emergency_monitoring',
            function=self.emergency_monitoring_task,
            period=0.005,  # 200 Hz
            priority=TaskPriority.EMERGENCY,
            deadline=0.003,  # 3ms deadline
            resources={'cpu': 0.05, 'memory': 20.0}
        )

        # Time-critical tasks
        perception_task = TaskDefinition(
            name='perception_processing',
            function=self.perception_processing_task,
            period=0.033,  # ~30 Hz
            priority=TaskPriority.TIME_CRITICAL,
            deadline=0.030,  # 30ms deadline
            resources={'cpu': 0.25, 'memory': 200.0, 'gpu': 0.3}
        )

        # Normal priority tasks
        planning_task = TaskDefinition(
            name='cognitive_planning',
            function=self.cognitive_planning_task,
            period=0.1,  # 10 Hz
            priority=TaskPriority.NORMAL,
            deadline=0.08,  # 80ms deadline
            resources={'cpu': 0.2, 'memory': 150.0}
        )

        # Add tasks to scheduler
        self.scheduler.add_task(emergency_task)
        self.scheduler.add_task(balance_task)
        self.scheduler.add_task(perception_task)
        self.scheduler.add_task(planning_task)

    def main_control_loop(self):
        """Main real-time control loop."""
        while rclpy.ok() and self.system_state['operational']:
            try:
                # Schedule and execute tasks
                ready_tasks = self.scheduler.schedule_tasks()

                for task in ready_tasks:
                    # Check if resources are available
                    if self.resource_manager.allocate_resources(task):
                        # Execute task
                        self.scheduler.execute_task(task)
                        # Release resources
                        self.resource_manager.release_resources(task)
                    else:
                        self.get_logger().warn(f'Insufficient resources for task: {task.name}')

                # Publish system status periodically
                if time.time() % 1.0 < 0.01:  # Every second
                    self.publish_system_status()

                # Small sleep to prevent busy waiting
                time.sleep(0.001)

            except Exception as e:
                self.get_logger().error(f'Main control loop error: {str(e)}')
                time.sleep(0.01)

    def balance_control_task(self):
        """Safety-critical balance control task."""
        # Get current IMU and joint data
        imu_data = self.safety_subsystem.get_current_imu_data()
        joint_data = self.action_subsystem.get_current_joint_data()

        # Calculate balance correction
        balance_correction = self.safety_subsystem.calculate_balance_correction(
            imu_data, joint_data
        )

        # Apply balance correction
        self.action_subsystem.apply_balance_correction(balance_correction)

        return balance_correction

    def emergency_monitoring_task(self):
        """Emergency monitoring task."""
        # Check for emergency conditions
        emergency_status = self.safety_subsystem.check_emergency_conditions()

        if emergency_status['emergency']:
            self.trigger_emergency_procedure(emergency_status)
            return emergency_status

        return {'emergency': False, 'condition': 'normal'}

    def perception_processing_task(self):
        """Perception processing task."""
        # Process latest sensor data
        vision_data = self.perception_subsystem.get_latest_vision_data()
        lidar_data = self.perception_subsystem.get_latest_lidar_data()

        # Perform perception processing
        perception_result = self.perception_subsystem.process_perception(
            vision_data, lidar_data
        )

        # Update cognition subsystem with perception data
        self.cognition_subsystem.update_perception_data(perception_result)

        return perception_result

    def cognitive_planning_task(self):
        """Cognitive planning task."""
        # Get current perception and goal data
        current_state = self.cognition_subsystem.get_current_state()
        goals = self.cognition_subsystem.get_goals()

        # Generate action plan
        action_plan = self.cognition_subsystem.generate_action_plan(
            current_state, goals
        )

        # Update action subsystem with plan
        self.action_subsystem.update_action_plan(action_plan)

        return action_plan

    def joint_state_callback(self, msg):
        """Handle joint state updates."""
        self.action_subsystem.update_joint_states(msg)

    def imu_callback(self, msg):
        """Handle IMU updates."""
        self.safety_subsystem.update_imu_data(msg)

    def vision_callback(self, msg):
        """Handle vision updates."""
        self.perception_subsystem.update_vision_data(msg)

    def trigger_emergency_procedure(self, emergency_status: Dict):
        """Trigger emergency procedure."""
        self.system_state['safety_mode'] = True
        self.system_state['current_behavior'] = 'emergency_stop'

        # Publish safety alert
        alert_msg = String()
        alert_msg.data = f"EMERGENCY: {emergency_status['condition']}"
        self.safety_alert_pub.publish(alert_msg)

        # Emergency stop all actions
        self.action_subsystem.emergency_stop()

        self.get_logger().error(f'Emergency procedure triggered: {emergency_status}')

    def publish_system_status(self):
        """Publish overall system status."""
        status = {
            'operational': self.system_state['operational'],
            'safety_mode': self.system_state['safety_mode'],
            'current_behavior': self.system_state['current_behavior'],
            'resource_status': self.resource_manager.get_resource_status(),
            'performance_metrics': self.scheduler.get_performance_report(),
            'timestamp': time.time()
        }

        status_msg = String()
        status_msg.data = str(status)
        self.system_status_pub.publish(status_msg)

class PerceptionSubsystem:
    """
    Perception subsystem managing sensors and data processing.
    """

    def __init__(self):
        self.vision_data = None
        self.lidar_data = None
        self.audio_data = None
        self.perception_cache = {}
        self.data_timestamps = {}

    def update_vision_data(self, image_msg):
        """Update with new vision data."""
        self.vision_data = image_msg
        self.data_timestamps['vision'] = time.time()

    def get_latest_vision_data(self):
        """Get latest vision data."""
        return self.vision_data

    def get_latest_lidar_data(self):
        """Get latest LIDAR data."""
        return self.lidar_data

    def process_perception(self, vision_data, lidar_data):
        """Process perception data from multiple sensors."""
        # In real implementation, this would run through perception pipelines
        # For simulation, return dummy processed data
        processed_data = {
            'objects': self.detect_objects(vision_data),
            'spatial_map': self.create_spatial_map(lidar_data),
            'features': self.extract_features(vision_data)
        }
        return processed_data

    def detect_objects(self, vision_data):
        """Detect objects in vision data."""
        # Simulated object detection
        return [
            {'name': 'person', 'position': [1.0, 0.5, 0.0], 'confidence': 0.85},
            {'name': 'table', 'position': [2.0, 1.0, 0.0], 'confidence': 0.92}
        ]

    def create_spatial_map(self, lidar_data):
        """Create spatial map from LIDAR data."""
        # Simulated spatial mapping
        return {
            'resolution': 0.05,
            'origin': [0, 0, 0],
            'occupied_cells': 150,
            'free_cells': 850
        }

    def extract_features(self, vision_data):
        """Extract features from vision data."""
        # Simulated feature extraction
        return np.random.random(512).tolist()

class CognitionSubsystem:
    """
    Cognition subsystem for planning and reasoning.
    """

    def __init__(self):
        self.perception_data = None
        self.goals = []
        self.current_plan = None
        self.memory = []
        self.knowledge_base = {}

    def update_perception_data(self, perception_data):
        """Update with new perception data."""
        self.perception_data = perception_data

    def get_current_state(self):
        """Get current system state."""
        return {
            'perception': self.perception_data,
            'goals': self.goals,
            'current_plan': self.current_plan,
            'memory': self.memory[-10:]  # Last 10 memory items
        }

    def get_goals(self):
        """Get current goals."""
        return self.goals

    def generate_action_plan(self, current_state, goals):
        """Generate action plan based on current state and goals."""
        if not goals:
            return {'actions': [], 'status': 'idle'}

        # Simple planning algorithm for simulation
        plan = {
            'actions': [],
            'status': 'planning',
            'timestamp': time.time()
        }

        for goal in goals:
            if goal['type'] == 'navigation':
                plan['actions'].append({
                    'type': 'navigate',
                    'target': goal['target'],
                    'parameters': {'speed': 0.5, 'accuracy': 0.1}
                })
            elif goal['type'] == 'manipulation':
                plan['actions'].append({
                    'type': 'manipulate',
                    'target': goal['target'],
                    'parameters': {'grip_type': 'precision'}
                })

        self.current_plan = plan
        return plan

class ActionSubsystem:
    """
    Action subsystem for motor control and locomotion.
    """

    def __init__(self):
        self.joint_states = {}
        self.current_plan = None
        self.motor_controllers = {}
        self.locomotion_controller = None

    def update_joint_states(self, joint_msg):
        """Update with current joint states."""
        self.joint_states = {
            'positions': list(joint_msg.position),
            'velocities': list(joint_msg.velocity),
            'efforts': list(joint_msg.effort),
            'names': list(joint_msg.name)
        }

    def get_current_joint_data(self):
        """Get current joint data."""
        return self.joint_states

    def update_action_plan(self, plan):
        """Update with new action plan."""
        self.current_plan = plan

    def apply_balance_correction(self, correction):
        """Apply balance correction to motors."""
        # Simulate applying balance correction
        pass

    def emergency_stop(self):
        """Emergency stop all motors."""
        # Simulate emergency stop
        pass

class SafetySubsystem:
    """
    Safety subsystem for monitoring and protection.
    """

    def __init__(self):
        self.imu_data = {}
        self.safety_limits = {
            'max_tilt_angle': 30.0,  # degrees
            'max_angular_velocity': 2.0,  # rad/s
            'min_joint_effort': -100.0,  # N*m
            'max_joint_effort': 100.0    # N*m
        }
        self.emergency_conditions = []

    def update_imu_data(self, imu_msg):
        """Update with new IMU data."""
        self.imu_data = {
            'linear_acceleration': [
                imu_msg.linear_acceleration.x,
                imu_msg.linear_acceleration.y,
                imu_msg.linear_acceleration.z
            ],
            'angular_velocity': [
                imu_msg.angular_velocity.x,
                imu_msg.angular_velocity.y,
                imu_msg.angular_velocity.z
            ],
            'orientation': [
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w
            ]
        }

    def get_current_imu_data(self):
        """Get current IMU data."""
        return self.imu_data

    def check_emergency_conditions(self) -> Dict:
        """Check for emergency conditions."""
        conditions = []

        # Check orientation limits
        if 'orientation' in self.imu_data:
            euler = self.quaternion_to_euler(self.imu_data['orientation'])
            roll, pitch, yaw = euler

            if abs(roll) > np.radians(self.safety_limits['max_tilt_angle']) or \
               abs(pitch) > np.radians(self.safety_limits['max_tilt_angle']):
                conditions.append('EXCESSIVE_TILT')

        # Check angular velocity limits
        if 'angular_velocity' in self.imu_data:
            ang_vel = self.imu_data['angular_velocity']
            ang_vel_magnitude = np.sqrt(sum(v**2 for v in ang_vel))

            if ang_vel_magnitude > self.safety_limits['max_angular_velocity']:
                conditions.append('EXCESSIVE_ANGULAR_VELOCITY')

        return {
            'emergency': len(conditions) > 0,
            'condition': conditions[0] if conditions else 'NORMAL',
            'all_conditions': conditions
        }

    def calculate_balance_correction(self, imu_data, joint_data):
        """Calculate balance correction based on IMU and joint data."""
        # Simple balance control algorithm
        if 'orientation' in imu_data:
            euler = self.quaternion_to_euler(imu_data['orientation'])
            roll, pitch, yaw = euler

            # Calculate correction based on tilt
            roll_correction = -roll * 0.5  # Proportional control
            pitch_correction = -pitch * 0.5

            return {
                'roll_correction': roll_correction,
                'pitch_correction': pitch_correction,
                'timestamp': time.time()
            }

        return {'roll_correction': 0.0, 'pitch_correction': 0.0}

    def quaternion_to_euler(self, quat: List[float]) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles."""
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)

    architecture_node = HumanoidSystemArchitecture()

    try:
        rclpy.spin(architecture_node)
    except KeyboardInterrupt:
        pass
    finally:
        architecture_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Distributed Computing and Communication Architecture

```python
import zmq
import json
import threading
import time
from typing import Dict, List, Optional, Callable
import asyncio
import multiprocessing as mp
from dataclasses import dataclass

@dataclass
class Message:
    """Message structure for inter-process communication."""
    type: str
    source: str
    destination: str
    data: Dict
    timestamp: float
    correlation_id: Optional[str] = None

class MessageBroker:
    """
    Message broker for distributed humanoid system communication.
    Uses ZeroMQ for efficient message passing between processes.
    """

    def __init__(self, broker_port: int = 5555):
        self.context = zmq.Context()
        self.broker_port = broker_port

        # Create router and dealer sockets
        self.router = self.context.socket(zmq.ROUTER)
        self.dealer = self.context.socket(zmq.DEALER)

        self.router.bind(f"tcp://*:{broker_port}")
        self.dealer.bind(f"tcp://*:{broker_port + 1}")

        # Start broker thread
        self.broker_thread = threading.Thread(target=self._run_broker, daemon=True)
        self.broker_thread.start()

    def _run_broker(self):
        """Run the message broker."""
        try:
            zmq.proxy(self.router, self.dealer)
        except Exception as e:
            print(f"Broker error: {e}")
        finally:
            self.router.close()
            self.dealer.close()
            self.context.term()

class ProcessNode:
    """
    Base class for distributed processing nodes.
    """

    def __init__(self, node_name: str, broker_port: int = 5555):
        self.node_name = node_name
        self.broker_port = broker_port
        self.context = zmq.Context()

        # Create DEALER socket to connect to broker
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, node_name)
        self.socket.connect(f"tcp://localhost:{broker_port + 1}")

        # Message handlers
        self.message_handlers = {}

        # Start message processing thread
        self.processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processing_thread.start()

    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler

    def send_message(self, destination: str, message_type: str, data: Dict) -> str:
        """Send a message to another node."""
        message = Message(
            type=message_type,
            source=self.node_name,
            destination=destination,
            data=data,
            timestamp=time.time()
        )

        # Send JSON-serialized message
        self.socket.send_multipart([
            destination.encode(),
            b"",
            json.dumps(message.__dict__).encode()
        ])

        return str(message.timestamp)  # Use timestamp as correlation ID

    def _process_messages(self):
        """Process incoming messages."""
        while True:
            try:
                # Receive message
                identity, empty, json_msg = self.socket.recv_multipart()

                # Parse message
                msg_dict = json.loads(json_msg.decode())
                message = Message(**msg_dict)

                # Handle message based on type
                if message.type in self.message_handlers:
                    self.message_handlers[message.type](message)
                else:
                    print(f"Unknown message type: {message.type}")

            except Exception as e:
                print(f"Message processing error: {e}")

    def close(self):
        """Close the node connection."""
        self.socket.close()
        self.context.term()

class PerceptionNode(ProcessNode):
    """
    Distributed perception processing node.
    """

    def __init__(self, broker_port: int = 5555):
        super().__init__("perception_node", broker_port)

        # Register message handlers
        self.register_handler("sensor_data", self.handle_sensor_data)
        self.register_handler("request_objects", self.handle_object_request)

        # Simulated perception data
        self.perception_cache = {}

    def handle_sensor_data(self, message: Message):
        """Handle incoming sensor data."""
        sensor_type = message.data.get('sensor_type')
        sensor_data = message.data.get('data')

        # Process sensor data based on type
        if sensor_type == 'vision':
            processed_objects = self.process_vision_data(sensor_data)
            self.perception_cache['objects'] = processed_objects

            # Send processed data to cognition node
            self.send_message(
                "cognition_node",
                "perception_update",
                {
                    "objects": processed_objects,
                    "timestamp": message.timestamp
                }
            )

        elif sensor_type == 'lidar':
            processed_map = self.process_lidar_data(sensor_data)
            self.perception_cache['spatial_map'] = processed_map

    def handle_object_request(self, message: Message):
        """Handle request for object information."""
        requested_objects = message.data.get('objects', [])

        response_data = {
            "objects": self.perception_cache.get('objects', []),
            "request_id": message.correlation_id
        }

        self.send_message(message.source, "object_response", response_data)

    def process_vision_data(self, data):
        """Process vision sensor data."""
        # Simulate vision processing
        return [
            {"name": "person", "position": [1.0, 0.5, 0.0], "confidence": 0.85},
            {"name": "table", "position": [2.0, 1.0, 0.0], "confidence": 0.92}
        ]

    def process_lidar_data(self, data):
        """Process LIDAR sensor data."""
        # Simulate LIDAR processing
        return {
            "resolution": 0.05,
            "occupied_cells": 150,
            "free_cells": 850
        }

class CognitionNode(ProcessNode):
    """
    Distributed cognition processing node.
    """

    def __init__(self, broker_port: int = 5555):
        super().__init__("cognition_node", broker_port)

        # Register message handlers
        self.register_handler("perception_update", self.handle_perception_update)
        self.register_handler("goal_request", self.handle_goal_request)
        self.register_handler("command", self.handle_command)

        # System state
        self.current_state = {
            "objects": [],
            "spatial_map": {},
            "goals": [],
            "plan": []
        }

    def handle_perception_update(self, message: Message):
        """Handle perception data update."""
        self.current_state["objects"] = message.data.get("objects", [])

        # Generate plan if goals exist
        if self.current_state["goals"]:
            plan = self.generate_plan()
            self.current_state["plan"] = plan

            # Send plan to action node
            self.send_message(
                "action_node",
                "action_plan",
                {
                    "plan": plan,
                    "timestamp": message.timestamp
                }
            )

    def handle_goal_request(self, message: Message):
        """Handle request for current goals."""
        response_data = {
            "goals": self.current_state["goals"],
            "request_id": message.correlation_id
        }

        self.send_message(message.source, "goal_response", response_data)

    def handle_command(self, message: Message):
        """Handle high-level commands."""
        command = message.data.get("command")
        target = message.data.get("target")

        # Add goal based on command
        if command == "navigate_to":
            goal = {
                "type": "navigation",
                "target": target,
                "priority": 1
            }
            self.current_state["goals"].append(goal)

            # Generate new plan
            plan = self.generate_plan()
            self.current_state["plan"] = plan

            # Send plan to action node
            self.send_message("action_node", "action_plan", {"plan": plan})

    def generate_plan(self):
        """Generate action plan based on current state and goals."""
        plan = []

        for goal in self.current_state["goals"]:
            if goal["type"] == "navigation":
                plan.append({
                    "action": "navigate",
                    "target": goal["target"],
                    "parameters": {"speed": 0.5, "accuracy": 0.1}
                })
            elif goal["type"] == "manipulation":
                plan.append({
                    "action": "manipulate",
                    "target": goal["target"],
                    "parameters": {"grip_type": "precision"}
                })

        return plan

class ActionNode(ProcessNode):
    """
    Distributed action/locomotion processing node.
    """

    def __init__(self, broker_port: int = 5555):
        super().__init__("action_node", broker_port)

        # Register message handlers
        self.register_handler("action_plan", self.handle_action_plan)
        self.register_handler("joint_state_request", self.handle_joint_request)

        # Action execution state
        self.current_plan = []
        self.executing_action = None

    def handle_action_plan(self, message: Message):
        """Handle incoming action plan."""
        self.current_plan = message.data.get("plan", [])

        # Execute first action in plan
        if self.current_plan:
            self.execute_next_action()

    def handle_joint_request(self, message: Message):
        """Handle request for joint states."""
        # Simulate joint states
        joint_states = {
            "positions": [0.0] * 20,  # 20 joints
            "velocities": [0.0] * 20,
            "efforts": [0.0] * 20
        }

        response_data = {
            "joint_states": joint_states,
            "request_id": message.correlation_id
        }

        self.send_message(message.source, "joint_state_response", response_data)

    def execute_next_action(self):
        """Execute the next action in the plan."""
        if not self.current_plan:
            return

        action = self.current_plan.pop(0)
        self.executing_action = action

        # Simulate action execution
        if action["action"] == "navigate":
            self.execute_navigation(action)
        elif action["action"] == "manipulate":
            self.execute_manipulation(action)

    def execute_navigation(self, action):
        """Execute navigation action."""
        target = action["target"]
        print(f"Navigating to {target}")

        # Simulate navigation completion
        time.sleep(1.0)

        # Send completion message
        self.send_message(
            "cognition_node",
            "action_complete",
            {
                "action": action,
                "status": "completed",
                "timestamp": time.time()
            }
        )

    def execute_manipulation(self, action):
        """Execute manipulation action."""
        target = action["target"]
        print(f"Manipulating {target}")

        # Simulate manipulation completion
        time.sleep(1.5)

        # Send completion message
        self.send_message(
            "cognition_node",
            "action_complete",
            {
                "action": action,
                "status": "completed",
                "timestamp": time.time()
            }
        )

class DistributedSystemManager:
    """
    Manager for the distributed humanoid system.
    """

    def __init__(self, broker_port: int = 5555):
        # Start message broker
        self.broker = MessageBroker(broker_port)

        # Create nodes
        self.perception_node = PerceptionNode(broker_port)
        self.cognition_node = CognitionNode(broker_port)
        self.action_node = ActionNode(broker_port)

        # Start nodes in separate processes
        self.nodes = [self.perception_node, self.cognition_node, self.action_node]

    def send_command(self, command: str, target: str = None):
        """Send command to the system."""
        command_data = {
            "command": command,
            "target": target
        }

        self.cognition_node.send_message("cognition_node", "command", command_data)

    def get_system_status(self) -> Dict:
        """Get overall system status."""
        return {
            "nodes": ["perception_node", "cognition_node", "action_node"],
            "broker_port": self.broker.broker_port,
            "timestamp": time.time()
        }

    def shutdown(self):
        """Shutdown the distributed system."""
        for node in self.nodes:
            node.close()

# Example usage
def example_distributed_system():
    """
    Example of using the distributed humanoid system.
    """
    print("Initializing distributed humanoid system...")

    # Create system manager
    system_manager = DistributedSystemManager()

    print("System initialized. Sending commands...")

    # Send navigation command
    system_manager.send_command("navigate_to", [2.0, 2.0, 0.0])

    # Send manipulation command
    system_manager.send_command("manipulate", "red_cup")

    # Get system status
    status = system_manager.get_system_status()
    print(f"System status: {status}")

    # Let system run for a while
    time.sleep(5)

    # Shutdown
    system_manager.shutdown()
    print("System shutdown completed.")

if __name__ == "__main__":
    example_distributed_system()
```

## Labs and Exercises

### Exercise 1: Real-Time Scheduling Implementation
Implement a real-time scheduler that can handle the timing requirements of a humanoid robot system. Test the scheduler with different task priorities and deadline requirements to ensure timing constraints are met.

### Exercise 2: Distributed System Communication
Create a distributed system architecture using message passing between different subsystems. Implement fault tolerance mechanisms to handle node failures and ensure system reliability.

### Exercise 3: Resource Management and Load Balancing
Develop a resource management system that can dynamically allocate computational resources based on task requirements and system load. Test the system under varying load conditions.

### Exercise 4: Safety Architecture Validation
Implement a comprehensive safety architecture with multiple layers of protection. Validate the safety systems under various failure scenarios to ensure they meet safety requirements.

## Summary

This chapter explored the system architecture of autonomous humanoid robots, covering the complex integration of computational, mechanical, and software components required for autonomous operation. We examined real-time scheduling principles, distributed computing architectures, and safety-critical design patterns essential for humanoid robot systems. The examples demonstrated how to implement real-time task scheduling, distributed communication, and resource management for complex robotic systems. A well-designed architecture is fundamental to creating humanoid robots that can operate safely and effectively in dynamic human environments, balancing performance, safety, and reliability requirements.