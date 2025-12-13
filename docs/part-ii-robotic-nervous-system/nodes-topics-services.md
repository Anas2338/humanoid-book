---
sidebar_position: 2
---

# Nodes, Topics, and Services

## Overview

Nodes, topics, and services form the fundamental communication infrastructure of ROS 2. Understanding these concepts is essential for developing robust and efficient robotic applications. Nodes serve as the basic computational units that perform specific tasks, topics enable asynchronous communication through publish-subscribe patterns, and services provide synchronous request-response interactions for operations requiring immediate responses.

The architecture of ROS 2 is designed around a distributed system where nodes can run on different machines and communicate seamlessly. This distributed nature allows for fault tolerance, scalability, and the ability to run different components on hardware best suited for their requirements. The publish-subscribe model of topics enables loose coupling between nodes, allowing for flexible system architectures where publishers and subscribers can be added or removed without disrupting the overall system.

Quality of Service (QoS) settings provide fine-grained control over communication behavior, allowing developers to optimize for reliability, latency, bandwidth, or other performance metrics depending on the specific requirements of their robotic application. This is particularly important in safety-critical systems where communication guarantees are essential.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Design and implement ROS 2 nodes for specific computational tasks
- Choose appropriate communication patterns (topics vs services) for different scenarios
- Implement robust publishers, subscribers, clients, and servers
- Configure Quality of Service settings to meet application requirements
- Understand node lifecycle management and best practices
- Apply message definitions and interfaces effectively
- Debug communication issues in distributed ROS 2 systems

## Key Concepts

### Node Lifecycle and Management

ROS 2 nodes follow a well-defined lifecycle that allows for better resource management and system reliability:

- **Unconfigured state**: Node is created but not yet configured
- **Inactive state**: Node is configured but not actively processing
- **Active state**: Node is running and processing callbacks
- **Finalized state**: Node is shutting down and cleaning up resources

Nodes can transition between these states based on system requirements, enabling features like graceful startup, shutdown, and recovery from failures.

### Topic-Based Asynchronous Communication

Topics implement a publish-subscribe communication pattern:

- **Publishers** send messages to topics without knowing who will receive them
- **Subscribers** receive messages from topics without knowing who sent them
- Communication is decoupled in time and space
- Multiple publishers and subscribers can exist for the same topic
- Message delivery is asynchronous and non-blocking
- Quality of Service policies control delivery guarantees

### Service-Based Synchronous Communication

Services implement a request-response communication pattern:

- **Service clients** send requests and block until receiving a response
- **Service servers** process requests and send responses
- Communication is synchronous and blocking
- Useful for operations requiring immediate responses
- Provides request-response semantics with error handling
- Better suited for operations with clear start and end points

### Quality of Service (QoS) Settings

QoS settings provide control over communication behavior:

- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local (historical data)
- **History**: Keep all messages vs. keep last N messages
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: How to detect if a publisher is alive
- **Depth**: Size of the message queue

### Message Definitions and Interfaces

Messages are defined using IDL (Interface Definition Language):

- **.msg files** define message structures
- **.srv files** define service interfaces (request + response)
- **.action files** define action interfaces (goal + result + feedback)
- Messages are serialized for network transmission
- Support for multiple programming languages
- Version compatibility considerations

## Diagrams and Code

### Node-Topic Communication Pattern

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Node   │    │   Planner Node  │    │   Controller    │
│                 │    │                 │    │      Node       │
│ Publisher       │    │ Subscriber      │    │ Subscriber      │
│ (Topic: /sensors│    │ (Topic: /sensors│    │ (Topic: /cmd)   │
│  /laser_scan)   │───▶│ /laser_scan)    │───▶│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │   Visualization │
                    │     Node        │
                    │ Subscriber      │
                    │ (Topic: /sensors│
                    │  /laser_scan)   │
                    └─────────────────┘
```

### Advanced Publisher with QoS Configuration

```python
#!/usr/bin/env python3

"""
Advanced ROS 2 publisher with QoS configuration.
Demonstrates different QoS policies and their effects.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
import time
import random


class AdvancedPublisher(Node):
    """
    An advanced ROS 2 publisher demonstrating QoS configuration.
    """

    def __init__(self):
        super().__init__('advanced_publisher')

        # Define different QoS profiles for different use cases
        # For sensor data: Best effort with small history
        sensor_qos = QoSProfile(
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # For critical commands: Reliable with larger history
        cmd_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Create publishers with different QoS profiles
        self.sensor_publisher = self.create_publisher(String, 'sensor_data', sensor_qos)
        self.cmd_publisher = self.create_publisher(String, 'robot_commands', cmd_qos)

        # Create timers for different publishing rates
        self.sensor_timer = self.create_timer(0.1, self.publish_sensor_data)  # 10Hz
        self.cmd_timer = self.create_timer(1.0, self.publish_cmd_data)  # 1Hz

        self.sensor_counter = 0
        self.cmd_counter = 0

    def publish_sensor_data(self):
        """
        Publish sensor-like data with best-effort QoS.
        """
        msg = String()
        msg.data = f'Sensor reading: {random.random():.2f} at time {time.time():.2f}'
        self.sensor_publisher.publish(msg)
        self.get_logger().debug(f'Published sensor data: "{msg.data}"')
        self.sensor_counter += 1

    def publish_cmd_data(self):
        """
        Publish command-like data with reliable QoS.
        """
        msg = String()
        msg.data = f'Command #{self.cmd_counter}: Move to position {random.randint(1, 10)}'
        self.cmd_publisher.publish(msg)
        self.get_logger().info(f'Published command: "{msg.data}"')
        self.cmd_counter += 1


def main(args=None):
    """
    Main function to initialize and run the advanced publisher node.
    """
    rclpy.init(args=args)

    advanced_publisher = AdvancedPublisher()

    try:
        rclpy.spin(advanced_publisher)
    except KeyboardInterrupt:
        advanced_publisher.get_logger().info('Shutting down advanced publisher...')
    finally:
        advanced_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Service Server Implementation

```python
#!/usr/bin/env python3

"""
ROS 2 service server implementation.
Demonstrates request processing and response generation.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts
from std_srvs.srv import Trigger
import time


class RobotServiceServer(Node):
    """
    A service server that provides various robot-related services.
    """

    def __init__(self):
        super().__init__('robot_service_server')

        # Create multiple services
        self.add_service = self.create_service(
            AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

        self.trigger_service = self.create_service(
            Trigger, 'robot_trigger', self.trigger_callback)

        self.get_logger().info('Robot service server started')

    def add_two_ints_callback(self, request, response):
        """
        Callback for the add_two_ints service.
        """
        response.sum = request.a + request.b
        self.get_logger().info(f'Request: {request.a} + {request.b} = {response.sum}')
        return response

    def trigger_callback(self, request, response):
        """
        Callback for the trigger service (used for simple operations).
        """
        # Simulate some processing time
        time.sleep(0.1)

        response.success = True
        response.message = 'Robot operation completed successfully'
        self.get_logger().info('Trigger service called - operation completed')
        return response


def main(args=None):
    """
    Main function to initialize and run the service server node.
    """
    rclpy.init(args=args)

    robot_service_server = RobotServiceServer()

    try:
        rclpy.spin(robot_service_server)
    except KeyboardInterrupt:
        robot_service_server.get_logger().info('Shutting down service server...')
    finally:
        robot_service_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Node Lifecycle Management Example

```python
#!/usr/bin/env python3

"""
ROS 2 node with lifecycle management.
Demonstrates the lifecycle state transitions.
"""

import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile
from std_msgs.msg import String


class LifecycleManagedNode(LifecycleNode):
    """
    A lifecycle-managed node demonstrating state transitions.
    """

    def __init__(self):
        super().__init__('lifecycle_managed_node')
        self.get_logger().info('Lifecycle node created, currently inactive')
        self.pub = None

    def on_configure(self, state):
        """
        Callback for the configure transition.
        """
        self.get_logger().info(f'Configuring node, current state: {state}')

        # Create publisher in this state
        self.pub = self.create_publisher(String, 'lifecycle_chatter', QoSProfile(depth=10))
        self.get_logger().info('Publisher created')

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """
        Callback for the activate transition.
        """
        self.get_logger().info(f'Activating node, current state: {state}')

        # Activate the publisher
        self.pub.on_activate()

        # Create a timer that runs in active state
        self.timer = self.create_timer(1.0, self.timer_callback)

        return TransitionCallbackReturn.SUCCESS

    def timer_callback(self):
        """
        Timer callback that runs when node is active.
        """
        msg = String()
        msg.data = f'Lifecycle node message: {self.get_clock().now().nanoseconds}'
        self.pub.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')

    def on_deactivate(self, state):
        """
        Callback for the deactivate transition.
        """
        self.get_logger().info(f'Deactivating node, current state: {state}')

        # Deactivate the publisher
        self.pub.on_deactivate()

        # Destroy the timer
        self.destroy_timer(self.timer)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """
        Callback for the cleanup transition.
        """
        self.get_logger().info(f'Cleaning up node, current state: {state}')

        # Destroy the publisher
        self.destroy_publisher(self.pub)

        return TransitionCallbackReturn.SUCCESS


def main(args=None):
    """
    Main function to demonstrate lifecycle node management.
    """
    rclpy.init(args=args)

    lifecycle_node = LifecycleManagedNode()

    try:
        rclpy.spin(lifecycle_node)
    except KeyboardInterrupt:
        lifecycle_node.get_logger().info('Shutting down lifecycle node...')
    finally:
        lifecycle_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Labs and Exercises

### Exercise 1: Sensor Data Publisher-Subscriber
Implement a publisher-subscriber pattern for sensor data. Create a publisher that simulates sensor readings (e.g., temperature, distance, or IMU data) and a subscriber that processes and logs this data. Configure appropriate QoS settings for sensor data (best-effort, small history).

### Exercise 2: Robot Control Service
Create a service for robot control commands. Implement a service server that accepts commands (e.g., move forward, turn, stop) and returns status information. Create a client that sends commands and handles responses appropriately.

### Exercise 3: QoS Policy Experimentation
Experiment with different QoS settings in your publisher-subscriber example. Test reliability settings (best-effort vs. reliable), history policies (keep-all vs. keep-last), and durability settings. Document the effects on message delivery under different network conditions and system loads.

### Exercise 4: Lifecycle Node Implementation
Implement a lifecycle node that manages a hardware resource (e.g., a camera or sensor). The node should properly configure the resource when transitioning to the active state and release it when transitioning to the inactive state. Test the state transitions using ROS 2 lifecycle tools.