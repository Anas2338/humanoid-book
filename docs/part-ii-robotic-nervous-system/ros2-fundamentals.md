---
sidebar_position: 1
---

# ROS 2 Fundamentals

## Overview

Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster. Unlike traditional operating systems, ROS 2 is middleware that provides hardware abstraction, device drivers, libraries, visualizers, message-passing, package management, and more. It serves as the "nervous system" for robotic applications, enabling different software components to communicate and coordinate effectively.

ROS 2 represents a significant evolution from its predecessor ROS 1, addressing critical issues related to security, real-time performance, and multi-robot systems. It provides a robust foundation for developing complex robotic applications by abstracting low-level communication details and providing standardized interfaces for common robotic tasks.

The architecture of ROS 2 is built on Data Distribution Service (DDS), a middleware standard that enables reliable, real-time communication between distributed systems. This design choice provides ROS 2 with enhanced capabilities for security, quality of service (QoS) controls, and robust inter-process communication that is essential for safety-critical robotic applications.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Explain the architecture and design principles of ROS 2
- Identify and describe the core concepts: nodes, topics, services, and actions
- Compare and contrast ROS 1 and ROS 2 architectures and capabilities
- Understand Quality of Service (QoS) policies and their importance in robotic systems
- Appreciate the role of DDS in enabling robust robotic communication
- Design basic ROS 2 applications using the core communication patterns

## Key Concepts

### ROS 2 Architecture and Middleware

ROS 2 is built on a distributed architecture where multiple processes (nodes) communicate through a publish-subscribe pattern. The core architectural elements include:

- **DDS Implementation**: Underlying middleware that handles message passing
- **Nodes**: Individual processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request-response communication
- **Actions**: Asynchronous goal-oriented communication for long-running tasks

### Nodes and Processes

A node is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of ROS 2 applications:

- Each node runs as a separate process
- Nodes can be written in different programming languages (C++, Python, etc.)
- Nodes communicate with each other through topics, services, and actions
- A single robot application typically consists of many nodes working together

### Topics and Message Passing

Topics enable asynchronous communication between nodes using a publish-subscribe pattern:

- Publishers send messages to topics
- Subscribers receive messages from topics
- Multiple publishers and subscribers can exist for the same topic
- Communication is decoupled in time and space
- Quality of Service (QoS) policies control message delivery characteristics

### Services and Request-Response Patterns

Services provide synchronous communication for request-response interactions:

- A service client sends a request and waits for a response
- A service server processes requests and sends responses
- Communication is synchronous and blocking
- Useful for operations that have a clear beginning and end

### Actions for Long-Running Tasks

Actions are designed for long-running, goal-oriented tasks that require feedback:

- Goals: Requests for long-running tasks
- Results: Final outcomes of completed tasks
- Feedback: Periodic updates during task execution
- Cancelation: Ability to cancel running tasks
- Useful for navigation, manipulation, and other complex operations

## Diagrams and Code

### ROS 2 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node A        │    │   Node B        │    │   Node C        │
│                 │    │                 │    │                 │
│ Publisher       │    │ Subscriber      │    │ Service Server  │
│ (Topic: /cmd)   │    │ (Topic: /cmd)   │    │ (Service: /nav) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   DDS Layer     │
                    │   (Middleware)  │
                    └─────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Network/Inter-Process              │
         │              Communication Layer                │
         └─────────────────────────────────────────────────┘
```

### Basic ROS 2 Publisher Example

```python
#!/usr/bin/env python3

"""
Simple ROS 2 publisher example that publishes messages to a topic.
This demonstrates the basic publisher-subscriber pattern in ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class MinimalPublisher(Node):
    """
    A simple ROS 2 publisher node that sends messages to a topic.
    """

    def __init__(self):
        super().__init__('minimal_publisher')

        # Create a publisher that publishes String messages to the 'topic' topic
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Create a timer that calls the timer_callback method every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter to keep track of message number
        self.i = 0

    def timer_callback(self):
        """
        Callback function that is called by the timer.
        Publishes a message to the topic.
        """
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    """
    Main function to initialize and run the publisher node.
    """
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    try:
        # Spin the node so the callback function is called
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Basic ROS 2 Subscriber Example

```python
#!/usr/bin/env python3

"""
Simple ROS 2 subscriber example that listens to messages from a topic.
This demonstrates the basic publisher-subscriber pattern in ROS 2.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    """
    A simple ROS 2 subscriber node that receives messages from a topic.
    """

    def __init__(self):
        super().__init__('minimal_subscriber')

        # Create a subscription to the 'topic' topic with String messages
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)  # QoS depth (number of messages to buffer)

        # Prevent unused variable warning
        self.subscription  # type: ignore

    def listener_callback(self, msg):
        """
        Callback function that is called when a message is received.
        """
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """
    Main function to initialize and run the subscriber node.
    """
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    try:
        # Spin the node so the callback function is called
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### ROS 2 Service Client Example

```python
#!/usr/bin/env python3

"""
Simple ROS 2 service client example.
This demonstrates the request-response pattern in ROS 2.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalClientAsync(Node):
    """
    A simple ROS 2 service client node.
    """

    def __init__(self):
        super().__init__('minimal_client_async')

        # Create a client for the 'add_two_ints' service
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        """
        Send a request to the service.
        """
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    """
    Main function to initialize and run the client node.
    """
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)

    if response is not None:
        minimal_client.get_logger().info(
            f'Result of add_two_ints: {response.sum}')
    else:
        minimal_client.get_logger().info('Service call failed')

    # Destroy the node explicitly
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Labs and Exercises

### Exercise 1: ROS 2 Environment Setup
Set up a complete ROS 2 development environment on your system. Install the latest ROS 2 distribution (Humble Hawksbill or later) and verify the installation by running basic ROS 2 commands like `ros2 topic list` and `ros2 node list`.

### Exercise 2: Publisher-Subscriber Implementation
Implement the publisher and subscriber code examples provided above. Run both nodes simultaneously and observe the message passing between them. Experiment with different QoS settings and observe their effects on message delivery.

### Exercise 3: Service Implementation
Create a custom service definition that performs a useful operation for robotics (e.g., converting coordinate frames, calculating robot kinematics, or processing sensor data). Implement both the service server and client, and test their interaction.

### Exercise 4: Quality of Service Exploration
Experiment with different QoS policies (reliability, durability, history, etc.) in your publisher-subscriber example. Document how each policy affects message delivery under different network conditions and system loads.