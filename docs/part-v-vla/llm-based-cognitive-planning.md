---
sidebar_position: 3
---

# LLM-Based Cognitive Planning

## Overview

Large Language Models (LLMs) have emerged as powerful tools for cognitive planning in robotics, enabling robots to reason about complex tasks, generate high-level plans, and adapt to novel situations using natural language as an interface. This chapter explores the integration of LLMs with robotic systems for cognitive planning, covering the architecture, implementation, and optimization of LLM-powered planning systems. These systems leverage the reasoning capabilities of LLMs to break down complex tasks into executable robotic actions while maintaining the flexibility to adapt to changing environments and requirements.

LLM-based cognitive planning addresses key challenges in robotics by providing a natural language interface for task specification and enabling robots to leverage pre-trained world knowledge for planning. Unlike traditional planning systems that require explicit programming for each task, LLM-based systems can generalize across different domains and adapt to novel situations by reasoning about the relationships between actions, objects, and goals. This capability is particularly valuable for service robots, assistive robotics, and other applications where robots need to operate in unstructured environments with diverse tasks.

The integration of LLMs with robotic planning systems involves several key components: natural language understanding to interpret high-level goals, plan generation to create executable action sequences, plan refinement to adapt plans based on environmental feedback, and execution monitoring to handle plan failures and replanning needs. Modern implementations leverage the extensive world knowledge encoded in LLMs to create more sophisticated and adaptable planning systems that can handle complex, multi-step tasks with minimal human intervention.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Understand the architecture of LLM-based cognitive planning systems for robotics
- Implement LLM integration with robotic planning frameworks
- Design natural language interfaces for robot task specification
- Create plan generation and refinement systems using LLMs
- Handle plan execution failures and replanning scenarios
- Evaluate the effectiveness of LLM-based planning systems
- Address challenges in grounding LLM knowledge to robotic actions

## Key Concepts

### LLM Integration with Planning Systems

Integration approaches for LLMs in robotic planning:

- **Plan Generation**: Using LLMs to generate high-level action sequences
- **Plan Refinement**: Adapting plans based on environmental constraints
- **Task Decomposition**: Breaking complex tasks into manageable subtasks
- **Knowledge Integration**: Leveraging LLM world knowledge for planning
- **Multi-step Reasoning**: Chain-of-thought reasoning for complex tasks
- **Contextual Planning**: Incorporating environmental and situational context

### Natural Language Task Specification

Methods for specifying robot tasks using natural language:

- **Goal Description**: Natural language specification of desired outcomes
- **Constraint Specification**: Natural language constraints on execution
- **Preference Expression**: Natural language preferences for plan selection
- **Temporal Specifications**: Natural language timing and sequencing requirements
- **Conditional Planning**: Natural language conditional statements
- **Exception Handling**: Natural language failure recovery instructions

### Plan Execution and Monitoring

Components for executing and monitoring LLM-generated plans:

- **Action Grounding**: Converting LLM actions to robot-executable commands
- **Execution Monitoring**: Tracking plan execution progress
- **Failure Detection**: Identifying when plans fail or need modification
- **Replanning Triggers**: Conditions that initiate plan revision
- **Feedback Integration**: Incorporating execution feedback into planning
- **Human-in-the-Loop**: Allowing human intervention during execution

### Grounding and World Modeling

Techniques for connecting LLM knowledge to physical reality:

- **Perceptual Grounding**: Connecting LLM concepts to sensor data
- **Action Grounding**: Connecting LLM actions to robot capabilities
- **Spatial Reasoning**: Grounding spatial concepts to robot coordinate systems
- **Object Affordances**: Connecting LLM object knowledge to manipulation capabilities
- **Temporal Grounding**: Connecting LLM temporal concepts to real-world time
- **Embodied Reasoning**: Grounding abstract concepts to physical actions

## Diagrams and Code

### LLM-Based Cognitive Planning Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Natural       │    │   LLM-Based     │    │   Plan          │
│   Language      │───▶│   Cognitive     │───▶│   Execution     │
│   Task          │    │   Planning      │    │   & Monitoring  │
│   Specification │    │   System        │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Task          │    │   High-Level    │    │   Low-Level     │
│   Understanding │    │   Plan          │    │   Robot         │
│   & Parsing     │    │   Generation    │    │   Commands      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Environment          │
                    │   Perception &         │
                    │   Feedback Integration │
                    └─────────────────────────┘
```

### LLM-Based Planning System Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import openai
import json
import time
from typing import Dict, List, Optional, Any
import re

class LLMBasedPlanningNode(Node):
    """
    LLM-based cognitive planning system for robotics.
    Uses Large Language Models to generate and refine robot plans.
    """

    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Publishers
        self.plan_pub = self.create_publisher(String, '/robot/plan', 10)
        self.action_pub = self.create_publisher(Pose, '/robot/action', 10)
        self.status_pub = self.create_publisher(String, '/llm_planning/status', 10)

        # Subscribers
        self.task_sub = self.create_subscription(
            String,
            '/llm_planning/task',
            self.task_callback,
            10
        )

        self.perception_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.perception_callback,
            10
        )

        # LLM configuration
        self.openai_client = None  # Will be configured with API key
        self.planning_system = LLMPlanningSystem()

        # System state
        self.current_task = None
        self.current_plan = None
        self.plan_execution_state = "idle"
        self.environment_state = {}
        self.execution_history = []

        # Plan execution parameters
        self.plan_execution_active = False
        self.current_step_index = 0

        self.get_logger().info('LLM-Based Planning Node initialized')

    def task_callback(self, msg):
        """
        Process natural language task specification
        """
        task_description = msg.data
        self.get_logger().info(f'Received task: "{task_description}"')

        # Update environment state with current information
        self.update_environment_state()

        try:
            # Generate plan using LLM
            plan = self.planning_system.generate_plan(
                task_description,
                self.environment_state
            )

            if plan:
                self.current_task = task_description
                self.current_plan = plan

                # Publish the plan
                plan_msg = String()
                plan_msg.data = json.dumps(plan)
                self.plan_pub.publish(plan_msg)

                # Update status
                status_msg = String()
                status_msg.data = f"Generated plan for: {task_description}"
                self.status_pub.publish(status_msg)

                self.get_logger().info(f'Plan generated with {len(plan.get("steps", []))} steps')

                # Start plan execution
                self.start_plan_execution()
            else:
                self.get_logger().error('Failed to generate plan')

        except Exception as e:
            self.get_logger().error(f'Plan generation error: {str(e)}')

    def perception_callback(self, msg):
        """
        Process perception data for environment state updates
        """
        try:
            # Convert image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Extract environment information (simulated)
            environment_info = self.extract_environment_info(cv_image)

            # Update environment state
            self.environment_state.update(environment_info)

            # If we're executing a plan and need replanning, trigger it
            if (self.plan_execution_active and
                self.should_replan(environment_info)):
                self.trigger_replanning()

        except Exception as e:
            self.get_logger().error(f'Perception processing error: {str(e)}')

    def extract_environment_info(self, image):
        """
        Extract environment information from image (simulated implementation)
        """
        # In real implementation, this would use computer vision to identify objects,
        # locations, obstacles, etc.
        # For simulation, return dummy information
        return {
            "objects": [
                {"name": "red_cup", "position": [0.5, 0.5, 0.0], "type": "graspable"},
                {"name": "blue_box", "position": [1.0, 0.0, 0.0], "type": "container"},
                {"name": "table", "position": [0.0, 0.0, 0.0], "type": "surface"}
            ],
            "obstacles": [
                {"position": [0.8, 0.3, 0.0], "size": [0.2, 0.2, 0.5]}
            ],
            "robot_position": [0.0, 0.0, 0.0],
            "timestamp": time.time()
        }

    def update_environment_state(self):
        """
        Update environment state with current information
        """
        # In real implementation, this would query current sensor data
        # For simulation, we'll keep the state updated from perception
        pass

    def should_replan(self, new_environment_info):
        """
        Determine if replanning is needed based on environment changes
        """
        # Check if critical environment information has changed
        # For simulation, replan if new objects are detected
        new_objects = new_environment_info.get("objects", [])
        current_objects = self.environment_state.get("objects", [])

        return len(new_objects) != len(current_objects)

    def trigger_replanning(self):
        """
        Trigger replanning based on environment changes
        """
        if self.current_task:
            self.get_logger().info('Environment change detected, triggering replanning')

            try:
                # Generate new plan with updated environment state
                new_plan = self.planning_system.generate_plan(
                    self.current_task,
                    self.environment_state
                )

                if new_plan:
                    self.current_plan = new_plan
                    # Publish updated plan
                    plan_msg = String()
                    plan_msg.data = json.dumps(new_plan)
                    self.plan_pub.publish(plan_msg)

                    self.get_logger().info('Replanning completed')
                else:
                    self.get_logger().error('Replanning failed')

            except Exception as e:
                self.get_logger().error(f'Replanning error: {str(e)}')

    def start_plan_execution(self):
        """
        Start executing the current plan
        """
        if self.current_plan and len(self.current_plan.get('steps', [])) > 0:
            self.plan_execution_active = True
            self.current_step_index = 0

            # Execute the first step
            self.execute_next_step()

    def execute_next_step(self):
        """
        Execute the next step in the plan
        """
        if not self.current_plan or not self.plan_execution_active:
            return

        steps = self.current_plan.get('steps', [])
        if self.current_step_index >= len(steps):
            # Plan completed
            self.plan_execution_active = False
            self.get_logger().info('Plan execution completed')
            return

        current_step = steps[self.current_step_index]

        # Execute the step (for simulation, we'll just log it)
        self.execute_step(current_step)

        # Move to next step
        self.current_step_index += 1

        # Schedule next step execution
        if self.current_step_index < len(steps):
            # In real implementation, this would be based on step completion
            # For simulation, execute next step after a delay
            timer = self.create_timer(1.0, self.execute_next_step)
        else:
            self.get_logger().info('All plan steps completed')

    def execute_step(self, step):
        """
        Execute a single plan step
        """
        step_type = step.get('type', 'unknown')
        step_params = step.get('parameters', {})

        self.get_logger().info(f'Executing step: {step_type} with params: {step_params}')

        # In real implementation, this would execute the actual robot action
        # For simulation, we'll just log the action
        if step_type == 'navigate':
            target = step_params.get('target_position')
            self.get_logger().info(f'Navigating to {target}')
        elif step_type == 'grasp':
            target = step_params.get('target_object')
            self.get_logger().info(f'Grasping {target}')
        elif step_type == 'place':
            target = step_params.get('target_location')
            self.get_logger().info(f'Placing at {target}')
        elif step_type == 'detect':
            target = step_params.get('target_object')
            self.get_logger().info(f'Detecting {target}')

        # Add to execution history
        self.execution_history.append({
            'step': step,
            'timestamp': time.time(),
            'status': 'completed'
        })

def main(args=None):
    rclpy.init(args=args)

    llm_planning_node = LLMBasedPlanningNode()

    try:
        rclpy.spin(llm_planning_node)
    except KeyboardInterrupt:
        pass
    finally:
        llm_planning_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### LLM Planning System with Chain-of-Thought Reasoning

```python
import openai
import json
import re
from typing import Dict, List, Optional, Any
import time

class LLMPlanningSystem:
    """
    LLM-based planning system with chain-of-thought reasoning.
    Generates robot plans using large language models with step-by-step reasoning.
    """

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            openai.api_key = api_key
        self.default_model = "gpt-3.5-turbo"
        self.planning_schema = self._create_planning_schema()

    def _create_planning_schema(self) -> Dict:
        """
        Create schema for structured planning output.
        """
        return {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning for the plan"
                },
                "plan": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["navigate", "grasp", "place", "detect", "turn", "wait", "other"]
                                    },
                                    "description": {"type": "string"},
                                    "parameters": {"type": "object"},
                                    "preconditions": {"type": "array", "items": {"type": "string"}},
                                    "effects": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["id", "type", "description", "parameters"]
                            }
                        },
                        "estimated_duration": {"type": "number"}
                    },
                    "required": ["goal", "steps"]
                }
            },
            "required": ["reasoning", "plan"]
        }

    def generate_plan(self, task_description: str, environment_state: Dict) -> Optional[Dict]:
        """
        Generate a plan for the given task using LLM reasoning.
        """
        try:
            # Create a detailed prompt for the LLM
            prompt = self._create_planning_prompt(task_description, environment_state)

            response = openai.ChatCompletion.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": self._get_planning_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Validate and return the plan
            return self._validate_plan(result)

        except Exception as e:
            print(f"LLM planning failed: {str(e)}")
            # Return a fallback plan
            return self._generate_fallback_plan(task_description)

    def _create_planning_prompt(self, task_description: str, environment_state: Dict) -> str:
        """
        Create prompt for LLM-based planning.
        """
        environment_str = json.dumps(environment_state, indent=2)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        prompt = f"""
        Current Time: {current_time}

        Environment State:
        {environment_str}

        Task: "{task_description}"

        Please generate a detailed plan to accomplish this task. Think step-by-step:

        1. Analyze the environment and identify relevant objects, locations, and obstacles
        2. Break down the task into specific, executable steps
        3. Consider preconditions and effects of each action
        4. Account for the robot's capabilities and environmental constraints
        5. Generate a sequence of actions that accomplishes the goal

        Provide your response in the following JSON format:
        {{
            "reasoning": "Step-by-step reasoning explaining your plan",
            "plan": {{
                "goal": "Clear description of the task goal",
                "steps": [
                    {{
                        "id": 1,
                        "type": "action_type",
                        "description": "Human-readable description of the action",
                        "parameters": {{"param1": "value1", "param2": "value2"}},
                        "preconditions": ["condition1", "condition2"],
                        "effects": ["effect1", "effect2"]
                    }}
                ],
                "estimated_duration": 0.0
            }}
        }}

        Action types should be one of: navigate, grasp, place, detect, turn, wait, other
        Each step should be specific enough for a robot to execute.
        """

        return prompt

    def _get_planning_system_prompt(self) -> str:
        """
        Get system prompt for planning task.
        """
        return """
        You are an expert robotic planning system. Your task is to generate detailed, executable plans for robots based on natural language task descriptions. Consider the environment state, robot capabilities, and task requirements when creating plans. Think step-by-step and provide clear reasoning for your plan. Each plan step should be specific, executable, and include necessary parameters for robot execution.
        """

    def _validate_plan(self, plan_data: Dict) -> Optional[Dict]:
        """
        Validate the generated plan and ensure it meets requirements.
        """
        try:
            plan = plan_data.get('plan', {})
            steps = plan.get('steps', [])

            # Validate required fields
            if not plan.get('goal'):
                print("Plan validation failed: Missing goal")
                return None

            if not steps:
                print("Plan validation failed: No steps provided")
                return None

            # Validate each step
            for i, step in enumerate(steps):
                required_fields = ['id', 'type', 'description', 'parameters']
                for field in required_fields:
                    if field not in step:
                        print(f"Plan validation failed: Step {i} missing field '{field}'")
                        return None

                # Validate step type
                valid_types = ['navigate', 'grasp', 'place', 'detect', 'turn', 'wait', 'other']
                if step['type'] not in valid_types:
                    print(f"Plan validation failed: Invalid step type '{step['type']}'")
                    return None

            return plan_data

        except Exception as e:
            print(f"Plan validation error: {str(e)}")
            return None

    def _generate_fallback_plan(self, task_description: str) -> Optional[Dict]:
        """
        Generate a simple fallback plan when LLM fails.
        """
        # Simple keyword-based fallback planning
        task_lower = task_description.lower()

        steps = []
        step_id = 1

        if 'go to' in task_lower or 'navigate to' in task_lower:
            steps.append({
                "id": step_id,
                "type": "navigate",
                "description": f"Navigate to target location",
                "parameters": {"target": "default_location"},
                "preconditions": [],
                "effects": ["robot_at_target"]
            })
            step_id += 1

        if 'grasp' in task_lower or 'pick up' in task_lower:
            steps.append({
                "id": step_id,
                "type": "grasp",
                "description": f"Grasp target object",
                "parameters": {"target": "default_object"},
                "preconditions": ["object_detected"],
                "effects": ["object_grasped"]
            })
            step_id += 1

        if 'place' in task_lower or 'put' in task_lower:
            steps.append({
                "id": step_id,
                "type": "place",
                "description": f"Place object at location",
                "parameters": {"target": "default_location"},
                "preconditions": ["object_grasped"],
                "effects": ["object_placed"]
            })
            step_id += 1

        if not steps:
            # Default navigation step
            steps.append({
                "id": step_id,
                "type": "navigate",
                "description": "Move to default location",
                "parameters": {"target": "default_location"},
                "preconditions": [],
                "effects": ["robot_moved"]
            })

        return {
            "reasoning": "Fallback plan generated due to LLM unavailability",
            "plan": {
                "goal": task_description,
                "steps": steps,
                "estimated_duration": len(steps) * 10.0  # Estimate 10 seconds per step
            }
        }

    def refine_plan(self, original_plan: Dict, feedback: Dict) -> Optional[Dict]:
        """
        Refine an existing plan based on execution feedback.
        """
        try:
            original_steps = original_plan['plan']['steps']
            feedback_str = json.dumps(feedback, indent=2)

            prompt = f"""
            Original Plan:
            {json.dumps(original_plan, indent=2)}

            Execution Feedback:
            {feedback_str}

            Please refine the plan based on the execution feedback. Consider:
            1. What went wrong or needs adjustment
            2. How to modify the plan to handle the feedback
            3. Whether to skip, repeat, or modify steps
            4. How to achieve the goal with the new information

            Return the refined plan in the same format as the original.
            """

            response = openai.ChatCompletion.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": self._get_planning_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return self._validate_plan(result)

        except Exception as e:
            print(f"Plan refinement failed: {str(e)}")
            return original_plan  # Return original if refinement fails

    def evaluate_plan(self, plan: Dict, environment_state: Dict) -> Dict:
        """
        Evaluate the quality and feasibility of a plan.
        """
        try:
            steps = plan['plan']['steps']

            evaluation = {
                "feasibility": 0.0,
                "complexity": len(steps),
                "estimated_time": plan['plan'].get('estimated_duration', len(steps) * 10.0),
                "risk_factors": [],
                "confidence": 0.8  # Default confidence
            }

            # Analyze risk factors
            for step in steps:
                step_type = step['type']

                if step_type == 'grasp':
                    evaluation['risk_factors'].append("Manipulation risk")
                elif step_type == 'navigate':
                    evaluation['risk_factors'].append("Navigation risk")
                elif step_type == 'place':
                    evaluation['risk_factors'].append("Placement risk")

            # Estimate feasibility based on environment
            objects_in_env = len(environment_state.get('objects', []))
            if objects_in_env == 0:
                evaluation['risk_factors'].append("Limited environmental information")

            # Calculate feasibility score
            risk_score = len(evaluation['risk_factors']) / 10.0
            evaluation['feasibility'] = max(0.0, 1.0 - risk_score)

            return evaluation

        except Exception as e:
            print(f"Plan evaluation error: {str(e)}")
            return {
                "feasibility": 0.5,
                "complexity": 0,
                "estimated_time": 0.0,
                "risk_factors": ["Evaluation error"],
                "confidence": 0.0
            }

# Example usage
def example_usage():
    """
    Example of using the LLM-based planning system.
    """
    # Initialize planning system (without API key, it will use fallback)
    planner = LLMPlanningSystem()

    # Example task
    task = "Go to the kitchen, find the red cup, grasp it, and place it on the dining table"

    # Example environment state
    environment = {
        "objects": [
            {"name": "red_cup", "position": [1.5, 0.5, 0.0], "type": "graspable"},
            {"name": "dining_table", "position": [2.0, 1.0, 0.0], "type": "surface"}
        ],
        "locations": [
            {"name": "kitchen", "position": [1.0, 0.0, 0.0]},
            {"name": "dining_area", "position": [2.0, 1.0, 0.0]}
        ],
        "robot_position": [0.0, 0.0, 0.0]
    }

    print(f"Generating plan for task: '{task}'")

    plan = planner.generate_plan(task, environment)

    if plan:
        print("\nGenerated Plan:")
        print(json.dumps(plan, indent=2))

        # Evaluate the plan
        evaluation = planner.evaluate_plan(plan, environment)
        print(f"\nPlan Evaluation: {evaluation}")
    else:
        print("Failed to generate plan")

if __name__ == "__main__":
    example_usage()
```

### Plan Execution and Monitoring System

```python
import time
import threading
from typing import Dict, List, Optional, Callable
import json

class PlanExecutionMonitor:
    """
    Monitor and execute plans generated by LLM-based planning system.
    Handles plan execution, monitoring, and replanning when needed.
    """

    def __init__(self, robot_interface: Optional[object] = None):
        self.robot_interface = robot_interface
        self.current_plan = None
        self.current_step_index = 0
        self.execution_state = "idle"  # idle, executing, paused, failed, completed
        self.execution_history = []
        self.plan_callbacks = {
            'step_started': [],
            'step_completed': [],
            'plan_completed': [],
            'plan_failed': []
        }

        # Execution parameters
        self.max_execution_time = 300  # 5 minutes
        self.step_timeout = 60  # 1 minute per step
        self.replan_threshold = 0.3  # Replan if confidence drops below this

    def execute_plan(self, plan: Dict) -> bool:
        """
        Execute a plan with monitoring and error handling.
        """
        if not plan or 'steps' not in plan.get('plan', {}):
            print("Invalid plan provided for execution")
            return False

        self.current_plan = plan
        self.current_step_index = 0
        self.execution_state = "executing"
        self.execution_start_time = time.time()

        print(f"Starting execution of plan with {len(plan['plan']['steps'])} steps")

        # Execute steps sequentially
        success = True
        for i, step in enumerate(plan['plan']['steps']):
            if self.execution_state != "executing":
                break

            step_success = self.execute_single_step(step, i)
            if not step_success:
                success = False
                self.execution_state = "failed"
                break

        if success and self.execution_state == "executing":
            self.execution_state = "completed"
            self._trigger_callbacks('plan_completed', plan)

        return success

    def execute_single_step(self, step: Dict, step_index: int) -> bool:
        """
        Execute a single plan step with monitoring.
        """
        print(f"Executing step {step_index + 1}: {step['description']}")

        # Record step start
        step_record = {
            'step': step,
            'step_index': step_index,
            'start_time': time.time(),
            'status': 'started'
        }

        # Trigger step started callback
        self._trigger_callbacks('step_started', step, step_index)

        # Execute the step based on its type
        try:
            execution_result = self._execute_step_type(step)

            # Record completion
            step_record['end_time'] = time.time()
            step_record['duration'] = step_record['end_time'] - step_record['start_time']
            step_record['status'] = 'completed' if execution_result['success'] else 'failed'
            step_record['result'] = execution_result

            # Trigger step completed callback
            self._trigger_callbacks('step_completed', step, step_index, execution_result)

        except Exception as e:
            step_record['end_time'] = time.time()
            step_record['duration'] = step_record['end_time'] - step_record['start_time']
            step_record['status'] = 'failed'
            step_record['error'] = str(e)
            step_record['result'] = {'success': False, 'error': str(e)}

            print(f"Step {step_index + 1} failed: {str(e)}")
            return False

        # Add to execution history
        self.execution_history.append(step_record)

        return execution_result['success']

    def _execute_step_type(self, step: Dict) -> Dict:
        """
        Execute a step based on its type.
        """
        step_type = step['type']
        parameters = step.get('parameters', {})

        print(f"Executing {step_type} step with parameters: {parameters}")

        # In a real implementation, these would interface with actual robot commands
        # For simulation, we'll return success/failure based on step type
        if step_type == 'navigate':
            return self._execute_navigate_step(parameters)
        elif step_type == 'grasp':
            return self._execute_grasp_step(parameters)
        elif step_type == 'place':
            return self._execute_place_step(parameters)
        elif step_type == 'detect':
            return self._execute_detect_step(parameters)
        elif step_type == 'turn':
            return self._execute_turn_step(parameters)
        elif step_type == 'wait':
            return self._execute_wait_step(parameters)
        else:
            # For 'other' or unknown types, return success
            time.sleep(1)  # Simulate execution time
            return {'success': True, 'details': f'Executed {step_type} step'}

    def _execute_navigate_step(self, parameters: Dict) -> Dict:
        """
        Execute navigation step.
        """
        target = parameters.get('target', 'unknown')
        print(f"Navigating to {target}")

        # Simulate navigation
        time.sleep(2)  # Simulate navigation time

        # Simulate success/failure (90% success rate)
        import random
        success = random.random() > 0.1

        return {
            'success': success,
            'details': f'Navigation to {target} {"succeeded" if success else "failed"}',
            'actual_position': [0.0, 0.0, 0.0] if success else None
        }

    def _execute_grasp_step(self, parameters: Dict) -> Dict:
        """
        Execute grasping step.
        """
        target = parameters.get('target', 'unknown')
        print(f"Attempting to grasp {target}")

        # Simulate grasping
        time.sleep(1.5)  # Simulate grasping time

        # Simulate success/failure (75% success rate)
        import random
        success = random.random() > 0.25

        return {
            'success': success,
            'details': f'Grasping of {target} {"succeeded" if success else "failed"}',
            'object_grasped': success
        }

    def _execute_place_step(self, parameters: Dict) -> Dict:
        """
        Execute placement step.
        """
        target = parameters.get('target', 'unknown')
        print(f"Attempting to place at {target}")

        # Simulate placement
        time.sleep(1.5)  # Simulate placement time

        # Simulate success/failure (85% success rate)
        import random
        success = random.random() > 0.15

        return {
            'success': success,
            'details': f'Placement at {target} {"succeeded" if success else "failed"}',
            'object_placed': success
        }

    def _execute_detect_step(self, parameters: Dict) -> Dict:
        """
        Execute detection step.
        """
        target = parameters.get('target', 'unknown')
        print(f"Detecting {target}")

        # Simulate detection
        time.sleep(1)  # Simulate detection time

        # Simulate success/failure (95% success rate)
        import random
        success = random.random() > 0.05

        return {
            'success': success,
            'details': f'Detection of {target} {"succeeded" if success else "failed"}',
            'object_detected': success
        }

    def _execute_turn_step(self, parameters: Dict) -> Dict:
        """
        Execute turning step.
        """
        direction = parameters.get('direction', 'right')
        angle = parameters.get('angle', 90)
        print(f"Turning {direction} by {angle} degrees")

        # Simulate turning
        time.sleep(1)  # Simulate turning time

        return {
            'success': True,
            'details': f'Turned {direction} by {angle} degrees',
            'final_orientation': [0.0, 0.0, 0.0]
        }

    def _execute_wait_step(self, parameters: Dict) -> Dict:
        """
        Execute wait step.
        """
        duration = parameters.get('duration', 1.0)
        print(f"Waiting for {duration} seconds")

        time.sleep(duration)  # Actual wait

        return {
            'success': True,
            'details': f'Waited for {duration} seconds'
        }

    def add_callback(self, event_type: str, callback: Callable):
        """
        Add a callback for plan execution events.
        """
        if event_type in self.plan_callbacks:
            self.plan_callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def _trigger_callbacks(self, event_type: str, *args):
        """
        Trigger callbacks for a specific event type.
        """
        if event_type in self.plan_callbacks:
            for callback in self.plan_callbacks[event_type]:
                try:
                    callback(*args)
                except Exception as e:
                    print(f"Callback error for {event_type}: {str(e)}")

    def pause_execution(self):
        """
        Pause current plan execution.
        """
        if self.execution_state == "executing":
            self.execution_state = "paused"
            print("Plan execution paused")

    def resume_execution(self):
        """
        Resume paused plan execution.
        """
        if self.execution_state == "paused":
            self.execution_state = "executing"
            print("Plan execution resumed")

    def stop_execution(self):
        """
        Stop current plan execution.
        """
        self.execution_state = "idle"
        print("Plan execution stopped")

    def get_execution_status(self) -> Dict:
        """
        Get current execution status.
        """
        return {
            'state': self.execution_state,
            'current_step': self.current_step_index,
            'total_steps': len(self.current_plan['plan']['steps']) if self.current_plan else 0,
            'execution_time': time.time() - self.execution_start_time if hasattr(self, 'execution_start_time') else 0,
            'history': self.execution_history
        }

    def should_replan(self, feedback: Dict) -> bool:
        """
        Determine if replanning is needed based on feedback.
        """
        if not feedback:
            return False

        # Check for critical failures
        critical_failures = ['collision', 'grasp_failure', 'navigation_failure']
        if any(fail_type in feedback for fail_type in critical_failures):
            return True

        # Check execution time vs plan estimate
        if (self.current_plan and
            'estimated_duration' in self.current_plan['plan']):
            estimated = self.current_plan['plan']['estimated_duration']
            actual = time.time() - self.execution_start_time
            if actual > estimated * 3:  # 3x longer than estimated
                return True

        return False

# Example usage and integration
def example_integration():
    """
    Example of integrating plan execution with monitoring.
    """
    # Create a simple callback to monitor execution
    def step_completed_callback(step, step_index, result):
        print(f"Step {step_index + 1} completed: {result['success']}")

    def plan_completed_callback(plan):
        print(f"Plan completed successfully!")
        print(f"Executed {len(plan['plan']['steps'])} steps")

    # Create monitor
    monitor = PlanExecutionMonitor()
    monitor.add_callback('step_completed', step_completed_callback)
    monitor.add_callback('plan_completed', plan_completed_callback)

    # Example plan (similar to what LLM planning system would generate)
    example_plan = {
        "reasoning": "Go to kitchen, find red cup, grasp it, place on table",
        "plan": {
            "goal": "Move red cup from kitchen to dining table",
            "steps": [
                {
                    "id": 1,
                    "type": "navigate",
                    "description": "Navigate to kitchen area",
                    "parameters": {"target": "kitchen"},
                    "preconditions": [],
                    "effects": ["robot_in_kitchen"]
                },
                {
                    "id": 2,
                    "type": "detect",
                    "description": "Detect red cup",
                    "parameters": {"target": "red_cup"},
                    "preconditions": ["robot_in_kitchen"],
                    "effects": ["cup_location_known"]
                },
                {
                    "id": 3,
                    "type": "grasp",
                    "description": "Grasp the red cup",
                    "parameters": {"target": "red_cup"},
                    "preconditions": ["cup_location_known"],
                    "effects": ["cup_grasped"]
                },
                {
                    "id": 4,
                    "type": "navigate",
                    "description": "Navigate to dining table",
                    "parameters": {"target": "dining_table"},
                    "preconditions": ["cup_grasped"],
                    "effects": ["robot_at_table"]
                },
                {
                    "id": 5,
                    "type": "place",
                    "description": "Place cup on dining table",
                    "parameters": {"target": "dining_table"},
                    "preconditions": ["robot_at_table", "cup_grasped"],
                    "effects": ["cup_placed"]
                }
            ],
            "estimated_duration": 120.0
        }
    }

    print("Executing example plan...")
    success = monitor.execute_plan(example_plan)
    print(f"Plan execution {'succeeded' if success else 'failed'}")

    # Get execution status
    status = monitor.get_execution_status()
    print(f"Execution status: {status['state']}")
    print(f"Steps completed: {len(status['history'])}/{status['total_steps']}")

if __name__ == "__main__":
    example_integration()
```

## Labs and Exercises

### Exercise 1: Chain-of-Thought Planning Implementation
Implement a chain-of-thought reasoning system for robotic planning that breaks down complex tasks into logical steps. Evaluate the system's ability to handle multi-step tasks and compare its performance with traditional planning approaches.

### Exercise 2: Plan Refinement with Feedback
Create a plan refinement system that adapts LLM-generated plans based on execution feedback. Implement mechanisms for detecting plan failures and generating alternative approaches when the original plan becomes infeasible.

### Exercise 3: Context-Aware Planning
Develop a context-aware planning system that incorporates environmental information into LLM-based planning. Test the system's ability to generate plans that account for object locations, obstacles, and other environmental factors.

### Exercise 4: Human-in-the-Loop Planning
Implement a human-in-the-loop planning system that allows humans to provide feedback and corrections during plan execution. Create interfaces for plan modification and validation during execution.

## Summary

This chapter explored LLM-based cognitive planning systems for robotics, demonstrating how large language models can be integrated with robotic planning frameworks to create more flexible and adaptable robotic systems. We covered the architecture of LLM-based planning systems, implemented chain-of-thought reasoning for complex task decomposition, and created plan execution and monitoring systems. The examples showed how LLMs can provide natural language interfaces for robot task specification while maintaining the ability to handle complex, multi-step tasks with environmental awareness. As we continue in this book, we'll explore additional aspects of Vision-Language-Action systems and their integration with robotic platforms.