---
sidebar_position: 2
---

# Voice-to-Action Systems

## Overview

Voice-to-Action systems represent a critical component of Vision-Language-Action (VLA) frameworks, enabling robots to interpret natural language commands and translate them into executable robotic actions. This chapter explores the architecture, implementation, and optimization of voice-to-action systems that allow robots to understand spoken instructions and perform corresponding tasks. These systems bridge the gap between human natural language and robotic capabilities, making robots more accessible and intuitive to operate.

The core challenge in voice-to-action systems lies in accurately converting spoken language into structured commands that robots can understand and execute. This involves multiple stages: speech recognition to convert audio to text, natural language understanding to extract meaning and intent, and action mapping to generate appropriate robotic behaviors. Modern voice-to-action systems leverage large language models and multimodal AI to create more sophisticated and flexible command interpretation capabilities.

Voice-to-action systems are particularly valuable in scenarios where traditional interfaces are impractical or where natural human-robot interaction is desired. Applications include assistive robotics for elderly care, collaborative manufacturing, search and rescue operations, and domestic service robots. The effectiveness of these systems depends on their ability to handle diverse accents, ambient noise, ambiguous instructions, and complex task specifications while maintaining real-time performance.

## Learning Outcomes

By the end of this chapter, you should be able to:

- Implement speech recognition and natural language understanding pipelines
- Design intent extraction and command mapping systems for robotics
- Integrate voice-to-action systems with robotic control frameworks
- Handle ambiguous or incomplete voice commands in robotics contexts
- Optimize voice-to-action systems for real-time performance
- Evaluate the accuracy and robustness of voice command interpretation
- Address challenges in noisy environments and diverse user populations

## Key Concepts

### Speech Recognition and Processing

Core components of speech recognition in robotics:

- **Audio Preprocessing**: Noise reduction, echo cancellation, and audio enhancement
- **Acoustic Modeling**: Converting audio signals to phonetic representations
- **Language Modeling**: Converting phonetic sequences to text with context
- **Speech-to-Text APIs**: Integration with cloud or on-device speech recognition
- **Real-time Processing**: Low-latency audio processing for interactive systems
- **Multi-microphone Arrays**: Beamforming and spatial audio processing

### Natural Language Understanding for Robotics

Techniques for interpreting human commands:

- **Intent Classification**: Identifying the high-level goal of a command
- **Entity Extraction**: Recognizing objects, locations, and parameters
- **Semantic Parsing**: Converting natural language to structured representations
- **Context Awareness**: Understanding commands in environmental context
- **Ambiguity Resolution**: Handling unclear or incomplete instructions
- **Dialogue Management**: Maintaining conversational context over time

### Action Mapping and Execution

Converting language understanding to robot actions:

- **Command Vocabulary**: Mapping language constructs to robot capabilities
- **Parameter Extraction**: Identifying specific values and targets from commands
- **Action Sequencing**: Breaking complex commands into executable steps
- **Constraint Checking**: Validating actions against robot and environment constraints
- **Error Recovery**: Handling failed or unsafe action attempts
- **Feedback Generation**: Providing status updates to users

### Robustness and Adaptation

Ensuring reliable operation in diverse conditions:

- **Noise Robustness**: Handling acoustic interference and environmental noise
- **Accented Speech**: Supporting diverse linguistic backgrounds
- **Domain Adaptation**: Adapting to specific application contexts
- **Online Learning**: Improving understanding through interaction
- **Fallback Mechanisms**: Graceful degradation when understanding fails
- **User Feedback Integration**: Learning from user corrections and preferences

## Diagrams and Code

### Voice-to-Action System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│   Speech        │───▶│   Natural       │
│   (Microphone,  │    │   Recognition   │    │   Language      │
│   Audio Stream) │    │   (STT)         │    │   Understanding │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio         │    │   Transcribed   │    │   Parsed        │
│   Preprocessing │    │   Text          │    │   Commands      │
│   & Enhancement │    │   Processing    │    │   & Intent      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Action Mapping &      │
                    │   Robot Command         │
                    │   Generation            │
                    └─────────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Robot Execution       │
                    │   & Feedback System     │
                    └─────────────────────────┘
```

### Voice-to-Action Processing Pipeline

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header
import speech_recognition as sr
import numpy as np
import json
import time
import threading
from queue import Queue
import openai  # For advanced language understanding
import re

class VoiceToActionNode(Node):
    """
    Voice-to-Action system for robotics applications.
    Processes voice commands and translates them to robot actions.
    """

    def __init__(self):
        super().__init__('voice_to_action_node')

        # Publishers
        self.command_pub = self.create_publisher(String, '/robot/command', 10)
        self.status_pub = self.create_publisher(String, '/voice_to_action/status', 10)
        self.action_pub = self.create_publisher(Pose, '/robot/action', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData,
            '/audio_data',
            self.audio_callback,
            10
        )

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Set energy threshold for silence detection
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True

        # Voice processing components
        self.command_queue = Queue()
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

        # Robot command vocabulary
        self.command_vocabulary = {
            'move': ['move', 'go', 'navigate', 'go to', 'move to'],
            'grasp': ['grasp', 'grab', 'pick up', 'take', 'catch'],
            'place': ['place', 'put', 'set down', 'release', 'drop'],
            'turn': ['turn', 'rotate', 'spin', 'face'],
            'stop': ['stop', 'halt', 'pause', 'cease'],
            'find': ['find', 'locate', 'look for', 'search for']
        }

        # Object and location references
        self.known_objects = {
            'red cup': 'red_cup_1',
            'blue box': 'blue_box_1',
            'green ball': 'green_ball_1',
            'table': 'table_1',
            'chair': 'chair_1'
        }

        # Location references
        self.known_locations = {
            'kitchen': 'kitchen_area',
            'living room': 'living_room_area',
            'bedroom': 'bedroom_area',
            'office': 'office_area',
            'dining table': 'dining_table_position',
            'couch': 'couch_position'
        }

        # System state
        self.listening_active = False
        self.last_command_time = time.time()
        self.command_history = []

        # Start voice processing
        self.start_listening()

        self.get_logger().info('Voice-to-Action Node initialized')

    def start_listening(self):
        """
        Start continuous listening for voice commands
        """
        self.listening_active = True
        self.get_logger().info('Voice-to-Action system started listening')

    def stop_listening(self):
        """
        Stop listening for voice commands
        """
        self.listening_active = False
        self.get_logger().info('Voice-to-Action system stopped listening')

    def audio_callback(self, msg):
        """
        Process incoming audio data
        """
        if not self.listening_active:
            return

        try:
            # Convert audio data to audio segment
            # In real implementation, this would process the AudioData message
            # For simulation, we'll simulate speech recognition
            pass

        except Exception as e:
            self.get_logger().error(f'Audio processing error: {str(e)}')

    def process_commands(self):
        """
        Process commands from the queue in a separate thread
        """
        while rclpy.ok() and self.listening_active:
            try:
                if not self.command_queue.empty():
                    command_text = self.command_queue.get_nowait()
                    self.process_voice_command(command_text)
            except:
                pass  # Queue empty, continue
            time.sleep(0.1)

    def process_voice_command(self, command_text):
        """
        Process a transcribed voice command and generate robot action
        """
        self.get_logger().info(f'Processing voice command: "{command_text}"')

        try:
            # Parse the command
            parsed_command = self.parse_command(command_text)

            if parsed_command:
                # Generate robot action based on parsed command
                robot_action = self.generate_robot_action(parsed_command)

                if robot_action:
                    # Publish robot command
                    command_msg = String()
                    command_msg.data = json.dumps(robot_action)
                    self.command_pub.publish(command_msg)

                    # Update status
                    status_msg = String()
                    status_msg.data = f"Executed: {command_text}"
                    self.status_pub.publish(status_msg)

                    # Add to command history
                    self.command_history.append({
                        'command': command_text,
                        'action': robot_action,
                        'timestamp': time.time()
                    })

                    self.get_logger().info(f'Command executed: {robot_action}')
                else:
                    self.get_logger().warn(f'Could not generate action for command: {command_text}')
            else:
                self.get_logger().warn(f'Could not parse command: {command_text}')

        except Exception as e:
            self.get_logger().error(f'Command processing error: {str(e)}')

    def parse_command(self, command_text):
        """
        Parse voice command into structured representation
        """
        command_text = command_text.lower().strip()

        # Identify command type
        command_type = self.identify_command_type(command_text)
        if not command_type:
            return None

        # Extract objects and locations
        entities = self.extract_entities(command_text)

        # Create structured command
        parsed_command = {
            'type': command_type,
            'entities': entities,
            'raw_text': command_text,
            'confidence': 0.9  # Simulated confidence
        }

        return parsed_command

    def identify_command_type(self, command_text):
        """
        Identify the type of command from text
        """
        for cmd_type, keywords in self.command_vocabulary.items():
            for keyword in keywords:
                if keyword in command_text:
                    return cmd_type
        return None

    def extract_entities(self, command_text):
        """
        Extract objects, locations, and parameters from command
        """
        entities = {
            'objects': [],
            'locations': [],
            'parameters': {}
        }

        # Extract known objects
        for obj_name, obj_id in self.known_objects.items():
            if obj_name in command_text:
                entities['objects'].append({
                    'name': obj_name,
                    'id': obj_id,
                    'confidence': 0.9
                })

        # Extract known locations
        for loc_name, loc_id in self.known_locations.items():
            if loc_name in command_text:
                entities['locations'].append({
                    'name': loc_name,
                    'id': loc_id,
                    'confidence': 0.9
                })

        # Extract numerical parameters
        numbers = re.findall(r'\d+\.?\d*', command_text)
        if numbers:
            entities['parameters']['numbers'] = [float(n) for n in numbers]

        # Extract directions
        directions = ['left', 'right', 'forward', 'backward', 'up', 'down']
        for direction in directions:
            if direction in command_text:
                if 'direction' not in entities['parameters']:
                    entities['parameters']['direction'] = []
                entities['parameters']['direction'].append(direction)

        return entities

    def generate_robot_action(self, parsed_command):
        """
        Generate robot action from parsed command
        """
        cmd_type = parsed_command['type']
        entities = parsed_command['entities']

        action = {
            'type': cmd_type,
            'timestamp': time.time()
        }

        if cmd_type == 'move':
            # Generate navigation action
            if entities['locations']:
                target_location = entities['locations'][0]['id']
                action['target'] = target_location
                action['action_type'] = 'navigate'
            else:
                # Default movement if no specific location
                action['action_type'] = 'move_forward'
                action['distance'] = 1.0  # meters

        elif cmd_type == 'grasp':
            # Generate grasping action
            if entities['objects']:
                target_object = entities['objects'][0]['id']
                action['target'] = target_object
                action['action_type'] = 'grasp'
            else:
                action['action_type'] = 'find_and_grasp'

        elif cmd_type == 'place':
            # Generate placement action
            if entities['locations']:
                target_location = entities['locations'][0]['id']
                action['target'] = target_location
                action['action_type'] = 'place'
            else:
                action['action_type'] = 'place_in_front'

        elif cmd_type == 'turn':
            # Generate turning action
            if 'direction' in entities['parameters']:
                direction = entities['parameters']['direction'][0]
                action['direction'] = direction
                action['action_type'] = f'turn_{direction}'
            else:
                action['action_type'] = 'turn_right'
                action['angle'] = 90  # degrees

        elif cmd_type == 'find':
            # Generate search action
            if entities['objects']:
                target_object = entities['objects'][0]['id']
                action['target'] = target_object
                action['action_type'] = 'search_for'
            else:
                action['action_type'] = 'scan_environment'

        else:
            # Default action for unrecognized commands
            action['action_type'] = 'idle'
            action['message'] = 'Command not understood'

        return action

    def simulate_speech_recognition(self, audio_data):
        """
        Simulate speech recognition (in real implementation, this would call STT service)
        """
        # In real implementation, this would process actual audio
        # For simulation, we'll return some example commands
        example_commands = [
            "Move to the kitchen",
            "Grasp the red cup",
            "Place the object on the table",
            "Turn left and go forward",
            "Find the blue box"
        ]

        # Simulate recognition delay
        time.sleep(0.5)

        # Return a random example command
        import random
        return random.choice(example_commands)

def main(args=None):
    rclpy.init(args=args)

    voice_node = VoiceToActionNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        pass
    finally:
        voice_node.stop_listening()
        voice_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Natural Language Understanding with LLMs

```python
import openai
import json
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

class AdvancedNLUProcessor:
    """
    Advanced Natural Language Understanding using Large Language Models.
    Provides sophisticated command parsing and context-aware interpretation.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NLU processor with LLM capabilities.
        """
        if api_key:
            openai.api_key = api_key

        self.command_schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["navigate", "grasp", "place", "search", "turn", "stop", "follow", "wait"]
                },
                "entities": {
                    "type": "object",
                    "properties": {
                        "target_object": {"type": "string"},
                        "target_location": {"type": "string"},
                        "direction": {"type": "string"},
                        "distance": {"type": "number"},
                        "duration": {"type": "number"}
                    }
                },
                "action_sequence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "parameters": {"type": "object"}
                        }
                    }
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["intent", "confidence"]
        }

    def parse_command_with_llm(self, command_text: str, context: Dict = None) -> Dict:
        """
        Parse command using LLM with structured output.
        """
        if context is None:
            context = {}

        # Create a detailed prompt for the LLM
        prompt = self._create_nlu_prompt(command_text, context)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self._get_nlu_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            self._log_error(f"LLM parsing failed: {str(e)}")
            # Fallback to simple parsing
            return self._fallback_parse(command_text)

    def _create_nlu_prompt(self, command_text: str, context: Dict) -> str:
        """
        Create prompt for LLM-based command parsing.
        """
        context_str = json.dumps(context, indent=2) if context else "No context available"

        prompt = f"""
        Parse the following human command for a robot:

        Command: "{command_text}"

        Context: {context_str}

        Please provide a structured interpretation in JSON format with the following structure:
        {{
            "intent": "...",
            "entities": {{
                "target_object": "...",
                "target_location": "...",
                "direction": "...",
                "distance": "...",
                "duration": "..."
            }},
            "action_sequence": [
                {{
                    "action": "...",
                    "parameters": {{...}}
                }}
            ],
            "confidence": 0.0-1.0
        }}

        The intent should be one of: navigate, grasp, place, search, turn, stop, follow, wait
        Only include entities that are explicitly mentioned or clearly implied in the command.
        Provide a confidence score between 0 and 1 based on how certain you are about the interpretation.
        """

        return prompt

    def _get_nlu_system_prompt(self) -> str:
        """
        Get system prompt for NLU task.
        """
        return """
        You are an expert Natural Language Understanding system for robotics.
        Your task is to interpret human commands for robots, extracting intent, entities, and action sequences.
        Be precise and only include information that is explicitly mentioned or clearly implied.
        Consider the context when interpreting ambiguous commands.
        """

    def _fallback_parse(self, command_text: str) -> Dict:
        """
        Fallback simple parsing when LLM is unavailable.
        """
        command_lower = command_text.lower()

        # Simple keyword-based intent detection
        if any(word in command_lower for word in ['go to', 'move to', 'navigate to']):
            intent = 'navigate'
        elif any(word in command_lower for word in ['grasp', 'grab', 'pick up']):
            intent = 'grasp'
        elif any(word in command_lower for word in ['place', 'put', 'set down']):
            intent = 'place'
        elif any(word in command_lower for word in ['find', 'search', 'look for']):
            intent = 'search'
        elif any(word in command_lower for word in ['turn', 'rotate']):
            intent = 'turn'
        elif any(word in command_lower for word in ['stop', 'halt']):
            intent = 'stop'
        else:
            intent = 'navigate'  # Default

        # Extract simple entities
        entities = {}

        # Look for location entities
        location_patterns = [
            r'to the (\w+)',
            r'at the (\w+)',
            r'in the (\w+)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, command_lower)
            if match:
                entities['target_location'] = match.group(1)
                break

        # Look for object entities
        object_patterns = [
            r'(\w+ \w+) cup',  # e.g., "red cup"
            r'(\w+) cup',
            r'(\w+ \w+) box',
            r'(\w+) box',
            r'(\w+ \w+) ball',
            r'(\w+) ball'
        ]

        for pattern in object_patterns:
            match = re.search(pattern, command_lower)
            if match:
                entities['target_object'] = match.group(1)
                break

        return {
            "intent": intent,
            "entities": entities,
            "action_sequence": [{"action": intent, "parameters": {}}],
            "confidence": 0.7  # Lower confidence for fallback
        }

    def _log_error(self, error_msg: str):
        """
        Log errors (in real implementation, this would use proper logging).
        """
        print(f"[NLU Error] {error_msg}")

class VoiceToActionWithLLM:
    """
    Voice-to-Action system enhanced with LLM-based NLU.
    """

    def __init__(self, llm_api_key: Optional[str] = None):
        self.nlu_processor = AdvancedNLUProcessor(llm_api_key)
        self.context_manager = RobotContextManager()

    def process_command(self, command_text: str) -> Dict:
        """
        Process voice command using LLM-enhanced NLU.
        """
        # Get current context
        context = self.context_manager.get_current_context()

        # Parse command with LLM
        parsed_result = self.nlu_processor.parse_command_with_llm(command_text, context)

        # Generate robot action from parsed result
        robot_action = self._generate_action_from_parsed(parsed_result)

        return {
            'original_command': command_text,
            'parsed_result': parsed_result,
            'robot_action': robot_action,
            'success': True
        }

    def _generate_action_from_parsed(self, parsed_result: Dict) -> Dict:
        """
        Generate robot action from LLM-parsed result.
        """
        intent = parsed_result['intent']
        entities = parsed_result.get('entities', {})

        action = {
            'type': intent,
            'parameters': {}
        }

        if intent == 'navigate':
            if 'target_location' in entities:
                action['target'] = entities['target_location']
            else:
                action['target'] = 'default_location'

        elif intent == 'grasp':
            if 'target_object' in entities:
                action['target'] = entities['target_object']
            else:
                action['action'] = 'search_and_grasp'

        elif intent == 'place':
            if 'target_location' in entities:
                action['target'] = entities['target_location']
            else:
                action['target'] = 'default_place_location'

        elif intent == 'turn':
            if 'direction' in entities:
                action['direction'] = entities['direction']
            else:
                action['direction'] = 'right'

        # Add any additional parameters
        for key, value in entities.items():
            if key not in ['target_object', 'target_location', 'direction']:
                action['parameters'][key] = value

        return action

class RobotContextManager:
    """
    Manages contextual information for voice-to-action system.
    """

    def __init__(self):
        self.current_location = "unknown"
        self.visible_objects = []
        self.robot_state = "idle"
        self.last_action = None
        self.conversation_context = []

    def get_current_context(self) -> Dict:
        """
        Get current context for NLU processing.
        """
        return {
            "current_location": self.current_location,
            "visible_objects": self.visible_objects,
            "robot_state": self.robot_state,
            "last_action": self.last_action,
            "conversation_history": self.conversation_context[-5:]  # Last 5 exchanges
        }

    def update_context(self, **kwargs):
        """
        Update context with new information.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_conversation_turn(self, user_utterance: str, robot_response: str):
        """
        Add a conversation turn to context.
        """
        self.conversation_context.append({
            "user": user_utterance,
            "robot": robot_response,
            "timestamp": time.time()
        })

# Example usage
def example_usage():
    """
    Example of using the advanced voice-to-action system.
    """
    # Initialize the system (without API key, it will use fallback)
    vta_system = VoiceToActionWithLLM()

    # Example commands
    commands = [
        "Go to the kitchen and get the red cup",
        "Turn left and then move forward two meters",
        "Find the blue box and bring it to me",
        "Place the object on the dining table"
    ]

    for command in commands:
        print(f"\nProcessing command: '{command}'")
        result = vta_system.process_command(command)

        print(f"Intent: {result['parsed_result']['intent']}")
        print(f"Entities: {result['parsed_result']['entities']}")
        print(f"Robot Action: {result['robot_action']}")
        print(f"Confidence: {result['parsed_result']['confidence']}")

if __name__ == "__main__":
    example_usage()
```

### Real-time Voice Processing with Wake Word Detection

```python
import pyaudio
import numpy as np
import threading
import queue
import time
import collections
from scipy import signal
import webrtcvad  # For voice activity detection

class RealTimeVoiceProcessor:
    """
    Real-time voice processing system with wake word detection
    and continuous command recognition for robotics applications.
    """

    def __init__(self):
        # Audio parameters
        self.rate = 16000  # Sample rate
        self.chunk = 1024  # Buffer size
        self.format = pyaudio.paInt16
        self.channels = 1
        self.frames_per_buffer = 1024

        # Wake word parameters
        self.wake_words = ['robot', 'hey robot', 'please robot']
        self.wake_word_threshold = 0.7

        # Voice activity detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
        self.speech_buffer_size = 1600  # Buffer for speech detection
        self.min_speech_duration = 0.5  # Minimum speech duration in seconds

        # Audio processing
        self.audio_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.is_listening = False
        self.listening_for_command = False

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Speech detection state
        self.speech_buffer = collections.deque(maxlen=int(self.rate * 0.1))  # 100ms buffer
        self.silence_duration = 0
        self.speech_start_time = None
        self.current_audio_segment = []

    def start_listening(self):
        """
        Start the audio listening process.
        """
        self.is_listening = True

        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.audio_callback
        )

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

        print("Started real-time voice processing")

    def stop_listening(self):
        """
        Stop the audio listening process.
        """
        self.is_listening = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.audio:
            self.audio.terminate()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Audio callback for real-time processing.
        """
        if self.is_listening:
            self.audio_queue.put(in_data)

        return (None, pyaudio.paContinue)

    def process_audio(self):
        """
        Process audio data in real-time.
        """
        while self.is_listening:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    self.process_audio_chunk(audio_data)
            except queue.Empty:
                time.sleep(0.01)  # Small delay to prevent busy waiting

    def process_audio_chunk(self, audio_chunk):
        """
        Process a chunk of audio data.
        """
        # Convert to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

        # Voice activity detection
        is_speech = self.is_speech_frame(audio_array)

        if is_speech:
            # Add to current audio segment
            self.current_audio_segment.extend(audio_array.tolist())

            if not self.listening_for_command:
                # Check for wake word in the audio
                if self.detect_wake_word(audio_array):
                    print("Wake word detected!")
                    self.start_command_listening()
        else:
            # Handle silence
            if self.listening_for_command:
                self.silence_duration += len(audio_array) / self.rate

                if self.silence_duration > self.min_speech_duration:
                    # End of speech detected
                    if len(self.current_audio_segment) > 0:
                        # Process the collected audio segment
                        command_audio = np.array(self.current_audio_segment, dtype=np.int16)
                        self.process_command_audio(command_audio)

                    self.end_command_listening()
            else:
                # Reset silence counter when not listening for commands
                self.silence_duration = 0

    def is_speech_frame(self, audio_frame):
        """
        Check if audio frame contains speech using WebRTC VAD.
        """
        # Convert to bytes for WebRTC VAD
        audio_bytes = audio_frame.tobytes()

        # Check if frame size is valid for VAD (must be 10, 20, or 30 ms)
        frame_duration = len(audio_frame) / self.rate
        if frame_duration < 0.01:  # Less than 10ms
            return False

        # WebRTC VAD requires specific frame sizes
        # For 16kHz, valid frame sizes are: 160, 320, 480 samples (10, 20, 30 ms)
        valid_sizes = [160, 320, 480]
        for size in valid_sizes:
            if len(audio_frame) >= size:
                frame_bytes = audio_frame[:size].tobytes()
                try:
                    return self.vad.is_speech(frame_bytes, self.rate)
                except:
                    continue

        return False

    def detect_wake_word(self, audio_segment):
        """
        Detect wake word in audio segment.
        In a real implementation, this would use speech recognition or keyword spotting.
        For simulation, we'll use a simple energy-based detection.
        """
        # Calculate audio energy
        energy = np.mean(np.abs(audio_segment.astype(np.float32)))

        # Simple threshold-based detection (in real implementation, use proper keyword spotting)
        return energy > 1000  # Threshold for demonstration

    def start_command_listening(self):
        """
        Start listening for command after wake word detection.
        """
        self.listening_for_command = True
        self.speech_start_time = time.time()
        self.current_audio_segment = []
        print("Listening for command...")

    def end_command_listening(self):
        """
        End command listening after silence.
        """
        self.listening_for_command = False
        self.silence_duration = 0
        print("Command listening ended")

    def process_command_audio(self, audio_segment):
        """
        Process collected command audio.
        In real implementation, this would send to speech recognition service.
        """
        print(f"Processing command audio of {len(audio_segment) / self.rate:.2f} seconds")

        # Simulate speech recognition
        simulated_command = self.simulate_speech_recognition(audio_segment)

        if simulated_command:
            print(f"Recognized command: '{simulated_command}'")
            self.command_queue.put(simulated_command)

    def simulate_speech_recognition(self, audio_segment):
        """
        Simulate speech recognition (in real implementation, this would call STT service).
        """
        # In real implementation, this would process the audio with STT
        # For simulation, return some example commands based on audio characteristics

        energy = np.mean(np.abs(audio_segment.astype(np.float32)))

        if energy > 2000:
            # High energy - likely a command
            example_commands = [
                "Move forward",
                "Turn left",
                "Grasp the object",
                "Go to the kitchen",
                "Stop moving"
            ]
            import random
            return random.choice(example_commands)
        else:
            return None

    def get_command(self):
        """
        Get the next recognized command.
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

def main():
    """
    Main function to demonstrate real-time voice processing.
    """
    processor = RealTimeVoiceProcessor()

    try:
        processor.start_listening()

        print("Voice processor started. Speak wake word followed by command.")
        print("Press Ctrl+C to stop.")

        while True:
            command = processor.get_command()
            if command:
                print(f"Processing command: {command}")
                # In real implementation, this would send to robot control system

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping voice processor...")
        processor.stop_listening()
        print("Voice processor stopped.")

if __name__ == "__main__":
    main()
```

## Labs and Exercises

### Exercise 1: Wake Word Detection Implementation
Implement a wake word detection system using audio signal processing techniques. Train the system to recognize specific wake words and measure its accuracy and false positive rate in different acoustic environments.

### Exercise 2: Command Vocabulary Expansion
Expand the voice-to-action system's command vocabulary to handle more complex robotic tasks. Implement semantic parsing for compound commands and evaluate the system's ability to handle ambiguous instructions.

### Exercise 3: Noise-Robust Speech Recognition
Integrate noise reduction and acoustic enhancement techniques into the voice-to-action system. Test the system's performance in various noisy environments and compare the effectiveness of different enhancement approaches.

### Exercise 4: Context-Aware Command Interpretation
Implement a context-aware voice-to-action system that uses environmental information to disambiguate commands. Train the system to understand references like "that object" or "over there" based on visual context.

## Summary

This chapter explored Voice-to-Action systems, which enable robots to understand and execute natural language commands. We covered the complete pipeline from speech recognition to action execution, implemented advanced natural language understanding with large language models, and created real-time voice processing systems with wake word detection. The examples demonstrated how to build robust voice interfaces for robotics applications that can handle diverse commands and environmental conditions. Voice-to-action systems represent a crucial component of intuitive human-robot interaction, making robots more accessible and easier to operate for non-expert users.