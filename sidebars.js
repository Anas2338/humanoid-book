// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    {
      type: 'category',
      label: 'Part I: Introduction & Foundations',
      items: [
        'part-i-introduction-foundations/introduction-to-physical-ai',
        'part-i-introduction-foundations/robotics-fundamentals',
        'part-i-introduction-foundations/ai-robotics-integration',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part II: Module 1 - The Robotic Nervous System',
      items: [
        'part-ii-robotic-nervous-system/ros2-fundamentals',
        'part-ii-robotic-nervous-system/nodes-topics-services',
        'part-ii-robotic-nervous-system/python-bridging',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part III: Module 2 - The Digital Twin (Gazebo & Unity)',
      items: [
        'part-iii-digital-twin/physics-simulation-fundamentals',
        'part-iii-digital-twin/gazebo-environments',
        'part-iii-digital-twin/unity-integration',
        'part-iii-digital-twin/sensor-simulation',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part IV: Module 3 - The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'part-iv-ai-robot-brain/nvidia-isaac-fundamentals',
        'part-iv-ai-robot-brain/isaac-sim-integration',
        'part-iv-ai-robot-brain/isaac-ros-pipelines',
        'part-iv-ai-robot-brain/gpu-acceleration',
        'part-iv-ai-robot-brain/nav2-navigation-with-isaac',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part V: Module 4 - Vision-Language-Action (VLA)',
      items: [
        'part-v-vla/vla-fundamentals',
        'part-v-vla/voice-to-action-systems',
        'part-v-vla/llm-based-cognitive-planning',
        'part-v-vla/vision-guided-manipulation',
        'part-v-vla/multimodal-integration',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part VI: Capstone - The Autonomous Humanoid',
      items: [
        'part-vi-capstone/capstone-overview',
        'part-vi-capstone/system-architecture',
        'part-vi-capstone/integration-workflows',
        'part-vi-capstone/final-project',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Part VII: Appendices',
      items: [
        'part-vii-appendices/development-environment-setup',
        'part-vii-appendices/mathematical-foundations',
        'part-vii-appendices/troubleshooting-debugging',
        'part-vii-appendices/tools-resources',
        'part-vii-appendices/references-further-reading',
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;