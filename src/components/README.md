# Docusaurus Components

This directory contains custom React components for the Physical AI & Humanoid Robotics documentation site.

## Available Components

### ChatbotProvider
A wrapper component that provides an integrated RAG chatbot interface across all documentation pages. Features include:

- Floating chat interface accessible from any page
- Automatic text selection detection
- Persistent conversation sessions
- Source attribution for responses
- Responsive design for all device sizes

## Integration

These components are designed to work with Docusaurus' swizzling and theme extension capabilities. The ChatbotProvider is integrated via the theme override at `src/theme/Layout.js`.

## Usage

The components are automatically included when the site is built. No additional configuration is required to use them on pages.