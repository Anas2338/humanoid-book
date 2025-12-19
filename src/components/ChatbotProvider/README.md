# Chatbot Provider Component

The ChatbotProvider component wraps the Docusaurus documentation content and provides an integrated chatbot interface.

## Features

- **Floating Chat Button**: A fixed-position button that appears on all pages
- **Expandable Interface**: Chat can be expanded or minimized as needed
- **Text Selection Integration**: Automatically detects text selected on the documentation pages
- **Persistent Session**: Maintains conversation context across page navigation
- **Responsive Design**: Works on desktop and mobile devices

## Components

### ChatbotProvider.jsx
Main wrapper component that:
- Renders the floating chat button when chat is minimized
- Provides the full chat interface when expanded
- Manages visibility and expansion state
- Passes the container reference for text selection

### ChatbotProvider.css
Styling for:
- Floating button with attractive gradient
- Expandable/collapsible chat container
- Header with title and action buttons
- Responsive design for different screen sizes

## Integration

The component is integrated into Docusaurus using a theme override at `src/theme/Layout.js`, which wraps all pages with the chatbot functionality.

## API

The component accepts the following props:
- `apiUrl`: The backend API URL (defaults to 'http://localhost:8000')

## Functionality

1. **Text Selection**: The component automatically detects when users select text on the documentation pages
2. **Context Awareness**: When text is selected, it's automatically included in the next query to constrain answers
3. **Session Management**: Conversations persist across page navigation
4. **Source Attribution**: Responses include links to the relevant documentation sections