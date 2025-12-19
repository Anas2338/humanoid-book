import React from 'react';
import ReactDOM from 'react-dom/client';
import RagChatComponent from '@rag-chat/RagChatComponent';

// Function to inject the RAG chat component into the page
function injectRagChat() {
  // Wait for the DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeRagChat);
  } else {
    initializeRagChat();
  }
}

// Initialize the RAG chat component
function initializeRagChat() {
  // Find or create the root element for the chat component
  let chatRoot = document.getElementById('rag-chat-root');

  if (!chatRoot) {
    // Create the root element if it doesn't exist
    chatRoot = document.createElement('div');
    chatRoot.id = 'rag-chat-root';
    document.body.appendChild(chatRoot);
  }

  // Render the RAG chat component
  const root = ReactDOM.createRoot(chatRoot);
  root.render(<RagChatComponent />);
}

// Initialize when the module is loaded
injectRagChat();

// Export for Docusaurus to use
export default {};