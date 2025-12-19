import React, { useState, useRef, useEffect } from 'react';
import RagChatComponent from '../../../rag-chatbot-frontend/docusaurus-components/RagChatComponent';
import './ChatbotProvider.css';

const ChatbotProvider = ({ children, apiUrl }) => {
  const containerRef = useRef(null);

  return (
    <div ref={containerRef} className="chatbot-provider">
      {children}

      {/* Directly render the RagChatComponent which has its own toggle functionality */}
      <RagChatComponent apiUrl={apiUrl} containerRef={containerRef} />
    </div>
  );
};

export default ChatbotProvider;