import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotProvider from '../components/ChatbotProvider/ChatbotProvider';

// Get API URL from environment or use default
const API_URL = typeof window !== 'undefined'
  ? window.ENV?.REACT_APP_API_URL || 'http://localhost:8000'
  : 'http://localhost:8000';

export default function Layout(props) {
  return (
    <OriginalLayout {...props}>
      <ChatbotProvider apiUrl={API_URL}>
        {props.children}
      </ChatbotProvider>
    </OriginalLayout>
  );
}