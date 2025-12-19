import React, { useState, useEffect, useRef } from 'react';
import { useColorMode } from '@docusaurus/theme-common';

// CSS for the chat component that matches Docusaurus theme
const chatStyles = `
  .rag-chat-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
    font-family: var(--ifm-font-family-base);
  }

  .rag-chat-window {
    width: 350px;
    height: 500px;
    border: 1px solid var(--ifm-color-emphasis-300);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    display: flex;
    flex-direction: column;
    background-color: var(--ifm-background-surface-color);
    overflow: hidden;
  }

  .rag-chat-header {
    background-color: var(--ifm-color-primary);
    color: white;
    padding: 12px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .rag-chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .rag-chat-message {
    max-width: 80%;
    padding: 8px 12px;
    border-radius: 18px;
    font-size: 0.9rem;
    line-height: 1.4;
  }

  .rag-chat-message-user {
    align-self: flex-end;
    background-color: var(--ifm-color-primary-lighter);
    border-bottom-right-radius: 4px;
  }

  .rag-chat-message-assistant {
    align-self: flex-start;
    background-color: var(--ifm-color-emphasis-100);
    border-bottom-left-radius: 4px;
  }

  .rag-chat-citations {
    font-size: 0.75rem;
    color: var(--ifm-color-emphasis-700);
    margin-top: 4px;
    padding-top: 4px;
    border-top: 1px dashed var(--ifm-color-emphasis-300);
  }

  .rag-chat-input-area {
    display: flex;
    padding: 12px;
    border-top: 1px solid var(--ifm-color-emphasis-300);
    background-color: var(--ifm-background-surface-color);
  }

  .rag-chat-input {
    flex: 1;
    padding: 8px 12px;
    border: 1px solid var(--ifm-color-emphasis-300);
    border-radius: 18px;
    font-size: 0.9rem;
    resize: none;
    overflow: hidden;
    max-height: 100px;
  }

  .rag-chat-input:focus {
    outline: none;
    border-color: var(--ifm-color-primary);
  }

  .rag-chat-send-button {
    margin-left: 8px;
    padding: 8px 16px;
    background-color: var(--ifm-color-primary);
    color: white;
    border: none;
    border-radius: 18px;
    cursor: pointer;
  }

  .rag-chat-send-button:hover {
    background-color: var(--ifm-color-primary-dark);
  }

  .rag-chat-toggle-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1001;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: var(--ifm-color-primary);
    color: white;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .rag-chat-toggle-button:hover {
    background-color: var(--ifm-color-primary-dark);
  }

  .rag-chat-typing-indicator {
    font-style: italic;
    color: var(--ifm-color-emphasis-600);
    font-size: 0.85rem;
    padding: 8px 12px;
  }
`;

// Function to get selected text
const getSelectedText = () => {
  const selection = window.getSelection();
  return selection.toString().trim();
};

const RagChatComponent = (props) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const { colorMode } = useColorMode();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize chat session
  useEffect(() => {
    const initSession = async () => {
      try {
        const response = await fetch(`${props.apiUrl || 'http://localhost:8000'}/api/chat/new`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ selectedText: getSelectedText() })
        });
        const data = await response.json();
        setSessionId(data.sessionId);
      } catch (error) {
        console.error('Error initializing chat session:', error);
      }
    };

    initSession();
  }, [props.apiUrl]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const adjustTextareaHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 100) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading || !sessionId) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    // Add user message to UI immediately
    const userMsgObj = {
      id: Date.now(),
      role: 'user',
      content: userMessage,
      citations: [],
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMsgObj]);

    try {
      // Get any currently selected text
      const selectedText = getSelectedText();

      const response = await fetch(`${props.apiUrl || 'http://localhost:8000'}/api/chat/${sessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          selectedText: selectedText || undefined
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Add assistant response to UI
      const assistantMsgObj = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        citations: data.citations || [],
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMsgObj]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to UI
      const errorMsgObj = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        citations: [],
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, errorMsgObj]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Render citations for assistant messages
  const renderCitations = (citations) => {
    if (!citations || citations.length === 0) return null;

    return (
      <div className="rag-chat-citations">
        Sources: {citations.map((cit, idx) => (
          <span key={idx}>
            {cit.chapter}
            {cit.section && `:${cit.section}`}
            {cit.page > 0 && `, p.${cit.page}`}
            {idx < citations.length - 1 && ', '}
          </span>
        ))}
      </div>
    );
  };

  return (
    <>
      {/* Add styles to document head */}
      <style>{chatStyles}</style>

      {!isOpen ? (
        <button
          className="rag-chat-toggle-button"
          onClick={toggleChat}
          aria-label="Open chat"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H16.5L14.06 19.44C13.2158 19.8211 12.2807 20.018 11.336 20.018C10.3913 20.018 9.4562 19.8211 8.612 19.44C7.76775 19.0589 7.0133 18.4972 6.40542 17.7928C5.79753 17.0885 5.35163 16.2604 5.1 15.367C4.84837 14.4736 4.8 13.53 4.8 12.6C4.8 11.67 4.84837 10.7264 5.1 9.833C5.35163 8.93959 5.79753 8.11152 6.40542 7.40717C7.0133 6.70283 7.76775 6.14114 8.612 5.76C9.4562 5.37886 10.3913 5.18204 11.336 5.18204C12.2807 5.18204 13.2158 5.37886 14.06 5.76C15.7552 6.56838 17.0931 8.0098 17.843 9.833C18.5929 11.6562 18.6996 13.7536 18.143 15.73" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M18 9L21 6M21 6L18 3M21 6H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      ) : (
        <div className="rag-chat-container">
          <div className="rag-chat-window">
            <div className="rag-chat-header">
              <span>Book Assistant</span>
              <button onClick={toggleChat} style={{ background: 'none', border: 'none', color: 'white', cursor: 'pointer' }}>
                ×
              </button>
            </div>

            <div className="rag-chat-messages">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`rag-chat-message ${
                    msg.role === 'user' ? 'rag-chat-message-user' : 'rag-chat-message-assistant'
                  }`}
                >
                  {msg.content}
                  {msg.role === 'assistant' && renderCitations(msg.citations)}
                </div>
              ))}

              {isLoading && (
                <div className="rag-chat-typing-indicator">
                  Assistant is thinking...
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="rag-chat-input-area">
              <textarea
                ref={textareaRef}
                className="rag-chat-input"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about the book..."
                rows={1}
              />
              <button
                className="rag-chat-send-button"
                onClick={handleSendMessage}
                disabled={isLoading || !inputValue.trim()}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default RagChatComponent;