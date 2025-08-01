import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import SplitView from './SplitView';
import './App.css';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

function App() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState('birlab_ollama_llama3_8b');
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [taskDescription, setTaskDescription] = useState('');
  const [taskResult, setTaskResult] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadAgents();
    loadSystemStatus();
  }, []);

  const loadAgents = async () => {
    try {
      const response = await api.get('/agents');
      setAgents(response.data);
      if (response.data.length > 0 && !response.data.find(a => a.agent_id === selectedAgent)) {
        // Prefer Ollama models since they're working, otherwise use first available
        const preferredAgent = response.data.find(a => a.provider === 'ollama') || response.data[0];
        setSelectedAgent(preferredAgent.agent_id);
      }
    } catch (error) {
      console.error('Failed to load agents:', error);
    }
  };

  const loadSystemStatus = async () => {
    try {
      const response = await api.get('/system/status');
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to load system status:', error);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: currentMessage,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      const response = await api.post('/chat', {
        message: currentMessage,
        agent_id: selectedAgent,
        temperature: 0.7,
        max_tokens: 2048
      });

      const agentMessage = {
        id: Date.now() + 1,
        text: response.data.response,
        sender: 'agent',
        agent: response.data.agent_used,
        agent_type: response.data.agent_used,
        timestamp: new Date().toLocaleTimeString(),
        // Response Analysis Data
        response_time: response.data.response_time || 0,
        response_length_words: response.data.response_length_words || 0,
        response_length_chars: response.data.response_length_chars || 0,
        quality_score: response.data.quality_score || 0,
        readability_score: response.data.readability_score || 0,
        sentiment_score: response.data.sentiment_score || 0,
        analysis: response.data.analysis || {},
        model_info: response.data.model_info || {}
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = {
        id: Date.now() + 1,
        text: `Error: ${error.response?.data?.detail || error.message}`,
        sender: 'error',
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setActiveTab('chat');
  };

  const getAgentIcon = (agentId) => {
    if (!agentId) return '🤖';
    if (agentId.includes('gemini')) return '🧠';
    if (agentId.includes('ollama')) return '🦙';
    if (agentId.includes('cohere')) return '⚔️';
    return '🤖';
  };

  return (
    <div className="app">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={startNewChat}>
            <span className="icon">💬</span>
            <span className="text">New chat</span>
          </button>
          <button 
            className="collapse-btn"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          >
            <span className="icon">{sidebarCollapsed ? '→' : '←'}</span>
          </button>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section">
            <button 
              className={`nav-item ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              <span className="icon">💬</span>
              <span className="text">Chat</span>
            </button>
          </div>

          <div className="nav-section">
            <button 
              className={`nav-item ${activeTab === 'prompts' ? 'active' : ''}`}
              onClick={() => setActiveTab('prompts')}
            >
              <span className="icon">☀️</span>
              <span className="text">Prompts</span>
            </button>
            <button 
              className={`nav-item ${activeTab === 'assistants' ? 'active' : ''}`}
              onClick={() => setActiveTab('assistants')}
            >
              <span className="icon">🤖</span>
              <span className="text">Assistants</span>
            </button>
            <button 
              className={`nav-item ${activeTab === 'files' ? 'active' : ''}`}
              onClick={() => setActiveTab('files')}
            >
              <span className="icon">📁</span>
              <span className="text">Files</span>
            </button>
            <button 
              className={`nav-item ${activeTab === 'plugins' ? 'active' : ''}`}
              onClick={() => setActiveTab('plugins')}
            >
              <span className="icon">🧩</span>
              <span className="text">Plugins</span>
            </button>
            <button 
              className={`nav-item ${activeTab === 'models' ? 'active' : ''}`}
              onClick={() => setActiveTab('models')}
            >
              <span className="icon">✨</span>
              <span className="text">Models</span>
            </button>
            <button 
              className={`nav-item ${activeTab === 'split-view' ? 'active' : ''}`}
              onClick={() => setActiveTab('split-view')}
            >
              <span className="icon">⊞</span>
              <span className="text">Split view</span>
            </button>
          </div>
        </nav>

        <div className="sidebar-search">
          <div className="search-container">
            <span className="search-icon">🔍</span>
            <input 
              type="text" 
              placeholder="Search chats" 
              className="search-input"
            />
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="system-status">
            {systemStatus && (
              <div className="status-info">
                <div className="status-item">
                  <span className="label">Agents:</span>
                  <span className="value">{systemStatus.total_agents}</span>
                </div>
                <div className="status-item">
                  <span className="label">Providers:</span>
                  <span className="value">{systemStatus.active_providers}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {activeTab === 'chat' && (
          <div className="chat-container">
            {messages.length === 0 ? (
              <div className="welcome-screen">
                <div className="logo-container">
                  <div className="logo">
                    <span className="logo-text">birlab</span>
                    <span className="logo-suffix">ai</span>
                  </div>
                </div>
                <h1 className="welcome-title">Hello, User!</h1>
                <p className="welcome-subtitle">How can I help you today?</p>
                
                <div className="agent-selector-welcome">
                  <label>Choose your AI:</label>
                  <select 
                    value={selectedAgent} 
                    onChange={(e) => setSelectedAgent(e.target.value)}
                    className="agent-select"
                  >
                    {agents.map(agent => (
                      <option key={agent.agent_id} value={agent.agent_id}>
                        {agent.name.replace(/[🧠⚡🌟👁️🚀🌴💬💻🔧🔍🎨🎵⚔️💡🌙🔄➕🔗🦙🦅🤖🔥🐦🐋🧙🐬🔍🤿🎩⚖️]/g, '').trim()}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            ) : (
              <div className="messages-container">
                {messages.map(message => (
                  <div key={message.id} className={`message-wrapper ${message.sender}`}>
                    <div className="message-content">
                      <div className="message-header">
                        <span className="message-sender">
                          {message.sender === 'user' ? 'You' : 'BirLab AI'}
                        </span>
                        {message.response_time && (
                          <span className="message-time">
                            {message.response_time.toFixed(2)}s
                          </span>
                        )}
                      </div>
                      <div className="message-text">
                        {message.text}
                      </div>
                      
                      {/* Response Analysis */}
                      {message.sender === 'agent' && message.analysis && (
                        <div className="response-analysis">
                          <div className="analysis-toggle">
                            <button 
                              className="toggle-btn"
                              onClick={(e) => {
                                const details = e.target.closest('.response-analysis').querySelector('.analysis-details');
                                details.style.display = details.style.display === 'none' ? 'block' : 'none';
                              }}
                            >
                              📊 Analysis
                            </button>
                          </div>
                          
                          <div className="analysis-details" style={{display: 'none'}}>
                            <div className="quick-metrics">
                              <div className="metric">
                                <span className="metric-label">Quality</span>
                                <span className="metric-value">{Math.round((message.quality_score || 0) * 100)}%</span>
                              </div>
                              <div className="metric">
                                <span className="metric-label">Speed</span>
                                <span className="metric-value">{message.response_time?.toFixed(2)}s</span>
                              </div>
                              <div className="metric">
                                <span className="metric-label">Words</span>
                                <span className="metric-value">{message.response_length_words}</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="message-wrapper agent">
                    <div className="message-content">
                      <div className="message-header">
                        <span className="message-sender">BirLab AI</span>
                      </div>
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}

            {/* Input Area */}
            <div className="input-area">
              <form onSubmit={sendMessage} className="message-form">
                <div className="input-container">
                  <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    placeholder="Ask anything..."
                    className="message-input"
                    disabled={isLoading}
                  />
                  <button 
                    type="submit" 
                    className="send-button"
                    disabled={isLoading || !currentMessage.trim()}
                  >
                    <span className="send-icon">↑</span>
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div className="models-container">
            <h2>Available AI Models</h2>
            <div className="models-grid">
              {agents.map(agent => (
                <div key={agent.agent_id} className="model-card">
                  <div className="model-header">
                    <h3>{agent.name.replace(/[🧠⚡🌟👁️🚀🌴💬💻🔧🔍🎨🎵⚔️💡🌙🔄➕🔗🦙🦅🤖🔥🐦🐋🧙🐬🔍🤿🎩⚖️]/g, '').trim()}</h3>
                    <span className="provider">{agent.provider}</span>
                  </div>
                  <div className="model-info">
                    <p>Context: {agent.context_length?.toLocaleString() || 'N/A'} tokens</p>
                    <p>Capabilities: {agent.capabilities?.join(', ') || 'General'}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'prompts' && (
          <div className="feature-container">
            <div className="feature-content">
              <h2>Prompts</h2>
              <p>Create and manage your custom prompts for better AI interactions.</p>
              <div className="coming-soon">
                <span className="icon">🚧</span>
                <h3>Coming Soon</h3>
                <p>This feature is currently under development. You'll be able to:</p>
                <ul>
                  <li>Save frequently used prompts</li>
                  <li>Create prompt templates</li>
                  <li>Share prompts with your team</li>
                  <li>Browse community prompts</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'assistants' && (
          <div className="feature-container">
            <div className="feature-content">
              <h2>Assistants</h2>
              <p>Create custom AI assistants with specialized knowledge and capabilities.</p>
              <div className="coming-soon">
                <span className="icon">🚧</span>
                <h3>Coming Soon</h3>
                <p>This feature is currently under development. You'll be able to:</p>
                <ul>
                  <li>Create custom AI assistants</li>
                  <li>Upload knowledge bases</li>
                  <li>Define assistant personalities</li>
                  <li>Share assistants with others</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'files' && (
          <div className="feature-container">
            <div className="feature-content">
              <h2>Files</h2>
              <p>Upload and manage files for AI analysis and processing.</p>
              <div className="coming-soon">
                <span className="icon">🚧</span>
                <h3>Coming Soon</h3>
                <p>This feature is currently under development. You'll be able to:</p>
                <ul>
                  <li>Upload documents, images, and data files</li>
                  <li>Analyze files with AI models</li>
                  <li>Extract insights from documents</li>
                  <li>Process multiple files at once</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'plugins' && (
          <div className="feature-container">
            <div className="feature-content">
              <h2>Plugins</h2>
              <p>Extend BirLab AI functionality with powerful plugins and integrations.</p>
              <div className="coming-soon">
                <span className="icon">🚧</span>
                <h3>Coming Soon</h3>
                <p>This feature is currently under development. You'll be able to:</p>
                <ul>
                  <li>Install third-party plugins</li>
                  <li>Connect to external APIs</li>
                  <li>Automate workflows</li>
                  <li>Create custom integrations</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'split-view' && (
          <div className="split-view-container">
            <SplitView />
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 