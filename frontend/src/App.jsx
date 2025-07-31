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

// Agent icons mapping
const AGENT_ICONS = {
  gemini_pro: '🧠',
  gemini_flash: '⚡',
  gemini_vision: '👁️',
  gemini_coder: '💻',
  gemini_researcher: '📚',
  gpt4: '🤖',
  claude: '🎨'
};

// Agent colors
const AGENT_COLORS = {
  gemini_pro: '#4285f4',
  gemini_flash: '#ea4335',
  gemini_vision: '#34a853',
  gemini_coder: '#fbbc04',
  gemini_researcher: '#9c27b0',
  gpt4: '#10a37f',
  claude: '#d4a574'
};

function App() {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState('gemini_pro');
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [taskDescription, setTaskDescription] = useState('');
  const [taskResult, setTaskResult] = useState(null);
  const messagesEndRef = useRef(null);

  // Load agents and system status on mount
  useEffect(() => {
    loadAgents();
    loadSystemStatus();
  }, []);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadAgents = async () => {
    try {
      const response = await api.get('/agents');
      setAgents(response.data);
      if (response.data.length > 0 && !response.data.find(a => a.id === selectedAgent)) {
        setSelectedAgent(response.data[0].id);
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
        agent_type: selectedAgent,
        temperature: 0.7,
        max_tokens: 2048
      });

      const agentMessage = {
        id: Date.now() + 1,
        text: response.data.response,
        sender: 'agent',
        agent: response.data.agent,
        agent_type: response.data.agent_type,
        execution_time: response.data.execution_time,
        timestamp: new Date().toLocaleTimeString()
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

  const executeTask = async (e) => {
    e.preventDefault();
    if (!taskDescription.trim() || isLoading) return;

    setIsLoading(true);
    setTaskResult(null);

    try {
      const response = await api.post('/task', {
        description: taskDescription,
        task_type: 'general',
        priority: 'MEDIUM',
        context: {},
        preferred_agent: selectedAgent
      });

      setTaskResult({
        success: true,
        result: response.data.result,
        agent_used: response.data.agent_used,
        execution_time: response.data.execution_time,
        task_id: response.data.task_id
      });
    } catch (error) {
      console.error('Task execution error:', error);
      setTaskResult({
        success: false,
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setImagePreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!imageFile || isLoading) return;

    setIsLoading(true);
    setTaskResult(null);

    try {
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('text', taskDescription || 'Analyze this image in detail');

      const response = await api.post('/multimodal', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setTaskResult({
        success: true,
        result: response.data.analysis,
        agent_used: response.data.agent,
        filename: response.data.filename,
        image_size: response.data.image_size
      });
    } catch (error) {
      console.error('Image analysis error:', error);
      setTaskResult({
        success: false,
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const getAgentIcon = (agentType) => AGENT_ICONS[agentType] || '🤖';
  const getAgentColor = (agentType) => AGENT_COLORS[agentType] || '#6b7280';

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <h1>🌟 BirLab AI</h1>
            <p>Ultimate coordination of 14+ AI providers</p>
          </div>
          <div className="header-right">
            {systemStatus && (
              <div className="system-status">
                <div className="status-item">
                  <span className="status-label">Agents:</span>
                  <span className="status-value">{systemStatus.active_agents}</span>
                </div>
                <div className="status-item">
                  <span className="status-label">Tasks:</span>
                  <span className="status-value">{systemStatus.completed_tasks}</span>
                </div>
                <div className="status-item">
                  <span className="status-label">Uptime:</span>
                  <span className="status-value">{Math.round(systemStatus.uptime_seconds)}s</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="nav-tabs">
        <button 
          className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          💬 Chat
        </button>
        <button 
          className={`nav-tab ${activeTab === 'tasks' ? 'active' : ''}`}
          onClick={() => setActiveTab('tasks')}
        >
          🎯 Tasks
        </button>
                    <button 
              className={`nav-tab ${activeTab === 'vision' ? 'active' : ''}`}
              onClick={() => setActiveTab('vision')}
            >
              👁️ Vision
            </button>
            <button 
              className={`nav-tab ${activeTab === 'split-view' ? 'active' : ''}`}
              onClick={() => setActiveTab('split-view')}
            >
              🔀 Split View
            </button>
            <button 
              className={`nav-tab ${activeTab === 'agents' ? 'active' : ''}`}
              onClick={() => setActiveTab('agents')}
            >
              🤖 Agents
            </button>
      </nav>

      <main className="main-content">
        {/* Agent Selector */}
        <div className="agent-selector">
          <label htmlFor="agent-select">Select AI Agent:</label>
          <select 
            id="agent-select"
            value={selectedAgent} 
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="agent-select"
          >
            {agents.map(agent => (
              <option key={agent.id} value={agent.id}>
                {getAgentIcon(agent.id)} {agent.name}
              </option>
            ))}
          </select>
        </div>

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <div className="chat-container">
            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="welcome-message">
                  <h3>Welcome to BirLab AI Chat! 🌟</h3>
                  <p>Select an AI agent and start chatting. Each agent has unique capabilities:</p>
                  <ul>
                    <li>🧠 <strong>Gemini Pro</strong>: 2M token context, advanced reasoning</li>
                    <li>⚡ <strong>Gemini Flash</strong>: Lightning-fast responses</li>
                    <li>👁️ <strong>Gemini Vision</strong>: Image understanding</li>
                    <li>💻 <strong>Gemini Coder</strong>: Programming specialist</li>
                    <li>📚 <strong>Gemini Researcher</strong>: Research and analysis</li>
                  </ul>
                </div>
              )}

              {messages.map(message => (
                <div key={message.id} className={`message ${message.sender}`}>
                  <div className="message-header">
                    <span className="message-sender">
                      {message.sender === 'user' ? '👤 You' : 
                       message.sender === 'error' ? '❌ Error' :
                       `${getAgentIcon(message.agent_type)} ${message.agent}`}
                    </span>
                    <span className="message-time">{message.timestamp}</span>
                    {message.execution_time && (
                      <span className="execution-time">
                        ⚡ {message.execution_time.toFixed(2)}s
                      </span>
                    )}
                  </div>
                  <div className="message-content">
                    {message.text}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="message agent loading">
                  <div className="message-header">
                    <span className="message-sender">
                      {getAgentIcon(selectedAgent)} {agents.find(a => a.id === selectedAgent)?.name || 'AI Agent'}
                    </span>
                  </div>
                  <div className="message-content">
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

            <form onSubmit={sendMessage} className="chat-input-form">
              <div className="chat-input-container">
                <input
                  type="text"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  placeholder="Type your message..."
                  className="chat-input"
                  disabled={isLoading}
                />
                <button 
                  type="submit" 
                  className="send-button"
                  disabled={isLoading || !currentMessage.trim()}
                >
                  {isLoading ? '⏳' : '🚀'}
                </button>
                <button 
                  type="button" 
                  onClick={clearChat}
                  className="clear-button"
                  title="Clear chat"
                >
                  🗑️
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Tasks Tab */}
        {activeTab === 'tasks' && (
          <div className="tasks-container">
            <h2>🎯 Complex Task Execution</h2>
            <p>Execute complex tasks using the AI coordinator system</p>

            <form onSubmit={executeTask} className="task-form">
              <div className="form-group">
                <label htmlFor="task-description">Task Description:</label>
                <textarea
                  id="task-description"
                  value={taskDescription}
                  onChange={(e) => setTaskDescription(e.target.value)}
                  placeholder="Describe the task you want to execute..."
                  className="task-textarea"
                  rows={4}
                  disabled={isLoading}
                />
              </div>

              <button 
                type="submit" 
                className="execute-button"
                disabled={isLoading || !taskDescription.trim()}
              >
                {isLoading ? '⏳ Executing...' : '🚀 Execute Task'}
              </button>
            </form>

            {taskResult && (
              <div className={`task-result ${taskResult.success ? 'success' : 'error'}`}>
                <div className="result-header">
                  <h3>
                    {taskResult.success ? '✅ Task Completed' : '❌ Task Failed'}
                  </h3>
                  {taskResult.success && (
                    <div className="result-meta">
                      <span>Agent: {taskResult.agent_used}</span>
                      <span>Time: {taskResult.execution_time?.toFixed(2)}s</span>
                      {taskResult.task_id && <span>ID: {taskResult.task_id.slice(0, 8)}</span>}
                    </div>
                  )}
                </div>
                <div className="result-content">
                  {taskResult.success ? taskResult.result : taskResult.error}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Vision Tab */}
        {activeTab === 'vision' && (
          <div className="vision-container">
            <h2>👁️ Multimodal Vision Analysis</h2>
            <p>Upload an image and ask questions about it using Gemini Vision</p>

            <div className="vision-form">
              <div className="form-group">
                <label htmlFor="image-upload">Upload Image:</label>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="file-input"
                />
              </div>

              {imagePreview && (
                <div className="image-preview">
                  <img src={imagePreview} alt="Preview" className="preview-image" />
                </div>
              )}

              <div className="form-group">
                <label htmlFor="image-question">Question about the image:</label>
                <input
                  id="image-question"
                  type="text"
                  value={taskDescription}
                  onChange={(e) => setTaskDescription(e.target.value)}
                  placeholder="What do you want to know about this image?"
                  className="image-question-input"
                  disabled={isLoading}
                />
              </div>

              <button 
                onClick={analyzeImage}
                className="analyze-button"
                disabled={isLoading || !imageFile}
              >
                {isLoading ? '⏳ Analyzing...' : '🔍 Analyze Image'}
              </button>
            </div>

            {taskResult && (
              <div className={`task-result ${taskResult.success ? 'success' : 'error'}`}>
                <div className="result-header">
                  <h3>
                    {taskResult.success ? '✅ Analysis Complete' : '❌ Analysis Failed'}
                  </h3>
                  {taskResult.success && (
                    <div className="result-meta">
                      <span>Agent: {taskResult.agent_used}</span>
                      {taskResult.filename && <span>File: {taskResult.filename}</span>}
                      {taskResult.image_size && <span>Size: {(taskResult.image_size / 1024).toFixed(1)}KB</span>}
                    </div>
                  )}
                </div>
                <div className="result-content">
                  {taskResult.success ? taskResult.result : taskResult.error}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Split View Tab */}
        {activeTab === 'split-view' && (
          <div className="content-section">
            <SplitView />
          </div>
        )}

        {/* Agents Tab */}
        {activeTab === 'agents' && (
          <div className="agents-container">
            <h2>🤖 Available AI Agents</h2>
            <p>Overview of all available AI agents and their capabilities</p>

            <div className="agents-grid">
              {agents.map(agent => (
                <div 
                  key={agent.id} 
                  className={`agent-card ${selectedAgent === agent.id ? 'selected' : ''}`}
                  onClick={() => setSelectedAgent(agent.id)}
                  style={{ borderColor: getAgentColor(agent.id) }}
                >
                  <div className="agent-header">
                    <div className="agent-icon" style={{ color: getAgentColor(agent.id) }}>
                      {getAgentIcon(agent.id)}
                    </div>
                    <div className="agent-info">
                      <h3>{agent.name}</h3>
                      <p className="agent-type">{agent.type}</p>
                    </div>
                    <div className="agent-status">
                      <span className={`status-badge ${agent.status}`}>
                        {agent.status}
                      </span>
                    </div>
                  </div>

                  <div className="agent-capabilities">
                    <h4>Capabilities:</h4>
                    <div className="capabilities-list">
                      {agent.capabilities.map(cap => (
                        <span key={cap} className="capability-tag">
                          {cap}
                        </span>
                      ))}
                    </div>
                  </div>

                  {agent.model_info && Object.keys(agent.model_info).length > 0 && (
                    <div className="agent-model-info">
                      <h4>Model Info:</h4>
                      <div className="model-info-grid">
                        {agent.model_info.provider && (
                          <div className="info-item">
                            <span>Provider:</span>
                            <span>{agent.model_info.provider}</span>
                          </div>
                        )}
                        {agent.model_info.context_length && (
                          <div className="info-item">
                            <span>Context:</span>
                            <span>{agent.model_info.context_length.toLocaleString()} tokens</span>
                          </div>
                        )}
                        {agent.model_info.version && (
                          <div className="info-item">
                            <span>Version:</span>
                            <span>{agent.model_info.version}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {agents.length === 0 && (
              <div className="no-agents">
                <h3>No agents available</h3>
                <p>Make sure your backend is running and API keys are configured.</p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 