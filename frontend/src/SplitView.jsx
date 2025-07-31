import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const SplitView = () => {
  const [message, setMessage] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [comparisonStats, setComparisonStats] = useState(null);
  const [maxModels, setMaxModels] = useState(4);

  // Load available models on component mount
  useEffect(() => {
    loadAvailableModels();
  }, []);

  const loadAvailableModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models/available`);
      setAvailableModels(response.data.all_models);
      
      // Auto-select first 2 models from different providers if possible
      const models = response.data.all_models;
      if (models.length >= 2) {
        const providers = new Set();
        const autoSelected = [];
        
        for (const model of models) {
          if (autoSelected.length < 2 && !providers.has(model.provider)) {
            autoSelected.push(model.id);
            providers.add(model.provider);
          }
        }
        
        setSelectedModels(autoSelected);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const handleModelToggle = (modelId) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId);
      } else if (prev.length < maxModels) {
        return [...prev, modelId];
      } else {
        // Replace first model if at max capacity
        return [modelId, ...prev.slice(1)];
      }
    });
  };

  const runComparison = async () => {
    if (!message.trim() || selectedModels.length < 1) {
      alert('Please enter a message and select at least 1 model');
      return;
    }

    setIsLoading(true);
    setResults([]);
    setComparisonStats(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/split-view`, {
        message: message.trim(),
        model_ids: selectedModels,
        max_models: maxModels
      });

      setResults(response.data.results);
      setComparisonStats(response.data.comparison_stats);
    } catch (error) {
      console.error('Split view error:', error);
      alert('Error running comparison: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsLoading(false);
    }
  };

  const getModelDisplayName = (agentId) => {
    const model = availableModels.find(m => m.id === agentId);
    return model ? model.name : agentId;
  };

  const getModelProvider = (agentId) => {
    const model = availableModels.find(m => m.id === agentId);
    return model ? model.provider : 'unknown';
  };

  const formatResponseTime = (time) => {
    return `${time}s`;
  };

  // Helper to highlight bests
  const getHighlightClass = (result) => {
    if (!comparisonStats) return '';
    if (comparisonStats.fastest_model === result.agent_id) return 'highlight-fastest';
    if (comparisonStats.longest_response === result.agent_id) return 'highlight-longest';
    if (comparisonStats.highest_quality === result.agent_id) return 'highlight-quality';
    return '';
  };

  return (
    <div className="split-view">
      <div className="split-view-header">
        <h2>🔀 Split View Comparison</h2>
        <p>Compare responses from multiple AI models side-by-side</p>
      </div>

      {/* Input Section */}
      <div className="input-section">
        <div className="message-input-container">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter your prompt to compare across multiple models..."
            className="message-input"
            rows={3}
            disabled={isLoading}
          />
        </div>

        {/* Model Selection */}
        <div className="model-selection">
          <div className="selection-header">
            <h3>Select Models ({selectedModels.length}/{maxModels})</h3>
            <div className="max-models-control">
              <label>Max Models:</label>
              <select 
                value={maxModels} 
                onChange={(e) => setMaxModels(Number(e.target.value))}
                disabled={isLoading}
              >
                <option value={2}>2</option>
                <option value={3}>3</option>
                <option value={4}>4</option>
                <option value={6}>6</option>
              </select>
            </div>
          </div>

          <div className="models-grid">
            {availableModels.map(model => (
              <div 
                key={model.id}
                className={`model-card ${selectedModels.includes(model.id) ? 'selected' : ''}`}
                onClick={() => handleModelToggle(model.id)}
              >
                <div className="model-info">
                  <span className="provider-badge">{model.provider}</span>
                  <span className="model-name">{model.name}</span>
                </div>
                <div className="model-details">
                  <small>Context: {model.context_length}</small>
                </div>
              </div>
            ))}
          </div>
        </div>

        <button 
          onClick={runComparison}
          disabled={isLoading || !message.trim() || selectedModels.length === 0}
          className="compare-button"
        >
          {isLoading ? '🔄 Comparing...' : '🚀 Compare Models'}
        </button>
      </div>

      {/* Comparison Stats */}
      {comparisonStats && (
        <div className="comparison-stats">
          <h3>📊 Comparison Results</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Total Time:</span>
              <span className="stat-value">{comparisonStats.total_comparison_time}s</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Successful:</span>
              <span className="stat-value">{comparisonStats.successful_responses}/{comparisonStats.total_models}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Average Time:</span>
              <span className="stat-value">{comparisonStats.average_response_time}s</span>
            </div>
            {comparisonStats.fastest_model && (
              <div className="stat-item highlight-fastest">
                <span className="stat-label">Fastest:</span>
                <span className="stat-value">{getModelDisplayName(comparisonStats.fastest_model)}</span>
              </div>
            )}
            {comparisonStats.longest_response && (
              <div className="stat-item highlight-longest">
                <span className="stat-label">Longest:</span>
                <span className="stat-value">{getModelDisplayName(comparisonStats.longest_response)}</span>
              </div>
            )}
            {comparisonStats.highest_quality && (
              <div className="stat-item highlight-quality">
                <span className="stat-label">Best Quality:</span>
                <span className="stat-value">{getModelDisplayName(comparisonStats.highest_quality)}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Results Grid */}
      {results.length > 0 && (
        <div className="results-section">
          <h3>🔥 Model Responses</h3>
          <div className="results-grid">
            {results.map((result, index) => (
              <div key={result.agent_id} className={`result-card ${getHighlightClass(result)}`}>
                <div className="result-header">
                  <div className="model-info">
                    <span className="provider-tag">{getModelProvider(result.agent_id)}</span>
                    <span className="model-name">{getModelDisplayName(result.agent_id)}</span>
                  </div>
                  <div className="result-stats">
                    <span className="response-time">⏱️ {formatResponseTime(result.response_time)}</span>
                    <span className="response-length">📝 {result.response_length_words} words / {result.response_length_chars} chars</span>
                    <span className="quality-score">⭐ {Math.round(result.quality_score * 100)}%</span>
                  </div>
                </div>
                <div className="result-content">
                  {result.error ? (
                    <div className="error-message">
                      ❌ Error: {result.error}
                    </div>
                  ) : (
                    <div className="response-text">
                      {result.response}
                    </div>
                  )}
                </div>
                <div className="result-footer">
                  <small>Generated at {new Date(result.timestamp).toLocaleTimeString()}</small>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner">
            <div className="spinner"></div>
            <p>Querying {selectedModels.length} models...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default SplitView; 