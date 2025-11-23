// @ts-nocheck
import { Brain, Database, Pause, Play, Plus, RotateCcw, Settings, Trash2, TrendingUp, Zap } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

// Activation functions with their derivatives
const activations = {
  relu: {
    fn: x => Math.max(0, x),
    derivative: x => x > 0 ? 1 : 0,
    name: 'ReLU'
  },
  sigmoid: {
    fn: x => 1 / (1 + Math.exp(-x)),
    derivative: x => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    name: 'Sigmoid'
  },
  tanh: {
    fn: x => Math.tanh(x),
    derivative: x => 1 - Math.tanh(x) ** 2,
    name: 'Tanh'
  },
  linear: {
    fn: x => x,
    derivative: x => 1,
    name: 'Linear'
  }
};

// Optimizer implementations
class Optimizer {
  constructor(params) {
    this.params = params;
  }
}

class SGD extends Optimizer {
  constructor(params, lr = 0.01, momentum = 0) {
    super(params);
    this.lr = lr;
    this.momentum = momentum;
    this.velocities = params.map(p => p.map(() => 0));
  }

  step(grads) {
    this.params.forEach((param, i) => {
      param.forEach((w, j) => {
        // Momentum helps accelerate SGD in relevant direction and dampens oscillations
        this.velocities[i][j] = this.momentum * this.velocities[i][j] - this.lr * grads[i][j];
        param[j] += this.velocities[i][j];
      });
    });
  }
}

class Adam extends Optimizer {
  constructor(params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    super(params);
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0;
    // Adam maintains two moving averages: first moment (mean) and second moment (variance)
    this.m = params.map(p => p.map(() => 0));
    this.v = params.map(p => p.map(() => 0));
  }

  step(grads) {
    this.t += 1;
    this.params.forEach((param, i) => {
      param.forEach((w, j) => {
        // Update biased first moment estimate
        this.m[i][j] = this.beta1 * this.m[i][j] + (1 - this.beta1) * grads[i][j];
        // Update biased second raw moment estimate
        this.v[i][j] = this.beta2 * this.v[i][j] + (1 - this.beta2) * grads[i][j] ** 2;
        
        // Compute bias-corrected first and second moment estimates
        const mHat = this.m[i][j] / (1 - this.beta1 ** this.t);
        const vHat = this.v[i][j] / (1 - this.beta2 ** this.t);
        
        // Update parameters with adaptive learning rate
        param[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
      });
    });
  }
}

const DeeplexLearningPlatform = () => {
  // Network architecture configuration
  const [layers, setLayers] = useState([
    { size: 2, activation: 'linear', type: 'input' },
    { size: 4, activation: 'relu', type: 'hidden' },
    { size: 1, activation: 'linear', type: 'output' }
  ]);

  // Training data - learners can modify this
  const [dataPoints, setDataPoints] = useState([
    { input: [0.5, 0.3], target: [1.0] },
    { input: [0.2, 0.8], target: [0.5] },
    { input: [0.9, 0.1], target: [0.8] },
    { input: [0.3, 0.6], target: [0.4] }
  ]);

  // Network parameters
  const [weights, setWeights] = useState([]);
  const [biases, setBiases] = useState([]);
  
  // Training configuration
  const [optimizer, setOptimizer] = useState('adam');
  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [batchSize, setBatchSize] = useState(4);
  const [epochs, setEpochs] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [speed, setSpeed] = useState(50);
  
  // Metrics
  const [currentLoss, setCurrentLoss] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [activations, setActivations] = useState([]);
  const [currentBatch, setCurrentBatch] = useState(0);
  
  const optimizerRef = useRef(null);
  const intervalRef = useRef(null);

  // Initialize network weights based on architecture
  const initializeNetwork = () => {
    const newWeights = [];
    const newBiases = [];
    
    // Xavier/He initialization for better convergence
    for (let i = 0; i < layers.length - 1; i++) {
      const inputSize = layers[i].size;
      const outputSize = layers[i + 1].size;
      
      // He initialization: scale by sqrt(2/inputSize) for ReLU, Xavier for others
      const scale = layers[i + 1].activation === 'relu' 
        ? Math.sqrt(2 / inputSize) 
        : Math.sqrt(1 / inputSize);
      
      const layerWeights = [];
      for (let j = 0; j < inputSize * outputSize; j++) {
        layerWeights.push((Math.random() * 2 - 1) * scale);
      }
      newWeights.push(layerWeights);
      
      // Biases initialized to small values
      const layerBiases = Array(outputSize).fill(0).map(() => Math.random() * 0.01);
      newBiases.push(layerBiases);
    }
    
    setWeights(newWeights);
    setBiases(newBiases);
    setEpochs(0);
    setLossHistory([]);
    setCurrentLoss(null);
    
    // Reinitialize optimizer with new parameters
    const allParams = [...newWeights, ...newBiases];
    if (optimizer === 'sgd') {
      optimizerRef.current = new SGD(allParams, learningRate, momentum);
    } else if (optimizer === 'adam') {
      optimizerRef.current = new Adam(allParams, learningRate);
    }
  };

  // Forward pass through the network
  const forwardPass = (input) => {
    if (!input || weights.length === 0 || biases.length === 0 || layers.length < 2) {
      return { activations: [input || []], preActivations: [] };
    }
    
    let layerActivations = [input];
    let layerPreActivations = [];
    
    try {
      for (let i = 0; i < layers.length - 1; i++) {
        const inputSize = layers[i].size;
        const outputSize = layers[i + 1].size;
        const currentInput = layerActivations[i];
        
        if (!currentInput || !weights[i] || !biases[i]) {
          continue;
        }
        
        // Compute weighted sum: output = input @ weights + bias
        const preActivation = [];
        for (let j = 0; j < outputSize; j++) {
          let sum = biases[i][j] || 0;
          for (let k = 0; k < inputSize; k++) {
            sum += (currentInput[k] || 0) * (weights[i][k * outputSize + j] || 0);
          }
          preActivation.push(sum);
        }
        layerPreActivations.push(preActivation);
        
        // Apply activation function with safety check
        const activationKey = layers[i + 1]?.activation || 'linear';
        const activationFn = activations[activationKey];
        
        if (!activationFn || typeof activationFn.fn !== 'function') {
          // Fallback to identity function if activation is invalid
          layerActivations.push(preActivation);
        } else {
          const output = preActivation.map(x => activationFn.fn(x));
          layerActivations.push(output);
        }
      }
    } catch (error) {
      console.error('Error in forward pass:', error);
      return { activations: [input], preActivations: [] };
    }
    
    return { activations: layerActivations, preActivations: layerPreActivations };
  };

  // Backward pass - compute gradients using chain rule
  const backwardPass = (input, target, forward) => {
    const { activations: layerActivations, preActivations: layerPreActivations } = forward;
    
    if (!layerActivations || layerActivations.length === 0) {
      return { weightsGrads: [], biasesGrads: [] };
    }
    
    const weightsGrads = weights.map(w => w.map(() => 0));
    const biasesGrads = biases.map(b => b.map(() => 0));
    
    try {
      // Start with gradient from loss function (MSE)
      const output = layerActivations[layerActivations.length - 1];
      let delta = output.map((o, i) => 2 * (o - (target[i] || 0)));
      
      // Backpropagate through each layer
      for (let i = layers.length - 2; i >= 0; i--) {
        const inputSize = layers[i].size;
        const outputSize = layers[i + 1].size;
        
        // Apply activation derivative with safety check
        const activationKey = layers[i + 1]?.activation || 'linear';
        const activationFn = activations[activationKey];
        
        if (activationFn && typeof activationFn.derivative === 'function' && layerPreActivations[i]) {
          delta = delta.map((d, j) => d * activationFn.derivative(layerPreActivations[i][j]));
        }
        
        // Compute weight gradients: grad_w = input^T @ delta
        if (layerActivations[i]) {
          for (let j = 0; j < inputSize; j++) {
            for (let k = 0; k < outputSize; k++) {
              weightsGrads[i][j * outputSize + k] += (layerActivations[i][j] || 0) * delta[k];
            }
          }
        }
        
        // Bias gradients are just the deltas
        biasesGrads[i] = delta.map((d, j) => (biasesGrads[i][j] || 0) + d);
        
        // Propagate delta to previous layer
        if (i > 0 && weights[i]) {
          const newDelta = Array(inputSize).fill(0);
          for (let j = 0; j < inputSize; j++) {
            for (let k = 0; k < outputSize; k++) {
              newDelta[j] += delta[k] * (weights[i][j * outputSize + k] || 0);
            }
          }
          delta = newDelta;
        }
      }
    } catch (error) {
      console.error('Error in backward pass:', error);
    }
    
    return { weightsGrads, biasesGrads };
  };

  // Training step with mini-batch
  const trainStep = () => {
    if (weights.length === 0 || biases.length === 0 || dataPoints.length === 0) {
      return 0;
    }
    
    let totalLoss = 0;
    const batchWeightsGrads = weights.map(w => w.map(() => 0));
    const batchBiasesGrads = biases.map(b => b.map(() => 0));
    
    // Sample a mini-batch
    const batchIndices = [];
    for (let i = 0; i < Math.min(batchSize, dataPoints.length); i++) {
      batchIndices.push(Math.floor(Math.random() * dataPoints.length));
    }
    setCurrentBatch(batchIndices.length);
    
    // Accumulate gradients over batch
    batchIndices.forEach(idx => {
      const { input, target } = dataPoints[idx];
      const forward = forwardPass(input);
      const output = forward.activations[forward.activations.length - 1];
      
      // Compute loss (MSE)
      const loss = output.reduce((sum, o, i) => sum + (o - target[i]) ** 2, 0) / output.length;
      totalLoss += loss;
      
      // Backpropagation
      const { weightsGrads, biasesGrads } = backwardPass(input, target, forward);
      
      // Accumulate gradients
      weightsGrads.forEach((wg, i) => {
        wg.forEach((g, j) => {
          batchWeightsGrads[i][j] += g;
        });
      });
      biasesGrads.forEach((bg, i) => {
        bg.forEach((g, j) => {
          batchBiasesGrads[i][j] += g;
        });
      });
      
      // Store activations from first sample for visualization
      if (idx === batchIndices[0]) {
        setActivations(forward.activations);
      }
    });
    
    // Average gradients over batch
    const batchSizeActual = batchIndices.length;
    batchWeightsGrads.forEach(wg => wg.forEach((g, i) => wg[i] = g / batchSizeActual));
    batchBiasesGrads.forEach(bg => bg.forEach((g, i) => bg[i] = g / batchSizeActual));
    
    // Update parameters using optimizer
    const allGrads = [...batchWeightsGrads, ...batchBiasesGrads];
    optimizerRef.current.step(allGrads);
    
    // Update state
    const avgLoss = totalLoss / batchSizeActual;
    setCurrentLoss(avgLoss);
    setLossHistory(h => [...h.slice(-199), avgLoss]);
    setEpochs(e => e + 1);
    
    return avgLoss;
  };

  // Toggle training
  const toggleTraining = () => {
    if (weights.length === 0 || biases.length === 0) {
      initializeNetwork();
      // Set a small delay to ensure state updates before starting training
      setTimeout(() => setIsTraining(true), 100);
    } else {
      setIsTraining(!isTraining);
    }
  };

  // Training loop
  useEffect(() => {
    if (isTraining && weights.length > 0) {
      intervalRef.current = setInterval(() => {
        const loss = trainStep();
        if (loss < 0.0001) {
          setIsTraining(false);
        }
      }, 101 - speed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isTraining, speed, weights, biases, optimizer, learningRate, momentum, batchSize, dataPoints]);

  // Add layer
  const addLayer = () => {
    const newLayers = [...layers];
    newLayers.splice(newLayers.length - 1, 0, {
      size: 3,
      activation: 'relu',
      type: 'hidden'
    });
    setLayers(newLayers);
    initializeNetwork();
  };

  // Remove layer
  const removeLayer = (index) => {
    if (layers.length <= 3) return; // Keep at least input, one hidden, and output
    const newLayers = layers.filter((_, i) => i !== index);
    setLayers(newLayers);
    initializeNetwork();
  };

  // Update layer
  const updateLayer = (index, field, value) => {
    const newLayers = [...layers];
    newLayers[index][field] = value;
    setLayers(newLayers);
    if (field === 'size' || field === 'activation') {
      initializeNetwork();
    }
  };

  // Add data point
  const addDataPoint = () => {
    setDataPoints([...dataPoints, {
      input: Array(layers[0].size).fill(0),
      target: Array(layers[layers.length - 1].size).fill(0)
    }]);
  };

  // Update data point
  const updateDataPoint = (index, field, valueIndex, value) => {
    const newDataPoints = [...dataPoints];
    newDataPoints[index][field][valueIndex] = parseFloat(value) || 0;
    setDataPoints(newDataPoints);
  };

  // Remove data point
  const removeDataPoint = (index) => {
    if (dataPoints.length <= 1) return;
    setDataPoints(dataPoints.filter((_, i) => i !== index));
  };

  // Initialize on mount
  useEffect(() => {
    initializeNetwork();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Brain className="w-12 h-12 text-purple-400" />
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Deeplex Learning Platform
                </h1>
                <p className="text-purple-300 text-sm mt-1">
                  Configure, train, and visualize neural networks with complete control
                </p>
              </div>
            </div>
            
            <div className="flex gap-3">
              <button
                onClick={toggleTraining}
                disabled={dataPoints.length === 0}
                className={`${
                  isTraining 
                    ? 'bg-red-600 hover:bg-red-700' 
                    : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
                } disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-6 py-3 font-semibold flex items-center gap-2 transition-all shadow-lg`}
              >
                {isTraining ? <><Pause className="w-5 h-5" /> Pause</> : <><Play className="w-5 h-5" /> Train</>}
              </button>
              <button
                onClick={initializeNetwork}
                className="bg-indigo-600 hover:bg-indigo-700 rounded-lg px-6 py-3 font-semibold flex items-center gap-2 transition-all"
              >
                <RotateCcw className="w-5 h-5" /> Reset
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="space-y-6">
            {/* Architecture Configuration */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-5 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-blue-400" />
                Network Architecture
              </h2>
              
              <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                {layers.map((layer, idx) => (
                  <div key={idx} className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-purple-300">
                        {layer.type === 'input' ? 'Input Layer' : 
                         layer.type === 'output' ? 'Output Layer' : 
                         `Hidden Layer ${idx}`}
                      </span>
                      {layer.type === 'hidden' && (
                        <button
                          onClick={() => removeLayer(idx)}
                          className="text-red-400 hover:text-red-300 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                    
                    <div className="space-y-2">
                      <div>
                        <label className="text-xs text-slate-400 block mb-1">Neurons</label>
                        <input
                          type="number"
                          min="1"
                          max="20"
                          value={layer.size}
                          onChange={(e) => updateLayer(idx, 'size', parseInt(e.target.value) || 1)}
                          disabled={layer.type === 'input' || isTraining}
                          className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-sm disabled:opacity-50"
                        />
                      </div>
                      
                      {layer.type !== 'input' && (
                        <div>
                          <label className="text-xs text-slate-400 block mb-1">Activation</label>
                          <select
                            value={layer.activation}
                            onChange={(e) => updateLayer(idx, 'activation', e.target.value)}
                            disabled={isTraining}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-sm disabled:opacity-50"
                          >
                            {Object.entries(activations).map(([key, act]) => (
                              <option key={key} value={key}>{act.name}</option>
                            ))}
                          </select>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              
              <button
                onClick={addLayer}
                disabled={isTraining}
                className="w-full mt-3 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/50 rounded-lg px-4 py-2 text-sm font-medium flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
              >
                <Plus className="w-4 h-4" /> Add Hidden Layer
              </button>
            </div>

            {/* Optimizer Configuration */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-5 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Optimizer Settings
              </h2>
              
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-slate-300 block mb-2">Optimizer</label>
                  <select
                    value={optimizer}
                    onChange={(e) => {
                      setOptimizer(e.target.value);
                      initializeNetwork();
                    }}
                    disabled={isTraining}
                    className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-2 text-sm disabled:opacity-50"
                  >
                    <option value="sgd">SGD</option>
                    <option value="adam">Adam</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-sm text-slate-300 block mb-2">
                    Learning Rate: {learningRate.toFixed(4)}
                  </label>
                  <input
                    type="range"
                    min="0.0001"
                    max="0.1"
                    step="0.0001"
                    value={learningRate}
                    onChange={(e) => {
                      setLearningRate(parseFloat(e.target.value));
                      initializeNetwork();
                    }}
                    disabled={isTraining}
                    className="w-full"
                  />
                </div>
                
                {optimizer === 'sgd' && (
                  <div>
                    <label className="text-sm text-slate-300 block mb-2">
                      Momentum: {momentum.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="0.99"
                      step="0.01"
                      value={momentum}
                      onChange={(e) => {
                        setMomentum(parseFloat(e.target.value));
                        initializeNetwork();
                      }}
                      disabled={isTraining}
                      className="w-full"
                    />
                  </div>
                )}
                
                <div>
                  <label className="text-sm text-slate-300 block mb-2">
                    Batch Size: {batchSize}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max={Math.max(1, dataPoints.length)}
                    step="1"
                    value={Math.min(batchSize, dataPoints.length)}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    disabled={isTraining}
                    className="w-full"
                  />
                </div>
                
                <div>
                  <label className="text-sm text-slate-300 block mb-2">
                    Training Speed: {speed}%
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="100"
                    step="10"
                    value={speed}
                    onChange={(e) => setSpeed(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* Data Configuration */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-5 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-green-400" />
                Training Data
              </h2>
              
              <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                {dataPoints.map((point, idx) => (
                  <div key={idx} className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-medium text-slate-400">Sample {idx + 1}</span>
                      <button
                        onClick={() => removeDataPoint(idx)}
                        disabled={dataPoints.length <= 1}
                        className="text-red-400 hover:text-red-300 transition-colors disabled:opacity-30"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-xs text-slate-400 block mb-1">Input</label>
                        {point.input.map((val, i) => (
                          <input
                            key={i}
                            type="number"
                            step="0.1"
                            value={val}
                            onChange={(e) => updateDataPoint(idx, 'input', i, e.target.value)}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs mb-1"
                          />
                        ))}
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 block mb-1">Target</label>
                        {point.target.map((val, i) => (
                          <input
                            key={i}
                            type="number"
                            step="0.1"
                            value={val}
                            onChange={(e) => updateDataPoint(idx, 'target', i, e.target.value)}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs mb-1"
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <button
                onClick={addDataPoint}
                className="w-full mt-3 bg-green-600/20 hover:bg-green-600/30 border border-green-500/50 rounded-lg px-4 py-2 text-sm font-medium flex items-center justify-center gap-2 transition-colors"
              >
                <Plus className="w-4 h-4" /> Add Data Point
              </button>
            </div>
          </div>

          {/* Visualization Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Metrics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 backdrop-blur-lg rounded-xl p-5 border border-purple-500/30">
                <div className="text-sm text-purple-300 mb-1">Epoch</div>
                <div className="text-3xl font-bold">{epochs}</div>
              </div>
              
              <div className="bg-gradient-to-br from-red-600/20 to-red-800/20 backdrop-blur-lg rounded-xl p-5 border border-red-500/30">
                <div className="text-sm text-red-300 mb-1">Loss</div>
                <div className="text-3xl font-bold">
                  {currentLoss !== null ? currentLoss.toFixed(6) : '—'}
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 backdrop-blur-lg rounded-xl p-5 border border-blue-500/30">
                <div className="text-sm text-blue-300 mb-1">Batch Size</div>
                <div className="text-3xl font-bold">{currentBatch}</div>
              </div>
            </div>

            {/* Network Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                Network Visualization
              </h2>
              
              {activations.length > 0 ? (
                <div className="flex justify-around items-center min-h-[300px]">
                  {layers.map((layer, layerIdx) => (
                    <div key={layerIdx} className="flex flex-col items-center gap-3">
                      <div className="text-xs text-slate-400 mb-2">
                        {layer.type === 'input' ? 'Input' : 
                         layer.type === 'output' ? 'Output' : 
                         `Hidden ${layerIdx}`}
                      </div>
                      {activations[layerIdx]?.map((activation, neuronIdx) => (
                        <div
                          key={neuronIdx}
                          className="w-12 h-12 rounded-full flex items-center justify-center text-xs font-mono border-2 border-white/30 shadow-lg transition-all"
                          style={{
                            backgroundColor: `rgba(${activation > 0 ? '34, 197, 94' : '239, 68, 68'}, ${0.3 + Math.min(Math.abs(activation), 1) * 0.7})`
                          }}
                        >
                          {activation.toFixed(2)}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="min-h-[300px] flex items-center justify-center text-slate-400">
                  Start training to see network activations
                </div>
              )}
            </div>

            {/* Predictions vs Targets Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                Predictions vs Targets
              </h2>
              
              {weights.length > 0 && biases.length > 0 && dataPoints.length > 0 ? (
                <div className="space-y-3">
                  {dataPoints.map((point, idx) => {
                    const forward = forwardPass(point.input);
                    const prediction = forward.activations[forward.activations.length - 1] || [0];
                    const target = point.target;
                    const error = Math.abs((prediction[0] || 0) - (target[0] || 0));
                    const errorPercent = Math.min(error * 100, 100);
                    
                    return (
                      <div key={idx} className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-slate-400">Sample {idx + 1}</span>
                          <span className="text-xs font-mono text-slate-300">
                            Error: {error.toFixed(4)}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-3 mb-2">
                          <div>
                            <div className="text-xs text-blue-300 mb-1">Input</div>
                            <div className="text-sm font-mono text-slate-200">
                              [{point.input.map(v => v.toFixed(2)).join(', ')}]
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-purple-300 mb-1">Target</div>
                            <div className="text-sm font-mono text-slate-200">
                              [{target.map(v => v.toFixed(2)).join(', ')}]
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <div className="text-xs text-green-300 mb-1">Prediction</div>
                          <div className="text-sm font-mono text-slate-200 mb-2">
                            [{prediction.map(v => v.toFixed(4)).join(', ')}]
                          </div>
                        </div>
                        
                        {/* Visual error bar */}
                        <div className="relative h-2 bg-slate-900 rounded-full overflow-hidden">
                          <div 
                            className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500 to-red-500 transition-all duration-300"
                            style={{ width: `${errorPercent}%` }}
                          />
                        </div>
                        <div className="text-xs text-slate-400 mt-1">
                          {errorPercent < 1 ? 'Excellent match!' : 
                           errorPercent < 5 ? 'Good prediction' : 
                           errorPercent < 20 ? 'Learning in progress' : 
                           'Needs more training'}
                        </div>
                      </div>
                    );
                  })}
                  
                  <div className="text-xs text-slate-400 mt-4 leading-relaxed">
                    This view shows you exactly how well the network is predicting each training example. Watch the error bars shrink as the network learns. The color gradient from green to red visually represents the prediction accuracy, with green indicating the prediction is very close to the target and red showing larger errors that need correction.
                  </div>
                </div>
              ) : (
                <div className="min-h-[200px] flex items-center justify-center text-slate-400">
                  Configure network and add data to see predictions
                </div>
              )}
            </div>

            {/* Weight Distribution Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Weight Distributions
              </h2>
              
              {weights.length > 0 ? (
                <div className="space-y-4">
                  {weights.map((layerWeights, idx) => {
                    const maxWeight = Math.max(...layerWeights.map(Math.abs), 0.01);
                    const avgWeight = layerWeights.reduce((sum, w) => sum + Math.abs(w), 0) / layerWeights.length;
                    
                    return (
                      <div key={idx} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700">
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-sm font-medium text-slate-300">
                            Layer {idx + 1} → {idx + 2}
                          </span>
                          <span className="text-xs font-mono text-slate-400">
                            Avg: {avgWeight.toFixed(4)}
                          </span>
                        </div>
                        
                        {/* Weight histogram */}
                        <div className="flex items-end gap-0.5 h-20 mb-2">
                          {layerWeights.slice(0, 50).map((weight, wIdx) => {
                            const height = (Math.abs(weight) / maxWeight) * 100;
                            const isPositive = weight > 0;
                            
                            return (
                              <div
                                key={wIdx}
                                className="flex-1 rounded-t transition-all duration-300"
                                style={{
                                  height: `${height}%`,
                                  backgroundColor: isPositive 
                                    ? `rgba(34, 197, 94, ${0.4 + height / 200})` 
                                    : `rgba(239, 68, 68, ${0.4 + height / 200})`
                                }}
                              />
                            );
                          })}
                        </div>
                        
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>Negative weights</span>
                          <span>Positive weights</span>
                        </div>
                      </div>
                    );
                  })}
                  
                  <div className="text-xs text-slate-400 leading-relaxed">
                    These histograms show the distribution of weights in each layer. During training, you will see these distributions evolve as the optimizer adjusts the parameters. Healthy training typically shows weights spreading out from their initial values as the network learns meaningful patterns. If weights become too large, this might indicate the learning rate is too high, while weights stuck near zero suggest the network is not learning effectively.
                  </div>
                </div>
              ) : (
                <div className="min-h-[200px] flex items-center justify-center text-slate-400">
                  Initialize network to see weight distributions
                </div>
              )}
            </div>

            {/* Loss History */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                Training Progress
              </h2>
              
              {lossHistory.length > 0 ? (
                <div className="h-48 bg-slate-900/50 rounded-lg p-4 flex items-end gap-0.5">
                  {lossHistory.map((loss, i) => {
                    const maxLoss = Math.max(...lossHistory, 0.01);
                    const height = (loss / maxLoss) * 100;
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-gradient-to-t from-red-500 via-orange-500 to-yellow-500 rounded-t transition-all"
                        style={{
                          height: `${height}%`,
                          opacity: 0.3 + (i / lossHistory.length) * 0.7
                        }}
                      />
                    );
                  })}
                </div>
              ) : (
                <div className="h-48 bg-slate-900/50 rounded-lg p-4 flex items-center justify-center text-slate-400">
                  Loss history will appear here during training
                </div>
              )}
              
              <div className="mt-4 text-xs text-slate-400">
                This chart shows how the loss decreases as the network learns. A downward trend indicates successful learning, where the optimizer is finding better parameter values that minimize the prediction error on your training data.
              </div>
            </div>

            {/* Educational Info */}
            <div className="bg-gradient-to-br from-indigo-600/10 to-purple-600/10 backdrop-blur-lg rounded-xl p-6 border border-indigo-500/20">
              <h3 className="text-lg font-semibold mb-3 text-indigo-300">Understanding Your Configuration</h3>
              <div className="text-sm text-slate-300 space-y-2 leading-relaxed">
                <p>
                  You have configured a neural network with {layers.length} layers, using the {optimizer === 'adam' ? 'Adam' : 'SGD'} optimizer. {optimizer === 'adam' ? 'Adam adapts the learning rate for each parameter based on estimates of first and second moments of the gradients, which often leads to faster convergence.' : momentum > 0 ? `With momentum set to ${momentum.toFixed(2)}, the optimizer remembers past gradients to accelerate training in consistent directions and dampen oscillations.` : 'Without momentum, each update is based solely on the current gradient.'}
                </p>
                <p>
                  Your batch size of {batchSize} means the network processes {batchSize} {batchSize === 1 ? 'sample' : 'samples'} before updating its parameters. Smaller batches provide noisier gradient estimates but update more frequently, while larger batches give more accurate gradients but update less often. The learning rate of {learningRate.toFixed(4)} controls how much the parameters change with each update. Too high and training becomes unstable, too low and convergence takes forever.
                </p>
                <p>
                  Each layer applies an activation function after computing weighted sums. {layers.filter(l => l.activation === 'relu').length > 0 && 'ReLU (Rectified Linear Unit) outputs the input if positive, otherwise zero, providing non-linearity while being computationally efficient.'} {layers.filter(l => l.activation === 'sigmoid').length > 0 && 'Sigmoid squashes values between 0 and 1, useful for binary classification.'} {layers.filter(l => l.activation === 'tanh').length > 0 && 'Tanh squashes values between -1 and 1, often converging faster than sigmoid.'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeeplexLearningPlatform;