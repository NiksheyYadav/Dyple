// @ts-nocheck
import { Brain, Database, Pause, Play, Plus, RotateCcw, Settings, Sparkles, Target, Trash2, TrendingUp, Zap } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';

// Gemini API configuration
const GEMINI_API_KEY = 'AIzaSyDzLsQkUI3xB__KEZws1I0PWmuMyyJqERg';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent';

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

class RMSprop extends Optimizer {
  constructor(params, lr = 0.001, decay = 0.9, epsilon = 1e-8) {
    super(params);
    this.lr = lr;
    this.decay = decay;
    this.epsilon = epsilon;
    this.cache = params.map(p => p.map(() => 0));
  }

  step(grads) {
    this.params.forEach((param, i) => {
      param.forEach((w, j) => {
        this.cache[i][j] = this.decay * this.cache[i][j] + (1 - this.decay) * grads[i][j] ** 2;
        param[j] -= this.lr * grads[i][j] / (Math.sqrt(this.cache[i][j]) + this.epsilon);
      });
    });
  }
}

class Adagrad extends Optimizer {
  constructor(params, lr = 0.01, epsilon = 1e-8) {
    super(params);
    this.lr = lr;
    this.epsilon = epsilon;
    this.cache = params.map(p => p.map(() => 0));
  }

  step(grads) {
    this.params.forEach((param, i) => {
      param.forEach((w, j) => {
        this.cache[i][j] += grads[i][j] ** 2;
        param[j] -= this.lr * grads[i][j] / (Math.sqrt(this.cache[i][j]) + this.epsilon);
      });
    });
  }
}

class AdamW extends Optimizer {
  constructor(params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0.01) {
    super(params);
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.weightDecay = weightDecay;
    this.t = 0;
    this.m = params.map(p => p.map(() => 0));
    this.v = params.map(p => p.map(() => 0));
  }

  step(grads) {
    this.t += 1;
    this.params.forEach((param, i) => {
      param.forEach((w, j) => {
        // Weight decay
        param[j] -= this.lr * this.weightDecay * param[j];
        
        this.m[i][j] = this.beta1 * this.m[i][j] + (1 - this.beta1) * grads[i][j];
        this.v[i][j] = this.beta2 * this.v[i][j] + (1 - this.beta2) * grads[i][j] ** 2;
        
        const mHat = this.m[i][j] / (1 - this.beta1 ** this.t);
        const vHat = this.v[i][j] / (1 - this.beta2 ** this.t);
        
        param[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.epsilon);
      });
    });
  }
}

class Nadam extends Optimizer {
  constructor(params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    super(params);
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0;
    this.m = params.map(p => p.map(() => 0));
    this.v = params.map(p => p.map(() => 0));
  }

  step(grads) {
    this.t += 1;
    this.params.forEach((param, i) => {
      param.forEach((w, j) => {
        this.m[i][j] = this.beta1 * this.m[i][j] + (1 - this.beta1) * grads[i][j];
        this.v[i][j] = this.beta2 * this.v[i][j] + (1 - this.beta2) * grads[i][j] ** 2;
        
        const mHat = this.m[i][j] / (1 - this.beta1 ** this.t);
        const vHat = this.v[i][j] / (1 - this.beta2 ** this.t);
        
        // Nesterov momentum
        const mNesterov = this.beta1 * mHat + (1 - this.beta1) * grads[i][j] / (1 - this.beta1 ** this.t);
        
        param[j] -= this.lr * mNesterov / (Math.sqrt(vHat) + this.epsilon);
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
  
  // Dataset loading
  const [selectedDataset, setSelectedDataset] = useState('');
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [availableDatasets, setAvailableDatasets] = useState([
    { id: 'regression_small', name: 'Regression (Small)', description: '500 samples, 2 inputs, 1 output' },
    { id: 'regression_large', name: 'Regression (Large)', description: '2000 samples, 3 inputs, 1 output' },
    { id: 'classification_binary', name: 'Binary Classification', description: '1000 samples, 2 inputs, 2 classes' },
    { id: 'classification_3class', name: '3-Class Classification', description: '1500 samples, 2 inputs, 3 classes' },
    { id: 'xor_problem', name: 'XOR Problem', description: '400 samples, classic XOR' },
    { id: 'sin_wave', name: 'Sin Wave Prediction', description: '800 samples, sine wave pattern' },
    { id: 'polynomial', name: 'Polynomial Regression', description: '700 samples, degree 3 polynomial' },
    { id: 'multi_output', name: 'Multi-Output', description: '1000 samples, 4 inputs, 2 outputs' }
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
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [showDatasetPreview, setShowDatasetPreview] = useState(false);
  const [samplesProcessed, setSamplesProcessed] = useState(0);
  const [bestLoss, setBestLoss] = useState(Infinity);
  const [trainLoss, setTrainLoss] = useState(null);
  const [valLoss, setValLoss] = useState(null);
  const [trainAccuracy, setTrainAccuracy] = useState(0);
  const [valAccuracy, setValAccuracy] = useState(0);
  const [confusionMatrix, setConfusionMatrix] = useState([]);
  const [targetEpochs, setTargetEpochs] = useState(100);
  const [suggestedArchitecture, setSuggestedArchitecture] = useState(null);
  const [aiSuggestionLoading, setAiSuggestionLoading] = useState(false);
  const [aiSuggestion, setAiSuggestion] = useState(null);
  
  const optimizerRef = useRef(null);
  const intervalRef = useRef(null);
  const logsEndRef = useRef(null);

  // Add training log
  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setTrainingLogs(prev => [...prev.slice(-99), { timestamp, message, type }]);
  };

  // Scroll to bottom of logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [trainingLogs]);

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
    setSamplesProcessed(0);
    setBestLoss(Infinity);
    setTrainLoss(null);
    setValLoss(null);
    setTrainAccuracy(0);
    setValAccuracy(0);
    setConfusionMatrix([]);
    
    // Reinitialize optimizer with new parameters
    const allParams = [...newWeights, ...newBiases];
    if (optimizer === 'sgd') {
      optimizerRef.current = new SGD(allParams, learningRate, momentum);
    } else if (optimizer === 'adam') {
      optimizerRef.current = new Adam(allParams, learningRate);
    } else if (optimizer === 'rmsprop') {
      optimizerRef.current = new RMSprop(allParams, learningRate);
    } else if (optimizer === 'adagrad') {
      optimizerRef.current = new Adagrad(allParams, learningRate);
    } else if (optimizer === 'adamw') {
      optimizerRef.current = new AdamW(allParams, learningRate);
    } else if (optimizer === 'nadam') {
      optimizerRef.current = new Nadam(allParams, learningRate);
    }
    
    addLog(`Network initialized with ${layers.length} layers`, 'success');
    addLog(`Optimizer: ${optimizer.toUpperCase()}, Learning Rate: ${learningRate}`, 'info');
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

  // Calculate accuracy and confusion matrix
  const calculateMetrics = (data, isValidation = false) => {
    if (!data || data.length === 0) return { accuracy: 0, loss: 0 };
    
    let correct = 0;
    let totalLoss = 0;
    const outputSize = layers[layers.length - 1].size;
    const isClassification = outputSize > 1;
    
    // Initialize confusion matrix for classification
    const matrix = isClassification ? Array(outputSize).fill(0).map(() => Array(outputSize).fill(0)) : [];
    
    data.forEach(point => {
      const forward = forwardPass(point.input);
      const prediction = forward.activations[forward.activations.length - 1] || [];
      const target = point.target;
      
      // Calculate loss
      const loss = prediction.reduce((sum, p, i) => sum + (p - (target[i] || 0)) ** 2, 0) / prediction.length;
      totalLoss += loss;
      
      if (isClassification) {
        // For classification: argmax
        const predictedClass = prediction.indexOf(Math.max(...prediction));
        const actualClass = target.indexOf(Math.max(...target));
        
        if (predictedClass === actualClass) correct++;
        
        // Update confusion matrix
        if (matrix[actualClass] && matrix[actualClass][predictedClass] !== undefined) {
          matrix[actualClass][predictedClass]++;
        }
      } else {
        // For regression: consider within 10% as correct
        const error = Math.abs(prediction[0] - target[0]);
        if (error < 0.1) correct++;
      }
    });
    
    const accuracy = (correct / data.length) * 100;
    const avgLoss = totalLoss / data.length;
    
    if (isValidation) {
      setValAccuracy(accuracy);
      setValLoss(avgLoss);
    } else {
      setTrainAccuracy(accuracy);
      setTrainLoss(avgLoss);
    }
    
    if (isClassification && !isValidation) {
      setConfusionMatrix(matrix);
    }
    
    return { accuracy, loss: avgLoss };
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
      
      // Store activations for visualization (update on every sample for smooth animation)
      setActivations(forward.activations);
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
    setSamplesProcessed(s => s + batchSizeActual);
    
    // Track best loss
    if (avgLoss < bestLoss) {
      setBestLoss(avgLoss);
    }
    
    const newEpoch = epochs + 1;
    setEpochs(newEpoch);
    
    // Auto-stop training if target epochs reached
    if (targetEpochs > 0 && newEpoch >= targetEpochs) {
      setIsTraining(false);
      addLog(`🎯 Reached target of ${targetEpochs} epochs! Training stopped.`, 'success');
    }
    
    // Calculate train/val metrics EVERY epoch for live updates
    const valData = dataPoints.slice(0, Math.max(5, Math.floor(dataPoints.length * 0.2)));
    const trainData = dataPoints.slice(Math.max(5, Math.floor(dataPoints.length * 0.2)));
    
    // Use setTimeout to ensure state updates are processed
    setTimeout(() => {
      calculateMetrics(trainData, false);
      calculateMetrics(valData, true);
    }, 0);
    
    // Log every 5 epochs to avoid spam
    if (newEpoch % 5 === 0) {
      addLog(`Epoch ${newEpoch}: Loss = ${avgLoss.toFixed(6)}, Best = ${Math.min(avgLoss, bestLoss).toFixed(6)}`, avgLoss < 0.01 ? 'success' : 'info');
    }
    
    return avgLoss;
  };

  // Toggle training
  const toggleTraining = () => {
    if (weights.length === 0 || biases.length === 0) {
      initializeNetwork();
      // Set a small delay to ensure state updates before starting training
      setTimeout(() => {
        setIsTraining(true);
        addLog('Training started', 'success');
      }, 100);
    } else {
      setIsTraining(!isTraining);
      addLog(isTraining ? 'Training paused' : 'Training resumed', 'warning');
    }
  };

  // Training loop
  useEffect(() => {
    if (isTraining && weights.length > 0) {
      intervalRef.current = setInterval(() => {
        // Run multiple training steps per interval for faster learning
        const stepsPerInterval = Math.max(1, Math.floor(speed / 20));
        let avgLoss = 0;
        
        for (let i = 0; i < stepsPerInterval; i++) {
          avgLoss += trainStep();
        }
        avgLoss /= stepsPerInterval;
        
        if (avgLoss < 0.0001) {
          setIsTraining(false);
          addLog(`Training converged! Final loss: ${avgLoss.toFixed(8)}`, 'success');
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

  // Generate synthetic dataset
  const generateSyntheticDataset = () => {
    const numSamples = Math.floor(Math.random() * 20) + 10; // 10-30 samples
    const inputSize = layers[0].size;
    const outputSize = layers[layers.length - 1].size;
    
    const newData = [];
    for (let i = 0; i < numSamples; i++) {
      const input = Array(inputSize).fill(0).map(() => Math.random());
      const target = Array(outputSize).fill(0).map(() => Math.random());
      newData.push({ input, target });
    }
    
    setDataPoints(newData);
    addLog(`Generated ${numSamples} synthetic data samples`, 'success');
  };

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

  // Suggest network architecture based on dataset
  const suggestArchitecture = (inputSize, outputSize, numSamples) => {
    const isClassification = outputSize >= 2 && datasetId && datasetId.includes('classification');
    
    // Heuristic: hidden layer size based on input/output
    const avgSize = Math.ceil((inputSize + outputSize) / 2);
    const hiddenSize1 = Math.max(4, Math.min(32, avgSize * 2));
    const hiddenSize2 = Math.max(4, Math.min(16, avgSize));
    
    const suggested = [
      { size: inputSize, activation: 'linear', type: 'input' },
      { size: hiddenSize1, activation: 'relu', type: 'hidden' },
      { size: hiddenSize2, activation: 'relu', type: 'hidden' },
      { size: outputSize, activation: isClassification ? 'sigmoid' : 'linear', type: 'output' }
    ];
    
    setSuggestedArchitecture({
      layers: suggested,
      reasoning: `Based on ${inputSize} inputs and ${outputSize} outputs with ${numSamples} samples, we recommend:
      • Input layer: ${inputSize} neurons
      • Hidden layer 1: ${hiddenSize1} neurons (ReLU) - captures complex patterns
      • Hidden layer 2: ${hiddenSize2} neurons (ReLU) - refines features
      • Output layer: ${outputSize} neurons (${isClassification ? 'Sigmoid for classification' : 'Linear for regression'})`
    });
    
    return suggested;
  };
  
  // Apply suggested architecture
  const applySuggestedArchitecture = () => {
    if (suggestedArchitecture) {
      setLayers(suggestedArchitecture.layers);
      addLog('Applied suggested architecture', 'success');
      setTimeout(() => initializeNetwork(), 100);
    }
  };
  
  // Get AI-powered suggestions from Gemini
  const getGeminiSuggestions = async () => {
    if (!selectedDataset || !datasetInfo) {
      addLog('Please load a dataset first', 'warning');
      return;
    }
    
    setAiSuggestionLoading(true);
    addLog('Asking Gemini AI for optimal settings...', 'info');
    
    try {
      const prompt = `You are an expert in deep learning and neural network architecture design. Analyze this dataset and provide optimal network configuration:

Dataset Information:
- Name: ${datasetInfo.name || selectedDataset}
- Samples: ${dataPoints.length}
- Input features: ${dataPoints[0]?.input?.length || 0}
- Output targets: ${dataPoints[0]?.target?.length || 0}
- Task type: ${dataPoints[0]?.target?.length > 1 ? 'Multi-output' : dataPoints[0]?.target?.[0] > 0 && dataPoints[0]?.target?.[0] < 1 ? 'Regression/Classification' : 'Regression'}

Current Architecture:
${layers.map((l, i) => `Layer ${i + 1}: ${l.size} neurons, ${l.activation} activation`).join('\\n')}

Provide a JSON response with:
1. Recommended layer sizes and activations
2. Best optimizer (sgd, adam, rmsprop, adagrad, adamw, nadam)
3. Optimal learning rate
4. Recommended batch size
5. Suggested number of epochs
6. Brief reasoning (2-3 sentences)

Format: {"layers": [{"size": number, "activation": "relu|sigmoid|tanh|linear", "type": "input|hidden|output"}], "optimizer": "adam", "learningRate": 0.001, "batchSize": 32, "epochs": 100, "reasoning": "explanation"}`;

      const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: prompt
            }]
          }]
        })
      });
      
      if (!response.ok) {
        throw new Error(`Gemini API error: ${response.status}`);
      }
      
      const data = await response.json();
      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (!aiText) {
        throw new Error('No response from Gemini');
      }
      
      // Extract JSON from response (handle markdown code blocks)
      const jsonMatch = aiText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('Invalid response format');
      }
      
      const suggestion = JSON.parse(jsonMatch[0]);
      setAiSuggestion(suggestion);
      addLog('✨ AI suggestions received!', 'success');
      
    } catch (error) {
      console.error('Gemini API error:', error);
      addLog(`AI suggestion failed: ${error.message}`, 'error');
    } finally {
      setAiSuggestionLoading(false);
    }
  };
  
  // Apply AI suggestions
  const applyAiSuggestions = () => {
    if (!aiSuggestion) return;
    
    if (aiSuggestion.layers) {
      setLayers(aiSuggestion.layers);
    }
    if (aiSuggestion.optimizer) {
      setOptimizer(aiSuggestion.optimizer);
    }
    if (aiSuggestion.learningRate) {
      setLearningRate(aiSuggestion.learningRate);
    }
    if (aiSuggestion.batchSize) {
      setBatchSize(aiSuggestion.batchSize);
    }
    if (aiSuggestion.epochs) {
      setTargetEpochs(aiSuggestion.epochs);
    }
    
    addLog('✨ Applied AI-recommended settings', 'success');
    setTimeout(() => initializeNetwork(), 100);
    setAiSuggestion(null);
  };
  
  // Get AI-powered suggestions from Gemini
  const getGeminiSuggestions = async () => {
    if (!selectedDataset || !datasetInfo) {
      addLog('Please load a dataset first', 'warning');
      return;
    }
    
    setAiSuggestionLoading(true);
    addLog('Asking Gemini AI for optimal settings...', 'info');
    
    try {
      const prompt = `You are an expert in deep learning and neural network architecture design. Analyze this dataset and provide optimal network configuration:

Dataset Information:
- Name: ${datasetInfo.name || selectedDataset}
- Samples: ${dataPoints.length}
- Input features: ${dataPoints[0]?.input?.length || 0}
- Output targets: ${dataPoints[0]?.target?.length || 0}
- Task type: ${dataPoints[0]?.target?.length > 1 ? 'Multi-output' : dataPoints[0]?.target?.[0] > 0 && dataPoints[0]?.target?.[0] < 1 ? 'Regression/Classification' : 'Regression'}

Current Architecture:
${layers.map((l, i) => `Layer ${i + 1}: ${l.size} neurons, ${l.activation} activation`).join('\n')}

Provide a JSON response with:
1. Recommended layer sizes and activations
2. Best optimizer (sgd, adam, rmsprop, adagrad, adamw, nadam)
3. Optimal learning rate
4. Recommended batch size
5. Suggested number of epochs
6. Brief reasoning (2-3 sentences)

Format: {"layers": [{"size": number, "activation": "relu|sigmoid|tanh|linear", "type": "input|hidden|output"}], "optimizer": "adam", "learningRate": 0.001, "batchSize": 32, "epochs": 100, "reasoning": "explanation"}`;

      const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: prompt
            }]
          }]
        })
      });
      
      if (!response.ok) {
        throw new Error(`Gemini API error: ${response.status}`);
      }
      
      const data = await response.json();
      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (!aiText) {
        throw new Error('No response from Gemini');
      }
      
      // Extract JSON from response (handle markdown code blocks)
      const jsonMatch = aiText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('Invalid response format');
      }
      
      const suggestion = JSON.parse(jsonMatch[0]);
      setAiSuggestion(suggestion);
      addLog('✨ AI suggestions received!', 'success');
      
    } catch (error) {
      console.error('Gemini API error:', error);
      addLog(`AI suggestion failed: ${error.message}`, 'error');
    } finally {
      setAiSuggestionLoading(false);
    }
  };
  
  // Apply AI suggestions
  const applyAiSuggestions = () => {
    if (!aiSuggestion) return;
    
    if (aiSuggestion.layers) {
      setLayers(aiSuggestion.layers);
    }
    if (aiSuggestion.optimizer) {
      setOptimizer(aiSuggestion.optimizer);
    }
    if (aiSuggestion.learningRate) {
      setLearningRate(aiSuggestion.learningRate);
    }
    if (aiSuggestion.batchSize) {
      setBatchSize(aiSuggestion.batchSize);
    }
    if (aiSuggestion.epochs) {
      setTargetEpochs(aiSuggestion.epochs);
    }
    
    addLog('✨ Applied AI-recommended settings', 'success');
    setTimeout(() => initializeNetwork(), 100);
    setAiSuggestion(null);
  };
  
  // Load dataset from JSON file
  const loadDataset = async (datasetId) => {
    if (!datasetId) {
      setSelectedDataset('');
      setDatasetInfo(null);
      return;
    }
    
    try {
      addLog(`Loading dataset: ${datasetId}...`, 'info');
      const response = await fetch(`/datasets/${datasetId}.json`);
      
      if (!response.ok) {
        throw new Error('Dataset not found');
      }
      
      const data = await response.json();
      
      // Handle both formats: plain array or object with data/metadata
      let dataPoints, metadata;
      
      if (Array.isArray(data)) {
        // Plain array format - create metadata from dataset info
        dataPoints = data;
        const datasetInfo = availableDatasets.find(ds => ds.id === datasetId);
        metadata = {
          name: datasetInfo?.name || datasetId,
          description: datasetInfo?.description || 'Dataset',
          samples: data.length,
          input_size: data[0]?.input?.length || 0,
          output_size: data[0]?.target?.length || 0
        };
      } else {
        // Object format with data and metadata
        dataPoints = data.data;
        metadata = data.metadata;
      }
      
      // Validate dataset structure
      if (!dataPoints || !Array.isArray(dataPoints) || dataPoints.length === 0) {
        throw new Error('Invalid dataset format - empty or not an array');
      }
      
      if (!dataPoints[0].input || !dataPoints[0].target) {
        throw new Error('Dataset samples missing input or target fields');
      }
      
      setDataPoints(dataPoints);
      setDatasetInfo(metadata);
      setSelectedDataset(datasetId);
      addLog(`✓ Loaded ${dataPoints.length} samples from ${metadata.name}`, 'success');
      
      // Auto-adjust network architecture based on dataset
      const inputSize = dataPoints[0].input.length;
      const outputSize = dataPoints[0].target.length;
      
      // Update layers to match dataset dimensions
      const newLayers = [
        { size: inputSize, activation: 'linear', type: 'input' },
        ...layers.slice(1, -1), // Keep hidden layers
        { size: outputSize, activation: layers[layers.length - 1].activation, type: 'output' }
      ];
      
      setLayers(newLayers);
      addLog(`Network adjusted: ${inputSize} inputs → ${outputSize} outputs`, 'info');
      
      // Suggest optimal architecture
      suggestArchitecture(inputSize, outputSize, dataPoints.length);
      
    } catch (error) {
      addLog(`✗ Failed to load dataset: ${error.message}`, 'error');
      console.error('Dataset load error:', error);
    }
  };

  // Update data point
  const updateDataPoint = (index, field, valueIndex, value) => {
    if (!dataPoints) return;
    const newDataPoints = [...dataPoints];
    newDataPoints[index][field][valueIndex] = parseFloat(value) || 0;
    setDataPoints(newDataPoints);
  };

  // Remove data point
  const removeDataPoint = (index) => {
    if (!dataPoints || dataPoints.length <= 1) return;
    setDataPoints(dataPoints.filter((_, i) => i !== index));
  };

  // Initialize on mount
  useEffect(() => {
    initializeNetwork();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-3 sm:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-4">
            <div className="flex items-center gap-3">
              <Brain className="w-10 h-10 sm:w-12 sm:h-12 text-purple-400" />
              <div>
                <h1 className="text-2xl sm:text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Deeplex Learning Platform
                </h1>
                <p className="text-purple-300 text-xs sm:text-sm mt-1">
                  Configure, train, and visualize neural networks with complete control
                </p>
              </div>
            </div>
            
            <div className="flex gap-2 sm:gap-3 w-full sm:w-auto">
              <button
                onClick={toggleTraining}
                disabled={!dataPoints || dataPoints.length === 0}
                className={`${
                  isTraining 
                    ? 'bg-red-600 hover:bg-red-700' 
                    : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
                } disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-4 sm:px-6 py-2 sm:py-3 font-semibold flex items-center justify-center gap-2 transition-all shadow-lg flex-1 sm:flex-none text-sm sm:text-base`}
              >
                {isTraining ? <><Pause className="w-4 h-4 sm:w-5 sm:h-5" /> Pause</> : <><Play className="w-4 h-4 sm:w-5 sm:h-5" /> Train</>}
              </button>
              <button
                onClick={initializeNetwork}
                className="bg-indigo-600 hover:bg-indigo-700 rounded-lg px-4 sm:px-6 py-2 sm:py-3 font-semibold flex items-center justify-center gap-2 transition-all flex-1 sm:flex-none text-sm sm:text-base"
              >
                <RotateCcw className="w-4 h-4 sm:w-5 sm:h-5" /> Reset
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {/* Configuration Panel */}
          <div className="space-y-4 sm:space-y-6">
            {/* AI Suggestion Button */}
            <button
              onClick={getGeminiSuggestions}
              disabled={aiSuggestionLoading || !selectedDataset}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl px-4 py-3 font-semibold transition-all flex items-center justify-center gap-2"
            >
              <Sparkles className={`w-5 h-5 ${aiSuggestionLoading ? 'animate-pulse' : ''}`} />
              {aiSuggestionLoading ? 'Asking Gemini AI...' : '✨ Get AI Suggestions'}
            </button>
            
            {/* AI Suggestions Panel */}
            {aiSuggestion && (
              <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 backdrop-blur-lg rounded-xl p-4 border border-purple-500/40">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-purple-400" />
                    <h3 className="text-sm font-semibold">Gemini AI Recommendations</h3>
                  </div>
                  <button
                    onClick={applyAiSuggestions}
                    className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded-lg text-xs font-medium transition-colors"
                  >
                    Apply All
                  </button>
                </div>
                <p className="text-xs text-slate-300 mb-3">
                  {aiSuggestion.reasoning}
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-slate-400">Optimizer</div>
                    <div className="font-semibold text-purple-300">{aiSuggestion.optimizer?.toUpperCase()}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-slate-400">Learning Rate</div>
                    <div className="font-semibold text-purple-300">{aiSuggestion.learningRate}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-slate-400">Batch Size</div>
                    <div className="font-semibold text-purple-300">{aiSuggestion.batchSize}</div>
                  </div>
                  <div className="bg-slate-800/50 rounded p-2">
                    <div className="text-slate-400">Epochs</div>
                    <div className="font-semibold text-purple-300">{aiSuggestion.epochs}</div>
                  </div>
                </div>
                {aiSuggestion.layers && (
                  <div className="mt-3 space-y-1">
                    <div className="text-xs text-slate-400 font-semibold">Architecture:</div>
                    {aiSuggestion.layers.map((layer, i) => (
                      <div key={i} className="text-xs text-slate-300">
                        Layer {i + 1}: {layer.size} neurons ({layer.activation})
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            
            {/* Architecture Suggestion Panel */}
            {suggestedArchitecture && !aiSuggestion && (
              <div className="bg-gradient-to-br from-purple-600/20 to-pink-600/20 backdrop-blur-lg rounded-xl p-4 border border-purple-500/40">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-purple-300" />
                    <h3 className="text-sm font-bold text-purple-200">Suggested Architecture</h3>
                  </div>
                  <button
                    onClick={applySuggestedArchitecture}
                    className="bg-purple-600 hover:bg-purple-700 text-white text-xs px-3 py-1 rounded-lg font-semibold transition-all flex items-center gap-1"
                  >
                    <Zap className="w-3 h-3" />
                    Apply
                  </button>
                </div>
                
                <div className="text-xs text-purple-100 leading-relaxed whitespace-pre-line mb-3">
                  {suggestedArchitecture.reasoning}
                </div>
                
                <div className="flex flex-wrap gap-2">
                  {suggestedArchitecture.layers.map((layer, i) => (
                    <div 
                      key={i}
                      className="bg-purple-900/30 border border-purple-500/30 rounded-lg px-2 py-1 text-xs"
                    >
                      <span className="text-purple-300 font-semibold">{layer.size}</span>
                      <span className="text-purple-400 ml-1">({layer.activation})</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {/* Architecture Configuration */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 sm:p-5 border border-white/20">
              <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                <Settings className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400" />
                Network Architecture
              </h2>
              
              <div className="space-y-3 max-h-96 pr-2" style={{overflowY: 'auto', overflowX: 'visible'}}>
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
                          disabled={isTraining}
                          className="w-full bg-slate-900 border border-slate-600 rounded px-3 py-1.5 text-sm disabled:opacity-50 text-white"
                        />
                      </div>
                      
                      {layer.type !== 'input' && (
                        <div>
                          <label className="text-xs sm:text-sm text-slate-300 block mb-2">Activation</label>
                          <select
                            value={layer.activation}
                            onChange={(e) => updateLayer(idx, 'activation', e.target.value)}
                            disabled={isTraining}
                            className="w-full bg-slate-900 border border-slate-600 rounded px-2 sm:px-3 py-2 text-xs sm:text-sm disabled:opacity-50"
                          >
                            <option value="relu">ReLU</option>
                            <option value="sigmoid">Sigmoid</option>
                            <option value="tanh">Tanh</option>
                            <option value="linear">Linear</option>
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
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 sm:p-5 border border-white/20">
              <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400" />
                Optimizer Settings
              </h2>
              
              <div className="space-y-3">
                <div>
                  <label className="text-xs sm:text-sm text-slate-300 block mb-2">Optimizer</label>
                  <select
                    value={optimizer}
                    onChange={(e) => {
                      setOptimizer(e.target.value);
                      initializeNetwork();
                    }}
                    disabled={isTraining}
                    className="w-full bg-slate-900 border border-slate-600 rounded px-2 sm:px-3 py-2 text-xs sm:text-sm disabled:opacity-50"
                  >
                    <option value="sgd">SGD - Stochastic Gradient Descent</option>
                    <option value="adam">Adam - Adaptive Moment Estimation</option>
                    <option value="rmsprop">RMSprop - Root Mean Square Propagation</option>
                    <option value="adagrad">Adagrad - Adaptive Gradient</option>
                    <option value="adamw">AdamW - Adam with Weight Decay</option>
                    <option value="nadam">Nadam - Nesterov Adam</option>
                  </select>
                </div>
                
                <div>
                  <label className="text-xs sm:text-sm text-slate-300 block mb-2">
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
                    max="1024"
                    step="1"
                    value={Math.min(batchSize, 1024)}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    disabled={isTraining}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>1</span>
                    <span>256</span>
                    <span>512</span>
                    <span>1024</span>
                  </div>
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
                
                <div>
                  <label className="text-sm text-slate-300 block mb-2">
                    Target Epochs: {targetEpochs === 0 ? 'Unlimited' : targetEpochs}
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1000"
                    step="50"
                    value={targetEpochs}
                    onChange={(e) => setTargetEpochs(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Unlimited</span>
                    <span>500</span>
                    <span>1000</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Data Configuration */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-5 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-green-400" />
                Training Data
              </h2>
              
              {/* Dataset Selector */}
              <div className="mb-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Load Sample Dataset
                </label>
                <select
                  value={selectedDataset}
                  onChange={(e) => loadDataset(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="">Manual Entry (Current)</option>
                  {availableDatasets.map(ds => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} - {ds.description}
                    </option>
                  ))}
                </select>
                
                {datasetInfo && (
                  <div className="mt-3 p-3 bg-purple-900/20 border border-purple-500/30 rounded-lg">
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Dataset:</span>
                        <span className="text-purple-300 font-medium">{datasetInfo.name}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Samples:</span>
                        <span className="text-white font-mono">{datasetInfo.samples}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Dimensions:</span>
                        <span className="text-white font-mono">{datasetInfo.input_size} → {datasetInfo.output_size}</span>
                      </div>
                      <p className="text-slate-300 text-xs mt-2 italic">{datasetInfo.description}</p>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                {dataPoints && dataPoints.map((point, idx) => (
                  <div key={idx} className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-medium text-slate-400">Sample {idx + 1}</span>
                      <button
                        onClick={() => removeDataPoint(idx)}
                        disabled={!dataPoints || dataPoints.length <= 1}
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
          <div className="lg:col-span-2 space-y-4 sm:space-y-6">
            {/* Overall Progress Bar */}
            {isTraining && targetEpochs > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-white">Overall Training Progress</span>
                  <span className="text-xs text-slate-300">
                    {Math.min(Math.floor((epochs / targetEpochs) * 100), 100)}%
                  </span>
                </div>
                <div className="w-full bg-slate-700/50 rounded-full h-3 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 transition-all duration-500"
                    style={{ width: `${Math.min((epochs / targetEpochs) * 100, 100)}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-slate-400 mt-1">
                  <span>Epoch {epochs} / {targetEpochs}</span>
                  <span>{samplesProcessed.toLocaleString()} samples processed</span>
                </div>
              </div>
            )}
            
            {/* Train/Val Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3">
              <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 backdrop-blur-lg rounded-xl p-3 border border-purple-500/30">
                <div className="text-xs text-purple-300 mb-1">Train Loss</div>
                <div className="text-lg sm:text-2xl font-bold">
                  {trainLoss !== null ? trainLoss.toFixed(4) : '—'}
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-pink-600/20 to-pink-800/20 backdrop-blur-lg rounded-xl p-3 border border-pink-500/30">
                <div className="text-xs text-pink-300 mb-1">Val Loss</div>
                <div className="text-lg sm:text-2xl font-bold">
                  {valLoss !== null ? valLoss.toFixed(4) : '—'}
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 backdrop-blur-lg rounded-xl p-3 border border-green-500/30">
                <div className="text-xs text-green-300 mb-1">Train Acc</div>
                <div className="text-lg sm:text-2xl font-bold">
                  {trainAccuracy.toFixed(1)}%
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 backdrop-blur-lg rounded-xl p-3 border border-blue-500/30">
                <div className="text-xs text-blue-300 mb-1">Val Acc</div>
                <div className="text-lg sm:text-2xl font-bold">
                  {valAccuracy.toFixed(1)}%
                </div>
              </div>
            </div>
            
            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-4">
              <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 backdrop-blur-lg rounded-xl p-3 sm:p-5 border border-purple-500/30">
                <div className="text-xs sm:text-sm text-purple-300 mb-1">Epoch</div>
                <div className="text-xl sm:text-3xl font-bold">{epochs}</div>
              </div>
              
              <div className="bg-gradient-to-br from-red-600/20 to-red-800/20 backdrop-blur-lg rounded-xl p-3 sm:p-5 border border-red-500/30">
                <div className="text-xs sm:text-sm text-red-300 mb-1">Loss</div>
                <div className="text-xl sm:text-3xl font-bold">
                  {currentLoss !== null ? currentLoss.toFixed(6) : '—'}
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 backdrop-blur-lg rounded-xl p-3 sm:p-5 border border-purple-500/30">
                <div className="text-xs sm:text-sm text-green-300 mb-1">Best Loss</div>
                <div className="text-xl sm:text-3xl font-bold">
                  {bestLoss !== Infinity ? bestLoss.toFixed(6) : '—'}
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 backdrop-blur-lg rounded-xl p-3 sm:p-5 border border-blue-500/30">
                <div className="text-xs sm:text-sm text-blue-300 mb-1">Samples</div>
                <div className="text-xl sm:text-3xl font-bold">{samplesProcessed.toLocaleString()}</div>
              </div>
            </div>
            
            {/* Training Progress Bar */}
            {isTraining && currentLoss !== null && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-slate-300 font-medium">Training Progress</span>
                  <span className="text-xs text-slate-400">
                    {currentLoss < 0.1 ? 'Converging...' : currentLoss < 0.5 ? 'Learning...' : 'Starting...'}
                  </span>
                </div>
                <div className="w-full bg-slate-700/50 rounded-full h-2 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 via-pink-500 to-green-500 transition-all duration-500"
                    style={{ 
                      width: `${Math.min((1 - Math.min(currentLoss, 1)) * 100, 100)}%` 
                    }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>Loss: {currentLoss.toFixed(6)}</span>
                  <span>Target: &lt; 0.001</span>
                </div>
              </div>
            )}

            {/* Network Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 sm:p-6 border border-white/20 relative">
              {isTraining && (
                <div className="absolute top-3 right-3 flex items-center gap-2 bg-green-500/20 border border-green-500/50 rounded-full px-3 py-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-xs text-green-300 font-semibold">Training...</span>
                </div>
              )}
              <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                <Brain className="w-4 h-4 sm:w-5 sm:h-5 text-purple-400" />
                Network Visualization
              </h2>
              
              {activations.length > 0 ? (
                <div className="flex justify-around items-center min-h-[200px] sm:min-h-[300px] overflow-x-auto">
                  {layers.map((layer, layerIdx) => (
                    <div key={layerIdx} className="flex flex-col items-center gap-2 sm:gap-3 min-w-[60px] sm:min-w-auto">
                      <div className="text-[10px] sm:text-xs text-slate-400 mb-1 sm:mb-2 font-semibold">
                        {layer.type === 'input' ? 'Input' : 
                         layer.type === 'output' ? 'Output' : 
                         `Hidden ${layerIdx}`}
                      </div>
                      {activations[layerIdx]?.map((activation, neuronIdx) => {
                        const absVal = Math.abs(activation);
                        const intensity = Math.min(absVal, 1);
                        const isPositive = activation > 0;
                        
                        return (
                          <div
                            key={neuronIdx}
                            className="relative w-10 h-10 sm:w-12 sm:h-12 rounded-full flex items-center justify-center text-[10px] sm:text-xs font-mono border-2 shadow-lg transition-all duration-300"
                            style={{
                              backgroundColor: isPositive 
                                ? `rgba(34, 197, 94, ${0.2 + intensity * 0.8})` 
                                : `rgba(239, 68, 68, ${0.2 + intensity * 0.8})`,
                              borderColor: isPositive 
                                ? `rgba(34, 197, 94, ${0.5 + intensity * 0.5})`
                                : `rgba(239, 68, 68, ${0.5 + intensity * 0.5})`,
                              boxShadow: isTraining 
                                ? `0 0 ${10 + intensity * 20}px ${isPositive ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)'}` 
                                : 'none',
                              transform: isTraining ? `scale(${1 + intensity * 0.1})` : 'scale(1)'
                            }}
                          >
                            <span className="text-white font-bold drop-shadow">{activation.toFixed(2)}</span>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="min-h-[200px] sm:min-h-[300px] flex items-center justify-center text-slate-400">
                  Start training to see network activations
                </div>
              )}
            </div>

            {/* Predictions vs Targets Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-blue-400" />
                Validation Samples
              </h2>
              
              {weights.length > 0 && biases.length > 0 && dataPoints && dataPoints.length > 0 ? (
                <div className="space-y-3">
                  {/* Show only first 5 samples as validation set */}
                  {dataPoints.slice(0, 5).map((point, idx) => {
                    const forward = forwardPass(point.input);
                    const prediction = forward.activations[forward.activations.length - 1] || [0];
                    const target = point.target;
                    const error = prediction.reduce((sum, p, i) => sum + Math.abs(p - (target[i] || 0)), 0) / prediction.length;
                    const errorPercent = Math.min(error * 100, 100);
                    
                    return (
                      <div key={idx} className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-slate-400">Sample {idx + 1}</span>
                          <span className="text-xs font-mono text-slate-300">
                            Error: {error.toFixed(4)}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-3 mb-3">
                          <div>
                            <div className="text-xs text-blue-300 mb-1">Input</div>
                            <div className="text-xs font-mono text-slate-200">
                              [{point.input.map(v => v.toFixed(2)).join(', ')}]
                            </div>
                          </div>
                          <div>
                            <div className="text-xs text-red-300 mb-1 flex items-center justify-between">
                              <span>Error</span>
                              <span className="font-bold">{error.toFixed(4)}</span>
                            </div>
                            <div className="relative h-2 bg-slate-900 rounded-full overflow-hidden">
                              <div 
                                className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500 to-red-500 transition-all duration-300"
                                style={{ width: `${Math.min(errorPercent, 100)}%` }}
                              />
                            </div>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-purple-900/30 border border-purple-500/30 rounded p-2">
                            <div className="text-xs text-purple-300 mb-1 font-semibold">Target</div>
                            <div className="text-sm font-mono text-white">
                              [{target.map(v => v.toFixed(3)).join(', ')}]
                            </div>
                          </div>
                          <div className="bg-green-900/30 border border-green-500/30 rounded p-2">
                            <div className="text-xs text-green-300 mb-1 font-semibold">Prediction</div>
                            <div className="text-sm font-mono text-white">
                              [{prediction.map(v => v.toFixed(3)).join(', ')}]
                            </div>
                          </div>
                        </div>
                        
                        <div className="text-xs text-center mt-2" style={{
                          color: errorPercent < 1 ? '#22c55e' : 
                                 errorPercent < 5 ? '#84cc16' : 
                                 errorPercent < 20 ? '#eab308' : '#ef4444'
                        }}>
                          {errorPercent < 1 ? '✓ Excellent match!' : 
                           errorPercent < 5 ? '○ Good prediction' : 
                           errorPercent < 20 ? '◐ Learning...' : 
                           '◯ Needs training'}
                        </div>
                      </div>
                    );
                  })}
                  
                  {dataPoints.length > 5 && (
                    <div className="text-center text-xs text-slate-400 py-2 border-t border-slate-700">
                      Showing 5 of {dataPoints.length} samples • Using remaining {dataPoints.length - 5} for training
                    </div>
                  )}
                  
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

            {/* Confusion Matrix - Only for Classification */}
            {layers.length > 0 && layers[layers.length - 1] > 1 && confusionMatrix.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <Target className="w-5 h-5 text-purple-400" />
                  Confusion Matrix (Validation Set)
                </h2>
                
                <div className="overflow-x-auto">
                  <div className="inline-block min-w-full">
                    <div className="text-xs text-slate-400 mb-3">
                      Shows how often each predicted class matches the actual class. Diagonal = correct predictions.
                    </div>
                    
                    <div className="grid gap-1" style={{ 
                      gridTemplateColumns: `auto repeat(${confusionMatrix.length}, minmax(50px, 1fr))` 
                    }}>
                      {/* Header Row */}
                      <div className="text-xs text-slate-400 p-2"></div>
                      {confusionMatrix.map((_, i) => (
                        <div key={`header-${i}`} className="text-xs text-center text-purple-300 font-semibold p-2">
                          Pred {i}
                        </div>
                      ))}
                      
                      {/* Matrix Rows */}
                      {confusionMatrix.map((row, i) => {
                        const maxVal = Math.max(...confusionMatrix.flat());
                        return (
                          <React.Fragment key={`row-${i}`}>
                            <div className="text-xs text-blue-300 font-semibold p-2 flex items-center">
                              Act {i}
                            </div>
                            {row.map((val, j) => {
                              const intensity = maxVal > 0 ? val / maxVal : 0;
                              const isCorrect = i === j;
                              return (
                                <div 
                                  key={`cell-${i}-${j}`}
                                  className={`p-3 rounded text-center text-sm font-mono transition-all ${
                                    isCorrect 
                                      ? 'border-2 border-green-400/50' 
                                      : 'border border-slate-600/30'
                                  }`}
                                  style={{
                                    backgroundColor: isCorrect
                                      ? `rgba(34, 197, 94, ${0.1 + intensity * 0.6})`
                                      : `rgba(148, 163, 184, ${0.05 + intensity * 0.4})`,
                                  }}
                                >
                                  <div className={isCorrect ? 'text-green-200 font-bold' : 'text-slate-300'}>
                                    {val}
                                  </div>
                                </div>
                              );
                            })}
                          </React.Fragment>
                        );
                      })}
                    </div>
                    
                    <div className="text-xs text-slate-400 mt-3">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-green-500/40 border-2 border-green-400/50 rounded"></div>
                        <span>Diagonal cells = Correct predictions</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Weight Distribution Visualization */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-yellow-400" />
                Weight Distributions
              </h2>
              
              {weights.length > 0 ? (
                <div className="space-y-4">
                  {weights.map((layerWeights, idx) => {
                    // Use reduce instead of Math.max(...array) to avoid stack overflow with large arrays
                    const maxWeight = Math.max(
                      layerWeights.reduce((max, w) => Math.max(max, Math.abs(w)), 0),
                      0.01
                    );
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
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 sm:p-6 border border-white/20">
              <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 sm:w-5 sm:h-5 text-green-400" />
                Training Progress
              </h2>
              
              {lossHistory.length > 0 ? (
                <div className="h-32 sm:h-48 bg-slate-900/50 rounded-lg p-2 sm:p-4 flex items-end gap-0.5">
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
                <div className="h-32 sm:h-48 bg-slate-900/50 rounded-lg p-4 flex items-center justify-center text-slate-400 text-xs sm:text-sm">
                  Loss history will appear here during training
                </div>
              )}
              
              <div className="mt-3 sm:mt-4 text-[10px] sm:text-xs text-slate-400">
                This chart shows how the loss decreases as the network learns. A downward trend indicates successful learning, where the optimizer is finding better parameter values that minimize the prediction error on your training data.
              </div>
            </div>

            {/* Educational Info */}
            <div className="bg-gradient-to-br from-indigo-600/10 to-purple-600/10 backdrop-blur-lg rounded-xl p-4 sm:p-6 border border-indigo-500/20">
              <h3 className="text-base sm:text-lg font-semibold mb-2 sm:mb-3 text-indigo-300">Understanding Your Configuration</h3>
              <div className="text-xs sm:text-sm text-slate-300 space-y-2 leading-relaxed">
                <p>
                  You have configured a neural network with {layers.length} layers, using the {optimizer === 'adam' ? 'Adam' : 'SGD'} optimizer. {optimizer === 'adam' ? 'Adam adapts the learning rate for each parameter based on estimates of first and second moments of the gradients, which often leads to faster convergence.' : momentum > 0 ? `With momentum set to ${momentum.toFixed(2)}, the optimizer remembers past gradients to accelerate training in consistent directions and dampen oscillations.` : 'Without momentum, each update is based solely on the current gradient.'}
                </p>
                <p className="hidden sm:block">
                  Your batch size of {batchSize} means the network processes {batchSize} {batchSize === 1 ? 'sample' : 'samples'} before updating its parameters. Smaller batches provide noisier gradient estimates but update more frequently, while larger batches give more accurate gradients but update less often. The learning rate of {learningRate.toFixed(4)} controls how much the parameters change with each update. Too high and training becomes unstable, too low and convergence takes forever.
                </p>
                <p className="hidden md:block">
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