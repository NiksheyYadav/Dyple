# NeuralForge - Interactive Neural Network Learning Platform

<div align="center">

![NeuralForge Banner](https://github.com/user-attachments/assets/f98676a2-9a01-4543-a6bb-23818e6cc515)

**An advanced, interactive neural network visualization and training platform built with React and TypeScript**

[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](LICENSE)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178C6?logo=typescript)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.4-646CFF?logo=vite)](https://vitejs.dev/)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots](#-screenshots) • [Technologies](#-technologies)

</div>

---

## 📖 Overview

**NeuralForge** is a comprehensive educational platform designed to demystify neural networks through interactive visualization and hands-on learning. Whether you're a student, educator, or AI enthusiast, NeuralForge provides an intuitive environment to build, train, and understand neural networks without writing a single line of code.

> **Note**: The repository is currently hosted under the name "Dyple" but the application has been rebranded to "NeuralForge" for better clarity and memorability.

### What Makes NeuralForge Special?

- **Zero Code Required**: Build and train neural networks entirely through an intuitive UI
- **Real-Time Visualization**: Watch your network learn with live updates of activations, weights, and loss
- **Educational Focus**: Detailed explanations help you understand what's happening under the hood
- **Production-Quality Code**: Built with modern React, TypeScript, and industry best practices
- **Fully Interactive**: Experiment with different architectures, optimizers, and hyperparameters instantly

---

## ✨ Features

### 🧠 Interactive Neural Network Builder
- **Dynamic Layer Management**: Add or remove hidden layers on the fly
- **Flexible Architecture**: Configure the number of neurons in each layer (1-20 neurons)
- **Multiple Activation Functions**: Choose from ReLU, Sigmoid, Tanh, and Linear activations
- **Visual Feedback**: See your network architecture in a clean, organized interface

### ⚡ Advanced Optimization Algorithms
- **SGD with Momentum**: Classic stochastic gradient descent with configurable momentum (0-0.99)
- **Adam Optimizer**: Adaptive learning rates with automatic bias correction
- **RMSprop**: Root mean square propagation for adaptive learning
- **Adagrad**: Adaptive gradient algorithm
- **AdamW**: Adam with weight decay regularization
- **Nadam**: Nesterov-accelerated Adam
- **Adjustable Learning Rate**: Fine-tune learning rate from 0.0001 to 0.1
- **Mini-Batch Training**: Configure batch sizes to balance training speed and stability

### 🚀 GPU Acceleration
- **WebGPU Support**: Automatic detection of GPU availability
- **Hardware Acceleration**: Enable GPU for faster training computations
- **Fallback to CPU**: Seamless operation when GPU is not available
- **Real-time Toggle**: Switch between CPU and GPU during setup

### 🤖 AI-Powered Suggestions
- **Gemini AI Integration**: Get intelligent recommendations for network architecture
- **Optimal Hyperparameters**: AI suggests learning rate, batch size, and epochs
- **Smart Architecture**: Receive layer size and activation function recommendations
- **Dataset Analysis**: AI analyzes your data to provide tailored suggestions
- **One-Click Apply**: Instantly apply all AI recommendations

### 📊 Real-Time Visualizations
- **Network Activations**: Visual representation of neuron activations across all layers
- **Weight Distribution Histograms**: Monitor how weights evolve during training
- **Training Progress Chart**: Track loss reduction over time
- **Prediction Accuracy**: Compare model predictions against target values with error bars
- **Color-Coded Feedback**: Green for positive activations, red for negative, with intensity indicating magnitude

### 🎯 Custom Training Data Management
- **Add/Remove Samples**: Build your custom dataset with unlimited training examples
- **Live Data Editing**: Modify input and target values during training
- **Multi-Input/Output Support**: Configure networks for various input/output dimensions
- **Immediate Feedback**: See how data changes affect predictions instantly

### 📈 Performance Metrics & Monitoring
- **Real-Time Loss Tracking**: Monitor Mean Squared Error (MSE) loss with 6-digit precision
- **Epoch Counter**: Track training iterations
- **Batch Size Display**: See current batch configuration
- **Error Analysis**: Per-sample error breakdown with visual indicators
- **Training Speed Control**: Adjust visualization update speed (10-100%)

### 🎨 Beautiful, Modern UI
- **Gradient Design**: Stunning purple-to-slate gradient background
- **Glass-Morphism**: Modern backdrop-blur effects for panels
- **Responsive Layout**: Three-column adaptive layout for optimal space usage
- **Smooth Animations**: Transition effects for interactive elements
- **Dark Theme**: Eye-friendly dark interface perfect for extended use
- **Professional Typography**: Clean, readable fonts with proper hierarchy

### 🎓 Educational Content
- **Contextual Explanations**: Learn what each parameter does and why it matters
- **Configuration Insights**: Understand your current network setup with plain English descriptions
- **Training Guidance**: Visual cues (Excellent match, Good prediction, Learning in progress, Needs more training)
- **Algorithm Explanations**: Built-in descriptions of optimization algorithms and activation functions

---

## 🚀 Installation

### Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **npm** (comes with Node.js) or **yarn**

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/NiksheyYadav/Dyple.git
   cd Dyple
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   VITE_GEMINI_API_KEY=your_api_key_here
   ```
   
   > Get your free API key at [Google AI Studio](https://makersuite.google.com/app/apikey)

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:5173` and start exploring!

### Build for Production

To create an optimized production build:

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

To preview the production build locally:

```bash
npm run preview
```

---

## 📘 Usage

### Getting Started with Your First Neural Network

#### Step 1: Configure Network Architecture

1. Start with the default 3-layer network (Input → Hidden → Output)
2. Adjust the number of neurons in the hidden layer (default: 4)
3. Add more hidden layers by clicking **"Add Hidden Layer"** if needed
4. Select activation functions for each layer:
   - **ReLU**: Great for hidden layers, prevents vanishing gradients
   - **Sigmoid**: Good for binary classification outputs
   - **Tanh**: Alternative to sigmoid with centered output
   - **Linear**: For regression outputs

#### Step 2: Set Optimizer Parameters

1. **Choose an optimizer**:
   - **Adam**: Recommended for beginners (adaptive learning, faster convergence)
   - **SGD**: More control with momentum parameter

2. **Adjust the Learning Rate** (default: 0.01):
   - Start with 0.01 for most problems
   - Increase for faster learning (risk: instability)
   - Decrease if training is unstable

3. **Configure Batch Size** (default: 4):
   - Smaller batches: More frequent updates, noisier gradients
   - Larger batches: More stable gradients, fewer updates

4. **Set Training Speed** (visualization only):
   - Control how fast you see updates (doesn't affect learning)

#### Step 3: Add Training Data

1. Click **"Add Data Point"** to create new training samples
2. Enter **Input values**: Features for your model (e.g., [0.5, 0.3])
3. Enter **Target values**: Expected outputs (e.g., [1.0])
4. Add multiple samples to create a meaningful dataset
5. Remove samples by clicking the trash icon

#### Step 4: Train Your Network

1. Click the **"Train"** button to start training
2. Watch the visualizations update in real-time:
   - **Network Visualization**: See neuron activations
   - **Predictions vs Targets**: Monitor prediction accuracy
   - **Weight Distributions**: Observe weight evolution
   - **Training Progress**: Track loss reduction

3. Click **"Pause"** to temporarily stop training
4. Click **"Reset"** to reinitialize the network with new random weights

#### Step 5: Analyze Results

- **Check the Loss**: Lower is better (values near 0 indicate good fit)
- **Review Predictions**: Compare predicted vs target values
- **Monitor Error Bars**: Green = accurate, Red = needs improvement
- **Observe Weight Changes**: Healthy training shows evolving distributions

### Advanced Usage Tips

- **Experiment with Architecture**: Try different numbers of layers and neurons
- **Compare Optimizers**: Train the same dataset with SGD vs Adam
- **Adjust Learning Rate**: Find the sweet spot between speed and stability
- **Add More Data**: More training samples generally lead to better generalization
- **Watch for Overfitting**: If loss gets very low but predictions seem off, you might be overfitting

---

## 📸 Screenshots

### Initial State - Network Configuration
![Initial State](https://github.com/user-attachments/assets/f98676a2-9a01-4543-a6bb-23818e6cc515)
*Configure your neural network architecture, optimizer settings, and training data before starting*

### Active Training - Real-Time Visualization
![Training Active](https://github.com/user-attachments/assets/9d1d5774-644e-40af-b8cf-ad7fe54313a5)
*Watch your network learn in real-time with live updates of activations, weights, loss, and predictions*

---

## 🛠 Technologies

### Core Framework
- **[React 19](https://reactjs.org/)** - Modern UI library with concurrent features
- **[TypeScript 5.9](https://www.typescriptlang.org/)** - Type-safe JavaScript for robust code
- **[Vite 6.4](https://vitejs.dev/)** - Lightning-fast build tool and dev server

### Styling & UI
- **[Tailwind CSS 4.1](https://tailwindcss.com/)** - Utility-first CSS framework
- **[Lucide React](https://lucide.dev/)** - Beautiful, consistent icon library
- **PostCSS & Autoprefixer** - CSS processing and compatibility

### Development Tools
- **React DOM 19** - React rendering for web
- **@vitejs/plugin-react** - Fast Refresh and JSX support
- **TypeScript ESM** - Modern ES modules support

---

## 📁 Project Structure

```
NeuralForge/
├── src/
│   ├── App.tsx           # Main application component with neural network logic
│   ├── main.tsx          # Application entry point
│   └── index.css         # Global styles and Tailwind imports
├── public/               # Static assets
├── index.html            # HTML template
├── package.json          # Project dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── vite.config.ts        # Vite build configuration
├── tailwind.config.js    # Tailwind CSS configuration
├── postcss.config.js     # PostCSS configuration
└── README.md            # This file
```

---

## 🧮 How It Works

### Neural Network Implementation

NeuralForge implements a fully functional feedforward neural network from scratch:

1. **Initialization**: Weights are initialized using He initialization for ReLU and Xavier initialization for other activations
2. **Forward Pass**: Input propagates through layers with weighted sums and activation functions
3. **Backward Pass**: Gradients are computed using backpropagation and the chain rule
4. **Optimization**: SGD with momentum or Adam optimizer updates weights based on gradients
5. **Mini-Batch Training**: Gradients are accumulated over a batch before updating weights

### Activation Functions

- **ReLU**: `f(x) = max(0, x)` - Fast, prevents vanishing gradients
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))` - Squashes output to [0, 1]
- **Tanh**: `f(x) = tanh(x)` - Squashes output to [-1, 1]
- **Linear**: `f(x) = x` - No transformation, for regression

### Optimizers

**SGD with Momentum**:
```
velocity = momentum * velocity - learning_rate * gradient
parameter += velocity
```

**Adam Optimizer**:
```
m = β₁ * m + (1 - β₁) * gradient          # First moment
v = β₂ * v + (1 - β₂) * gradient²         # Second moment
m_hat = m / (1 - β₁^t)                    # Bias correction
v_hat = v / (1 - β₂^t)                    # Bias correction
parameter -= learning_rate * m_hat / (√v_hat + ε)
```

---

## 🤝 Contributing

Contributions are welcome! Whether it's:

- 🐛 Bug reports
- ✨ Feature requests
- 📝 Documentation improvements
- 🔧 Code contributions

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **ISC License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Nikshey Yadav**

- GitHub: [@NiksheyYadav](https://github.com/NiksheyYadav)

---

## 🙏 Acknowledgments

- Inspired by the need for accessible machine learning education
- Built with modern web technologies for optimal performance
- Designed to make neural networks understandable for everyone

---

## 📚 Learn More

### Recommended Resources

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng
- [Understanding Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) - 3Blue1Brown

### Related Projects

- TensorFlow Playground - Google's neural network visualization
- ConvNetJS - Deep Learning in JavaScript
- ML5.js - Friendly Machine Learning for the Web

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and React

</div>
