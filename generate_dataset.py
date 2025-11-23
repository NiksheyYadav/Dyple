"""
Generate a large synthetic dataset for the Deeplex Neural Network Trainer
This script creates various types of datasets for training and testing
"""

import json
import numpy as np
from typing import List, Dict

def generate_regression_dataset(num_samples: int = 1000, num_inputs: int = 2, num_outputs: int = 1) -> List[Dict]:
    """Generate a regression dataset with linear and non-linear relationships"""
    dataset = []
    
    for _ in range(num_samples):
        # Generate random inputs between 0 and 1
        inputs = np.random.rand(num_inputs).tolist()
        
        # Generate targets based on a complex function of inputs
        # Using combination of linear, quadratic, and trigonometric functions
        target_value = 0
        for i, inp in enumerate(inputs):
            target_value += inp * (i + 1) * 0.3  # Linear component
            target_value += np.sin(inp * np.pi) * 0.2  # Non-linear component
        
        # Add some noise
        target_value += np.random.normal(0, 0.05)
        
        # Normalize target to [0, 1]
        target_value = (target_value + 1) / 2
        target_value = np.clip(target_value, 0, 1)
        
        targets = [target_value] * num_outputs
        
        dataset.append({
            "input": [round(x, 4) for x in inputs],
            "target": [round(x, 4) for x in targets]
        })
    
    return dataset

def generate_classification_dataset(num_samples: int = 1000, num_inputs: int = 2, num_classes: int = 3) -> List[Dict]:
    """Generate a classification dataset with multiple classes"""
    dataset = []
    
    for _ in range(num_samples):
        # Generate random inputs
        inputs = np.random.rand(num_inputs).tolist()
        
        # Determine class based on input values
        # Create regions in input space
        sum_inputs = sum(inputs)
        
        if sum_inputs < num_inputs * 0.33:
            class_idx = 0
        elif sum_inputs < num_inputs * 0.66:
            class_idx = 1
        else:
            class_idx = 2 if num_classes > 2 else 1
        
        # One-hot encoding for targets
        targets = [0.0] * num_classes
        targets[class_idx] = 1.0
        
        dataset.append({
            "input": [round(x, 4) for x in inputs],
            "target": targets
        })
    
    return dataset

def generate_xor_dataset(num_samples: int = 400) -> List[Dict]:
    """Generate the classic XOR problem dataset"""
    dataset = []
    
    # Generate samples around the four XOR points
    for _ in range(num_samples):
        # Randomly choose one of the four XOR cases
        case = np.random.randint(0, 4)
        
        if case == 0:  # (0, 0) -> 0
            inputs = [np.random.normal(0.1, 0.1), np.random.normal(0.1, 0.1)]
            target = [0.0]
        elif case == 1:  # (0, 1) -> 1
            inputs = [np.random.normal(0.1, 0.1), np.random.normal(0.9, 0.1)]
            target = [1.0]
        elif case == 2:  # (1, 0) -> 1
            inputs = [np.random.normal(0.9, 0.1), np.random.normal(0.1, 0.1)]
            target = [1.0]
        else:  # (1, 1) -> 0
            inputs = [np.random.normal(0.9, 0.1), np.random.normal(0.9, 0.1)]
            target = [0.0]
        
        # Clip to [0, 1]
        inputs = np.clip(inputs, 0, 1).tolist()
        
        dataset.append({
            "input": [round(x, 4) for x in inputs],
            "target": target
        })
    
    return dataset

def generate_sin_wave_dataset(num_samples: int = 500, num_inputs: int = 1, num_outputs: int = 1) -> List[Dict]:
    """Generate a sine wave prediction dataset"""
    dataset = []
    
    for _ in range(num_samples):
        # Input is x value
        x = np.random.rand() * 2 * np.pi
        inputs = [x / (2 * np.pi)]  # Normalize to [0, 1]
        
        # Target is sin(x), normalized to [0, 1]
        y = (np.sin(x) + 1) / 2
        
        # Add slight noise
        y += np.random.normal(0, 0.02)
        y = np.clip(y, 0, 1)
        
        targets = [y] * num_outputs
        
        dataset.append({
            "input": [round(inputs[0], 4)],
            "target": [round(t, 4) for t in targets]
        })
    
    return dataset

def generate_polynomial_dataset(num_samples: int = 600, degree: int = 2) -> List[Dict]:
    """Generate a polynomial regression dataset"""
    dataset = []
    
    # Random polynomial coefficients
    coeffs = np.random.rand(degree + 1) * 2 - 1  # Range [-1, 1]
    
    for _ in range(num_samples):
        # Generate input
        x = np.random.rand()
        inputs = [x]
        
        # Calculate polynomial
        y = sum(coeffs[i] * (x ** i) for i in range(degree + 1))
        
        # Normalize and add noise
        y = (y + 1) / 2
        y += np.random.normal(0, 0.03)
        y = np.clip(y, 0, 1)
        
        dataset.append({
            "input": [round(x, 4)],
            "target": [round(y, 4)]
        })
    
    return dataset

def generate_multi_output_dataset(num_samples: int = 800, num_inputs: int = 3, num_outputs: int = 2) -> List[Dict]:
    """Generate a multi-output dataset where each output depends on different inputs"""
    dataset = []
    
    for _ in range(num_samples):
        inputs = np.random.rand(num_inputs).tolist()
        
        # Output 1: depends on first half of inputs
        out1 = np.mean(inputs[:num_inputs//2])
        out1 += np.random.normal(0, 0.05)
        out1 = np.clip(out1, 0, 1)
        
        # Output 2: depends on second half of inputs
        out2 = np.mean(inputs[num_inputs//2:])
        out2 += np.random.normal(0, 0.05)
        out2 = np.clip(out2, 0, 1)
        
        targets = [round(out1, 4), round(out2, 4)] if num_outputs >= 2 else [round(out1, 4)]
        
        dataset.append({
            "input": [round(x, 4) for x in inputs],
            "target": targets
        })
    
    return dataset

def save_dataset(dataset: List[Dict], filename: str):
    """Save dataset to JSON file"""
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} samples to {filename}")

def save_dataset_js(dataset: List[Dict], filename: str, variable_name: str):
    """Save dataset as a JavaScript file for direct import"""
    with open(filename, 'w') as f:
        f.write(f"// Auto-generated dataset - {len(dataset)} samples\n")
        f.write(f"export const {variable_name} = ")
        json.dump(dataset, f, indent=2)
        f.write(";\n")
    print(f"Saved {len(dataset)} samples to {filename}")

def get_dataset_description(name: str) -> str:
    """Get description for each dataset type"""
    descriptions = {
        "regression_large": "Large regression dataset with 3 inputs and complex non-linear relationships",
        "regression_small": "Small regression dataset with 2 inputs for quick testing",
        "classification_3class": "3-class classification problem with 2 input features",
        "classification_binary": "Binary classification with 2 input features",
        "xor_problem": "Classic XOR problem - tests network's ability to learn non-linear boundaries",
        "sin_wave": "Sine wave prediction - single input to single output regression",
        "polynomial": "Polynomial regression with degree 3",
        "multi_output": "Multi-output regression with 4 inputs and 2 outputs"
    }
    return descriptions.get(name, "Custom dataset")

if __name__ == "__main__":
    print("Generating datasets for Deeplex Neural Network Trainer...")
    print("-" * 60)
    
    # Create datasets directory
    import os
    os.makedirs("public/datasets", exist_ok=True)
    os.makedirs("src/datasets", exist_ok=True)
    
    # Generate various datasets
    datasets = {
        "regression_large": generate_regression_dataset(2000, 3, 1),
        "regression_small": generate_regression_dataset(500, 2, 1),
        "classification_3class": generate_classification_dataset(1500, 2, 3),
        "classification_binary": generate_classification_dataset(1000, 2, 2),
        "xor_problem": generate_xor_dataset(400),
        "sin_wave": generate_sin_wave_dataset(800, 1, 1),
        "polynomial": generate_polynomial_dataset(700, 3),
        "multi_output": generate_multi_output_dataset(1000, 4, 2),
    }
    
    # Save as JSON files
    for name, data in datasets.items():
        save_dataset(data, f"public/datasets/{name}.json")
    
    # Create a combined dataset file for easy import
    combined = {
        "datasets": {
            name: {
                "data": data,
                "description": get_dataset_description(name),
                "inputs": len(data[0]["input"]) if data else 0,
                "outputs": len(data[0]["target"]) if data else 0,
                "samples": len(data)
            }
            for name, data in datasets.items()
        }
    }
    
    save_dataset(combined, "public/datasets/all_datasets.json")
    
    # Also create a JS version for direct import
    save_dataset_js(datasets["regression_large"], "src/datasets/sample_dataset.js", "sampleDataset")
    
    print("-" * 60)
    print("Dataset generation complete!")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total samples: {sum(len(d) for d in datasets.values())}")
