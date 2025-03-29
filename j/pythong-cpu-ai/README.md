# ResNet-50 Training with Split CPU Core Allocation

This project implements a custom training routine for ResNet-50 that divides CPU cores between forward and backward propagation phases. It simulates device separation in a CPU-only environment by controlling CPU core affinity for different parts of the training process.

## Features

- Automatically detects and divides available CPU cores into forward and backward groups
- Implements custom training loop with explicit control over which cores handle each phase
- Uses PyTorch and ResNet-50 architecture with CPU-only processing
- Includes benchmarking to compare performance with and without core separation
- Detailed logging of core allocation and training metrics

## Requirements

```
python >= 3.6
torch >= 1.7.0
torchvision >= 0.8.0
psutil >= 5.8.0
numpy >= 1.19.0
```

## Installation

1. Clone this repository
2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

> **Note:** The `venv` directory is included in `.gitignore` and should not be committed to version control. Each user should create their own virtual environment.

## Usage

Run the training script:

```bash
python train_resnet50_split_cores.py
```

## How It Works

### Core Separation

The script uses the `psutil` library to detect available CPU cores and divides them into two groups:

1. **Forward Group**: Handles the forward pass (model inference)
2. **Backward Group**: Handles the backward pass (gradient computation and weight updates)

To prevent system slowdown, the script reserves 2 cores for system operations if more than 4 cores are available.

For example, on an 8-core machine:
- Cores 0-2 would be assigned to forward propagation
- Cores 3-5 would be assigned to backward propagation
- Cores 6-7 would be reserved for system operations

The script uses CPU affinity settings to control which cores are active during each phase of training.

### Training Process

1. For each batch of data:
   - Set CPU affinity to forward cores
   - Perform forward pass through the model
   - Set CPU affinity to backward cores
   - Compute loss, gradients, and update weights

2. The script logs:
   - Which cores are used for each phase
   - Time spent in forward vs. backward passes
   - Training metrics (loss, accuracy)

### Benchmarking

The script includes an optional benchmark that compares:
- Training with split core allocation
- Training with all cores available for both passes

This helps evaluate the performance impact of core separation.

## Customization

You can modify the following parameters in the script:

- `num_epochs`: Number of training epochs
- `batch_size`: Number of samples per batch
- `learning_rate`: Learning rate for the optimizer

## Notes

- This implementation is primarily for educational purposes to explore resource management in deep learning training
- Performance may vary depending on your CPU architecture and the specific workload
- For real-world training, GPU acceleration would typically be preferred
