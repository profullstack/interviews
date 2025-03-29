#!/usr/bin/env python3

"""
ResNet-50 Training with Split CPU Core Allocation

This script implements a custom training loop for ResNet-50 on CPU only,
with CPU cores divided between forward and backward propagation.
This implementation explicitly disables CUDA to ensure CPU-only operation.
"""

import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50
import logging

# Force CPU-only mode by setting CUDA_VISIBLE_DEVICES to empty
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Double-check that CUDA is not available
if torch.cuda.is_available():
    raise RuntimeError("CUDA should not be available in this CPU-only implementation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CPUAffinityManager:
    """
    Manages CPU core affinity for different parts of the training process.
    Leaves at least 2 cores free if more than 4 cores are available.
    """
    def __init__(self):
        # Get the process
        self.process = psutil.Process(os.getpid())
        
        # Get all available CPU cores
        self.all_cores = list(range(psutil.cpu_count(logical=True)))
        self.num_cores = len(self.all_cores)
        
        # Reserve cores for system if we have more than 4 cores
        reserved_cores = 0
        if self.num_cores > 4:
            reserved_cores = 2
            logger.info(f"Reserving {reserved_cores} cores for system operations")
        
        # Get cores available for training
        self.available_cores = self.all_cores[:-reserved_cores] if reserved_cores > 0 else self.all_cores
        self.available_count = len(self.available_cores)
        
        # Split cores into forward and backward groups
        self.forward_cores = self.available_cores[:self.available_count//2]
        self.backward_cores = self.available_cores[self.available_count//2:]
        
        # Handle odd number of cores
        if len(self.forward_cores) < len(self.backward_cores):
            self.forward_cores.append(self.backward_cores.pop())
        
        logger.info(f"Total CPU cores: {self.num_cores}")
        logger.info(f"Cores available for training: {self.available_count}")
        logger.info(f"Forward pass cores: {self.forward_cores}")
        logger.info(f"Backward pass cores: {self.backward_cores}")
        if reserved_cores > 0:
            logger.info(f"Reserved system cores: {self.all_cores[-reserved_cores:]}")
        
        # Save original affinity to restore later
        self.original_affinity = self.process.cpu_affinity()
    
    def set_forward_affinity(self):
        """Set CPU affinity for forward pass"""
        self.process.cpu_affinity(self.forward_cores)
        logger.info(f"Set affinity to forward cores: {self.process.cpu_affinity()}")
    
    def set_backward_affinity(self):
        """Set CPU affinity for backward pass"""
        self.process.cpu_affinity(self.backward_cores)
        logger.info(f"Set affinity to backward cores: {self.process.cpu_affinity()}")
    
    def restore_original_affinity(self):
        """Restore original CPU affinity"""
        self.process.cpu_affinity(self.original_affinity)
        logger.info(f"Restored original affinity: {self.process.cpu_affinity()}")


def create_dummy_data(batch_size=32, num_batches=10):
    """
    Create dummy image data for training.
    
    Args:
        batch_size: Number of images per batch
        num_batches: Number of batches to create
        
    Returns:
        DataLoader with dummy data
    """
    # Create dummy inputs (batch_size, channels, height, width)
    inputs = torch.randn(batch_size * num_batches, 3, 224, 224)
    
    # Create dummy labels (batch_size)
    labels = torch.randint(0, 1000, (batch_size * num_batches,))
    
    # Create dataset and dataloader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def train_resnet50_split_cores(num_epochs=2, batch_size=32, learning_rate=0.001):
    """
    Train ResNet-50 with CPU cores split between forward and backward passes.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    # Initialize CPU affinity manager
    affinity_manager = CPUAffinityManager()
    
    # Explicitly set device to CPU and verify
    device = torch.device('cpu')
    logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # Initialize ResNet-50 model
    logger.info("Initializing ResNet-50 model...")
    model = resnet50(pretrained=False).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Create dummy data
    logger.info("Creating dummy data...")
    dataloader = create_dummy_data(batch_size=batch_size)
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with forward cores
            affinity_manager.set_forward_affinity()
            forward_start = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            forward_time = time.time() - forward_start
            
            # Backward pass with backward cores
            affinity_manager.set_backward_affinity()
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start
            
            # Update statistics
            running_loss += loss.item()
            
            # Log batch progress
            if i % 5 == 0:  # Log every 5 batches
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(dataloader)}, "
                           f"Loss: {loss.item():.4f}, "
                           f"Forward time: {forward_time:.4f}s, "
                           f"Backward time: {backward_time:.4f}s")
        
        # Log epoch statistics
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
                   f"Average loss: {running_loss/len(dataloader):.4f}")
    
    # Restore original CPU affinity
    affinity_manager.restore_original_affinity()
    logger.info("Training completed.")


def benchmark_comparison(num_epochs=1, batch_size=32):
    """
    Benchmark performance with and without core separation.
    
    Args:
        num_epochs: Number of training epochs for benchmark
        batch_size: Batch size for training
    """
    logger.info("\n===== BENCHMARKING =====\n")
    
    # Create dummy data
    dataloader = create_dummy_data(batch_size=batch_size, num_batches=5)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cpu')
    logger.info(f"Benchmark using device: {device} (CUDA available: {torch.cuda.is_available()})")
    model = resnet50(pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Initialize CPU affinity manager
    affinity_manager = CPUAffinityManager()
    
    # Benchmark with split cores
    logger.info("Benchmarking with split cores...")
    split_start = time.time()
    
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with forward cores
            affinity_manager.set_forward_affinity()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass with backward cores
            affinity_manager.set_backward_affinity()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    split_time = time.time() - split_start
    logger.info(f"Split cores time: {split_time:.2f}s")
    
    # Reset model
    model = resnet50(pretrained=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Benchmark with all cores
    logger.info("Benchmarking with all cores...")
    affinity_manager.restore_original_affinity()
    all_start = time.time()
    
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with all cores
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass with all cores
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    all_time = time.time() - all_start
    logger.info(f"All cores time: {all_time:.2f}s")
    
    # Compare results
    speedup = (all_time / split_time - 1) * 100
    logger.info(f"Performance difference: {speedup:.2f}% ({'faster' if speedup > 0 else 'slower'} with all cores)")


if __name__ == "__main__":
    logger.info("Starting ResNet-50 training with split CPU cores")
    
    # Train with split cores
    train_resnet50_split_cores(num_epochs=1, batch_size=16)
    
    # Optional: Run benchmark comparison
    benchmark_comparison(num_epochs=1, batch_size=16)
