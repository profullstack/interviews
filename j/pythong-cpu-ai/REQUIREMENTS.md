# 🧾 Product Requirements Document (PRD)

## Title:
Train ResNet-50 on CPU with Separated Core Allocation for Forward and Backward Passes

---

## Objective:
Implement a Python-based machine learning training routine that uses all available CPU cores to train a ResNet-50 model, while **separating CPU core usage between forward and backward propagation** to simulate device separation within CPU-only environments.

---

## Key Requirements

### 1. Detect CPU Resources
- Detect the total number of logical CPU cores using Python.
- Split the CPU cores evenly into two groups:
  - **Forward Group**: Handles forward propagation.
  - **Backward Group**: Handles backward propagation (gradient computation and weight updates).

### 2. Model & Framework
- Use the **ResNet-50** model architecture.
- Must be implemented using **PyTorch** (preferred) or TensorFlow if needed.
- Ensure training runs **only on CPU** (no GPU or other devices).
- Use `.to('cpu')` for all model and tensor allocations.

### 3. CPU Core Affinity
- Forward pass computations should run **only on the Forward Group** of CPU cores.
- Backward pass (loss computation, backpropagation, optimizer step) should run **only on the Backward Group** of CPU cores.
- Enforce core usage via OS-level CPU affinity controls:
  - Python modules: `os.sched_setaffinity`, `psutil`, or multiprocessing/thread affinity.

### 4. Custom Training Loop
- Do **not** use high-level APIs like `model.fit()`.
- Write a fully custom training loop with:
  - Separate functions or contexts for forward and backward passes.
  - Explicit enforcement of core allocation for each.

### 5. Data
- Use synthetic/dummy image data to validate functionality (no need for real datasets initially).
- Input shape should be `(batch_size, 3, 224, 224)` to match ImageNet expectations.

---

## Stretch Goals (Optional)
- Log which cores are active during each phase for transparency.
- Measure and report timing or CPU utilization stats per phase.
- Handle odd-numbered CPU core counts gracefully.

---

## Deliverables
- A standalone Python script or Jupyter notebook that:
  - Loads and trains ResNet-50 on CPU.
  - Demonstrates CPU-core-separated training loop.
- Logging output showing:
  - Total detected cores.
  - Core allocation per phase.
- A `README.md` with:
  - Setup instructions.
  - Explanation of how the CPU core splitting works.

---

## Notes
- This is a CPU-only simulation of device separation in training pipelines.
- It emphasizes low-level control over model execution and resource allocation.
