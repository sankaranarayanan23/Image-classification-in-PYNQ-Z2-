# MicroCNN Hardware Accelerator for Fashion-MNIST

This project demonstrates an end-to-end pipeline for training, compiling, and deploying a highly optimized, low-precision Quantized Convolutional Neural Network (MicroCNN) on a Xilinx PYNQ-Z2 FPGA board. 

The goal is to classify images from the Fashion-MNIST dataset with high accuracy while maintaining an extremely small hardware footprint (target ~15% utilization) using W4A4 (4-bit weight, 4-bit activation) quantization.

## Project Workflow

The project is broken down into four major steps, organized via Jupyter Notebooks:

### 1. Quantization-Aware Training (QAT)
**File**: `MicroCNN_train_cell (3).ipynb`
- Defines a custom 4-Layer (or 6-Layer) Convolutional Neural Network blueprint perfectly sized for the target FPGA.
- Uses **Brevitas** to apply Quantization-Aware Training (QAT). The model is trained directly with low-precision arithmetic (e.g., W4A4) to ensure accuracy doesn't drop when deployed to integer-only hardware.
- Outputs a trained PyTorch model weights file (`best_fashion_mobilenet.pth`).

### 2. FINN-ONNX Export
**File**: `01_export_finn.ipynb`
- Loads the trained PyTorch/Brevitas model and exports it to the **QONNX** format.
- Uses the Xilinx FINN compiler to verify the math against PyTorch.
- Fuses pre-processing (tensor resizing, normalization) and post-processing (Top-1 classification) nodes directly into the graph.
- Generates a tidy, hardware-ready ONNX model (`fashion_cnn_finn_tidy.onnx`).

### 3. Hardware Compilation (HLS & Vivado Synthesis)
**File**: `02_hardware_build_fixed.ipynb`
- Streamlines the computational graph (e.g., absorbing Batch Normalization into MultiThresholds).
- Translates the ONNX nodes into FINN High-Level Synthesis (HLS) hardware layers.
- Partitions the dataflow, calculates exact hardware dimensions (PE, SIMD), and folds the architecture to fit within the BRAM and LUT budget of the PYNQ-Z2 board.
- Triggers **Vivado** to synthesize the bitstream (`resizer.bit`) and hardware handoff files (`resizer.hwh`).

### 4. PYNQ-Z2 Deployment
**Location**: `deploy_pynq/03_pynq_deploy.ipynb`
- The final inference environment running directly on the ARM CPU of the PYNQ-Z2 board.
- Uses the `pynq` Python library to load the FPGA overlay.
- Features tools for single-image classification (via file or URL) and **real-time webcam classification** using OpenCV.
- Includes a benchmarking suite to calculate the exact hardware latency and FPS throughput.

---

## Technology Stack

**Machine Learning & Data**
- **PyTorch & Torchvision**: For dataset management, model definition, and basic training loops.
- **Brevitas**: A PyTorch library developed by AMD/Xilinx for Quantization-Aware Training.

**Compiler & Synthesis**
- **QONNX**: Represents the quantized neural network graph.
- **FINN Compiler**: An experimental framework from Xilinx Research Labs to map QNNs to FPGA hardware.
- **Vivado HLS**: Synthesizes the FINN hardware layers into an FPGA bitstream.

**Edge Deployment**
- **PYNQ**: Python productivity for Zynq. Allows interacting with the FPGA logic directly from Jupyter Notebooks running on the board.
- **OpenCV**: Used for real-time video capture and frame processing in the deployment environment.
- **Matplotlib & NumPy**: For tensor manipulation and visualization.
