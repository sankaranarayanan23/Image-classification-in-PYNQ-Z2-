# Fashion-MNIST Image Classification on PYNQ-Z2 using FINN

> A complete end-to-end pipeline for training a Quantization-Aware, W2A2 CNN in PyTorch/Brevitas, compiling it through the FINN FPGA dataflow compiler, and deploying it as a hardware accelerator on the Xilinx PYNQ-Z2 board.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [What is FINN?](#3-what-is-finn)
4. [System Requirements](#4-system-requirements)
5. [FINN Installation](#5-finn-installation)
6. [The Full Pipeline — Step by Step](#6-the-full-pipeline--step-by-step)
   - [Step 1 — Train the Model (Colab)](#step-1--train-the-model-colab)
   - [Step 2 — Export to FINN-ONNX](#step-2--export-to-finn-onnx)
   - [Step 3 — Hardware Build (Streamline → Synthesize)](#step-3--hardware-build-streamline--synthesize)
   - [Step 4 — Deploy on PYNQ-Z2](#step-4--deploy-on-pynq-z2)
7. [Model Architecture](#7-model-architecture)
8. [ONNX Artefact Glossary](#8-onnx-artefact-glossary)
9. [Deployment Package Contents](#9-deployment-package-contents)
10. [FINN Compiler Concepts Explained](#10-finn-compiler-concepts-explained)
11. [Troubleshooting](#11-troubleshooting)
12. [References](#12-references)

---

## 1. Project Overview

This project demonstrates how to take a custom quantized neural network from scratch all the way to a fully hardware-accelerated bitfile running on an FPGA. The target task is **Fashion-MNIST** — a 10-class grayscale image classification benchmark (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

The key insight of this project is that **neural network inference does not have to run on a CPU or GPU**. By quantizing weights and activations to just 2 bits, the entire CNN can be compiled into a custom, pipelined digital circuit — a **dataflow accelerator** — that runs directly on the reconfigurable logic fabric of a Zynq FPGA. The result is an inference engine that consumes far less power and has deterministic, ultra-low latency.

The complete pipeline has four stages:

```
[Google Colab]              [FINN Docker on Host PC]              [PYNQ-Z2 Board]
  Train QNN         ──►    Export → Compile → Synthesize   ──►    Deploy & Infer
  (Brevitas/PyTorch)        (QONNX → FINN-ONNX → .bit)             (Python driver)
```

> **Note:** The final PYNQ-Z2 deployment notebook (`03_pynq_deploy.ipynb`) will be added in a future update. The bitfile and driver package are already complete and included.

---

## 2. Repository Structure

```
.
├── MicroCNN_train_cell__3_.ipynb     # Step 1 — QAT training in Google Colab
├── 01_export_finn.ipynb              # Step 2 — Export PyTorch model → FINN-ONNX
├── 02_hardware_build_fixed.ipynb     # Step 3 — Full FINN hardware compilation pipeline
│
├── best_fashion_mobilenet.pth        # Trained model weights (saved from Step 1)
│
├── fashion_cnn_qonnx.onnx            # Raw QONNX export from Brevitas
├── fashion_cnn_qonnx_clean.onnx      # Cleaned-up QONNX (after qonnx_cleanup)
├── fashion_cnn_finn_tidy.onnx        # After ConvertQONNXtoFINN + TopK + tidy
├── fashion_cnn_streamlined.onnx      # After algebraic streamlining passes
├── fashion_cnn_dataflow_parent.onnx  # Parent model with StreamingDataflowPartition node
├── fashion_cnn_dataflow_model.onnx   # Inner dataflow graph (FPGA portion)
├── fashion_cnn_folded.onnx           # Dataflow model with PE/SIMD folding applied
├── fashion_cnn_hlssynth.onnx         # After HLS IP synthesis
├── fashion_cnn_zynq.onnx             # Final model after full Vivado Zynq build
├── fashion_mobilenet_hlssynth.onnx   # MobileNet variant HLS model (experimental)
├── preproc.onnx                      # Preprocessing subgraph (ToTensor / UINT8)
│
├── deploy.zip                        # Deployment package (bitfile + HWH + driver)
└── deploy_pynq.zip                   # Extended package with test images + inference notebook
```

---

## 3. What is FINN?

**FINN** (Fast, Scalable Quantized Neural Network Inference on FPGAs) is a research framework developed by AMD/Xilinx that compiles quantized neural networks into **fully custom FPGA dataflow accelerators**. Unlike GPU or CPU inference, a FINN-generated accelerator has:

- **No general-purpose processor** — every layer is a dedicated hardware unit.
- **Streaming dataflow** — data flows continuously through the pipeline, layer by layer, in parallel.
- **Deterministic latency** — no cache misses, no scheduling jitter.
- **Extremely low power** — the Zynq PL (programmable logic) is orders of magnitude more efficient per inference than a GPU for small quantized networks.

FINN works by taking a quantized ONNX model and applying a sequence of **graph transformations** that progressively lower the representation from a floating-point-friendly format down to RTL-level hardware primitives. These primitives are then stitched together and passed to Vivado/HLS for final synthesis into a bitfile.

The core transformation pipeline inside FINN is:

```
QONNX (Brevitas export)
    │
    ▼  ConvertQONNXtoFINN
FINN-ONNX (BipolarQuant, MultiThreshold)
    │
    ▼  Streamline (absorb BN/scaling into thresholds)
Streamlined ONNX (only threshold comparisons remain)
    │
    ▼  LowerConvsToMatMul, InferHWLayers
Dataflow ONNX (MVAU_hls, ConvolutionInputGenerator, StreamingMaxPool)
    │
    ▼  Folding (set PE/SIMD parallelism)
Folded ONNX
    │
    ▼  PrepareIP + HLSSynthIP (Vitis HLS)
HLS-synthesized ONNX (with timing/resource estimates)
    │
    ▼  InsertDWC + InsertFIFO + ZynqBuild (Vivado)
Bitfile (.bit) + Hardware Handoff (.hwh) + Python Driver
```

---

## 4. System Requirements

These requirements apply to the **host PC** running the FINN compilation pipeline (not the PYNQ board itself).

### Operating System
- **Ubuntu 18.04** or later with `bash` installed.
- Other Linux distributions may work but are not officially supported.
- Windows/macOS are **not supported** for FINN (you would need a Linux VM or WSL2).

### Software Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker | Any recent version | Must be configured to run **without root** |
| Vivado / Vitis | **2022.2** | Required for HLS synthesis and bitfile generation |
| Git | Any | For cloning the FINN repo |

> FINN runs entirely inside a Docker container. Docker must be set up so your user can run it without `sudo`. See the [Docker post-install guide](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

### Hardware Requirements (Host PC)

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB (for Zynq targets) |
| CPU Cores | 4 | 8+ (FINN parallelizes HLS across cores) |
| Storage | 50 GB free | 100 GB+ on a fast SSD |
| Internet | Required | For pulling Docker images |

> Vivado synthesis for the Pynq-Z2 (xc7z020) requires at minimum 8 GB of RAM. The HLS and implementation steps generate tens of GB of temporary files, controlled by `FINN_HOST_BUILD_DIR` (defaults to `/tmp/finn_dev_<username>`).

### PYNQ-Z2 Board

| Requirement | Details |
|-------------|---------|
| PYNQ image | v2.6 or later, flashed to SD card |
| Network | Board must be on the same network as the host PC |
| SSH access | Public key authentication set up between host and board |
| Python package | `bitstring` must be installed on the board: `sudo pip3 install bitstring` |

---

## 5. FINN Installation

### Step-by-step

**1. Install Vivado 2022.2**

Download and install Vivado 2022.2 from the AMD/Xilinx website. The WebPACK (free) edition is sufficient for the Pynq-Z2's xc7z020 part. After installation:

```bash
export FINN_XILINX_PATH=/opt/Xilinx   # adjust to your install path
export FINN_XILINX_VERSION=2022.2
```

Add these to your `~/.bashrc` to make them persistent.

**2. Install Docker (without root)**

```bash
# Install Docker (Ubuntu)
sudo apt-get update
sudo apt-get install docker.io

# Add your user to the docker group
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker   # activate without logout

# Verify it works without sudo
docker run hello-world
```

**3. Clone the FINN repository**

```bash
git clone https://github.com/Xilinx/finn/
cd finn
```

**4. Test the installation**

```bash
bash ./run-docker.sh quicktest
```

This builds the Docker image and runs a quick sanity check. The first run will take several minutes as Docker layers are downloaded.

**5. Launch the Jupyter notebook server**

```bash
bash ./run-docker.sh notebook
```

Open the URL printed in the terminal (e.g., `http://127.0.0.1:8888/?token=...`) in your browser. You can now create notebooks inside the FINN environment and access the full FINN API.

**6. Set up PYNQ board SSH keys** (for deployment)

```bash
# Inside the FINN Docker container:
cd /path/to/finn/ssh_keys
ssh-keygen    # generates id_rsa and id_rsa.pub
ssh-copy-id -i id_rsa.pub xilinx@<PYNQ_IP>

# Verify passwordless login:
ssh xilinx@<PYNQ_IP>
```

### Key Environment Variables

| Variable | Required | Example | Purpose |
|----------|----------|---------|---------|
| `FINN_XILINX_PATH` | ✅ | `/opt/Xilinx` | Path to Xilinx tools on host |
| `FINN_XILINX_VERSION` | ✅ | `2022.2` | Vivado/Vitis version |
| `FINN_HOST_BUILD_DIR` | Optional | `/data/finn_build` | Where temp build files go |
| `NUM_DEFAULT_WORKERS` | Optional | `4` | HLS parallelism (cores) |
| `JUPYTER_PORT` | Optional | `8888` | Jupyter port inside Docker |
| `PYNQ_BOARD` | Optional | `Pynq-Z2` | Target board for test suite |

---

## 6. The Full Pipeline — Step by Step

### Step 1 — Train the Model (Colab)

**Notebook:** `MicroCNN_train_cell__3_.ipynb`  
**Environment:** Google Colab (T4 GPU recommended)

This notebook trains the `Fashion_Fit_CNN` — a 4-layer Quantization-Aware CNN — on the FashionMNIST dataset.

**Key training choices:**

- **Input size:** 32×32 (FashionMNIST is 28×28, resized up to a power-of-2 for FINN compatibility).
- **Quantization:** Weights at 8-bit for Layer 1 and final FC; 2-bit weights and activations for the inner conv layers. This is the W2A2 scheme — each multiply-accumulate becomes a 2-bit popcount, massively reducing hardware cost.
- **Framework:** Brevitas — a PyTorch extension for QAT that tracks quantization state and is directly compatible with the FINN ONNX exporter.
- **Optimizer:** Adam with CosineAnnealingLR scheduler over 50 epochs.

The best checkpoint is saved as `best_fashion_mobilenet.pth`.

**Architecture defined in training:**

```python
class Fashion_Fit_CNN(nn.Module):
    features: [
        QuantConv2d(1 → 64,  W8) → BN → QuantReLU(A8) → MaxPool,
        QuantConv2d(64 → 64,  W2) → BN → QuantReLU(A2) → MaxPool,
        QuantConv2d(64 → 128, W2) → BN → QuantReLU(A2) → MaxPool,
        QuantConv2d(128 → 128,W2) → BN → QuantReLU(A2) → MaxPool,
    ]
    classifier: [QuantLinear(512 → 10, W8)]
```

---

### Step 2 — Export to FINN-ONNX

**Notebook:** `01_export_finn.ipynb`  
**Environment:** FINN Docker (Jupyter)

This notebook takes the `.pth` checkpoint and produces the FINN-compatible ONNX graph.

**Cell 1 — Load & Inspect Weights**

Loads `best_fashion_mobilenet.pth` and prints the state-dict to verify the architecture matches.

**Cell 2 — Export QONNX**

Re-instantiates `Fashion_Fit_CNN` with the same architecture, loads the weights, and calls Brevitas's `export_qonnx()` with a dummy 32×32 input. This produces `fashion_cnn_qonnx.onnx`. Then `qonnx_cleanup()` is run to canonicalize the graph into `fashion_cnn_qonnx_clean.onnx`.

**Cell 3 — Verify QONNX**

Runs both the original PyTorch model and the ONNX model on the same dummy input and checks `np.allclose()`. This confirms the export was lossless.

**Cell 4 — Merge Preprocessor + Convert to FINN-ONNX**

Several things happen here:

1. A `ToTensor` preprocessor (normalizes UINT8 → float) is exported as a separate subgraph (`preproc.onnx`) and merged in front of the main model. This means the hardware will accept raw UINT8 pixel bytes directly.
2. `ConvertQONNXtoFINN()` replaces Brevitas quantization annotations with FINN's internal `MultiThreshold` and packed-weight representations.
3. `InsertTopK(k=1)` adds a Top-1 argmax at the output so the accelerator returns a class index directly.
4. A series of tidy-up passes (InferShapes, FoldConstants, GiveReadableTensorNames, etc.) produce the final `fashion_cnn_finn_tidy.onnx`.

**Cell 5 — Verify Files**

Confirms all output files exist and have non-zero sizes.

---

### Step 3 — Hardware Build (Streamline → Synthesize)

**Notebook:** `02_hardware_build_fixed.ipynb`  
**Environment:** FINN Docker (Jupyter)  
**Duration:** ~30–60 minutes (HLS) + ~30–60 minutes (Vivado)

This is the heart of the FINN compilation pipeline. It transforms the neural network ONNX graph into an actual FPGA bitfile.

**Cell 1 — Setup**

Configures paths and sets the target board (`Pynq-Z2`, FPGA part `xc7z020clg400-1`) and clock period (`CLK_NS = 10.0` → 100 MHz, or `20.0` → 50 MHz for timing slack).

**Cell 3 — Streamlining**

This is a purely algebraic step — no hardware is generated yet.

FINN's streamlining passes absorb all the BatchNorm (scale + shift) parameters and floating-point scaling factors from QAT into the integer threshold values of the `MultiThreshold` nodes. After streamlining, the graph contains **no floating-point operations** — every layer is either a matrix-vector multiply with quantized weights, or a threshold comparison.

Key transformations applied:
- `MoveScalarLinearPastInvariants` — pulls BN scaling through MaxPool
- `Streamline` — absorbs BN mul/add into preceding/following thresholds
- `LowerConvsToMatMul` — converts convolutions to generalized matrix multiplies (the FINN MVAU abstraction)
- `AbsorbTransposeIntoMultiThreshold` — merges NHWC/NCHW layout changes into threshold nodes

Result: `fashion_cnn_streamlined.onnx`

**Cell 4 — HW Layer Conversion**

Converts the abstract streamlined operators into FINN's FPGA-specific custom ops:

| ONNX Op | FINN HW Op | Purpose |
|---------|-----------|---------|
| `MatMul` + `MultiThreshold` | `MVAU_hls` | Matrix-Vector Activation Unit |
| `Im2Col` (sliding window) | `ConvolutionInputGenerator_rtl` | Generates input patches for convolutions |
| `MaxPool` | `StreamingMaxPool_hls` | Streaming max pooling |
| `TopK` | `LabelSelect_hls` | Output class selection |

Then `CreateDataflowPartition()` splits the graph into:
- **Parent model** (`fashion_cnn_dataflow_parent.onnx`): The outer graph, with CPU-side pre/post processing and a single `StreamingDataflowPartition` node pointing to the FPGA portion.
- **Dataflow model** (`fashion_cnn_dataflow_model.onnx`): The inner graph containing only FPGA hardware nodes.

**Cell 6 — Folding (PE/SIMD Configuration)**

This step sets the degree of parallelism for each hardware layer. It controls the **hardware-software resource trade-off**:

- **SIMD** (Single Instruction Multiple Data): How many input channels are processed in parallel per cycle. Higher SIMD = more throughput, more LUTs.
- **PE** (Processing Elements): How many output neurons are computed in parallel per cycle. Higher PE = more throughput, more DSPs/LUTs.

For the Pynq-Z2's constrained resources, safe conservative values are used (PE=1, SIMD scaled to keep weight arrays under 1024 elements), with weights stored in BRAM.

Result: `fashion_cnn_folded.onnx`

**Cell 7 — Resource Estimates**

Reads the folded model and checks that `MH % PE == 0` for every MVAU layer. This is a prerequisite for valid hardware generation.

**Cells 8–10 — HLS Synthesis + Vivado Build + Driver Generation**

Three major phases run in sequence:

1. **`PrepareIP` + `HLSSynthIP`** (~15–20 min): Generates Vitis HLS C++ code for each layer and runs HLS synthesis to produce verified RTL IP cores with timing/resource estimates.

2. **`InsertDWC` + `InsertFIFO`** : Inserts Data Width Converters between layers whose stream widths don't match, and inserts shallow FIFOs at every inter-layer boundary to absorb pipeline bubbles and prevent deadlock.

3. **`ZynqBuild`** (~30–60 min): Stitches all IP cores together in Vivado IP Integrator, adds a DMA engine, runs full Vivado synthesis + place-and-route, and generates the final bitfile. Also generates the `.hwh` hardware handoff file that PYNQ uses to configure the PS-PL interface at runtime.

4. **`MakePYNQDriver`**: Auto-generates a Python driver (`driver.py`) that wraps the bitfile with a numpy-compatible interface — you call `accel.execute(input_img)` and get back a class prediction.

All outputs are bundled into `deploy.zip` and `deploy_pynq.zip`.

---

### Step 4 — Deploy on PYNQ-Z2

**Notebook:** `03_pynq_deploy.ipynb` (included in `deploy_pynq.zip`)  
**Environment:** Jupyter on the PYNQ-Z2 board itself  
**Status:** 🚧 Full deployment notebook coming soon

The deployment package (`deploy_pynq.zip`) contains everything needed to run inference on the board. You copy this zip to the board via SCP or the PYNQ file manager, unzip it, and run the notebook.

**What the deployment package includes:**

```
deploy_pynq/
├── resizer.bit              # FPGA bitfile — the compiled neural network
├── resizer.hwh              # Hardware handoff — describes AXI interfaces to PYNQ
├── driver/
│   ├── driver.py            # Top-level inference wrapper
│   ├── driver_base.py       # Base class with DMA transfer logic
│   ├── validate.py          # Batch validation against FashionMNIST test set
│   └── finn/, qonnx/        # Minimal runtime utilities (no full FINN needed on board)
├── 03_pynq_deploy.ipynb     # Inference demo notebook
├── tshirt.png               # Test image: T-shirt
├── sneaker.png              # Test image: Sneaker
├── bag_1.png, bag_2.png     # Test images: Bags
├── ankle.png                # Test image: Ankle boot
├── trouser.png, jean.png    # Test images: Trousers
```

**Typical usage on the board:**

```python
from driver import FINNExampleOverlay

# Load the bitfile onto the FPGA
accel = FINNExampleOverlay("resizer.bit", platform="zynq-iodma",
                            io_shape_dict=io_shape_dict)

# Run inference on a single image (32x32 grayscale, UINT8)
import numpy as np
from PIL import Image
img = np.array(Image.open("tshirt.png").convert("L").resize((32,32)))
result = accel.execute(img.reshape(1,1,32,32).astype(np.uint8))
print("Predicted class:", result)  # → 0 (T-shirt/top)
```

**What happens at inference time:**

1. The Python driver packs the input image into the correct byte layout and DMA-transfers it from the ARM CPU's memory into the FPGA's AXI stream input port.
2. The bitstream — the compiled neural network circuit — processes the data through its pipeline in a handful of clock cycles.
3. The Top-1 class index is DMA-transferred back from the FPGA to the CPU and returned as a numpy array.

The entire flow from `accel.execute()` to result is microseconds, with no floating-point math on the CPU.

---

## 7. Model Architecture

The `Fashion_Fit_CNN` is a 4-layer quantized CNN designed specifically to fit within the Pynq-Z2's resource budget.

```
Input: 32×32×1 (UINT8)
  │
  ├─ Conv2d(1→64, 3×3, W8A8) → BN → MaxPool(2) → 16×16×64
  ├─ Conv2d(64→64, 3×3, W2A2) → BN → MaxPool(2) → 8×8×64
  ├─ Conv2d(64→128, 3×3, W2A2) → BN → MaxPool(2) → 4×4×128
  ├─ Conv2d(128→128, 3×3, W2A2) → BN → MaxPool(2) → 2×2×128
  │
  Flatten → 512
  │
  └─ Linear(512→10, W8)
      │
  Output: class index (0–9)
```

**Why this architecture for FPGA:**

- **4× MaxPool layers** reduce the spatial dimensions aggressively. This minimizes the size of the weight matrices and the line buffers in the ConvolutionInputGenerators, keeping BRAM usage within the Pynq-Z2's budget.
- **W2A2 inner layers** mean all multiply-accumulates are 2-bit × 2-bit popcount operations, which map to LUTs rather than DSP slices. This is the primary reason the design fits on the small Pynq-Z2.
- **W8 first and last layers** (the "stem" and "head") use slightly higher precision to avoid accuracy collapse at the input/output boundaries — a well-established QAT heuristic.
- **512-element FC layer** (not 2048) is a deliberate design choice: the extra MaxPool at the end shrinks the spatial map from 4×4 to 2×2 before flattening, making the classifier tiny and trivially synthesizable.

---

## 8. ONNX Artefact Glossary

Each intermediate ONNX file represents a specific stage in the compilation process. You can visualize any of them with [Netron](https://netron.app/).

| File | Stage | Description |
|------|-------|-------------|
| `fashion_cnn_qonnx.onnx` | Raw Brevitas export | Contains `Quant`, `BinaryQuant` nodes; floating-point aware |
| `fashion_cnn_qonnx_clean.onnx` | After `qonnx_cleanup` | Canonicalized shapes, names, unused nodes removed |
| `fashion_cnn_finn_tidy.onnx` | After `ConvertQONNXtoFINN` | Quant nodes replaced with `MultiThreshold`; TopK added; UINT8 input |
| `fashion_cnn_streamlined.onnx` | After `Streamline` | BN/scaling fully absorbed; no floats remain; layout-normalized |
| `fashion_cnn_dataflow_parent.onnx` | After `CreateDataflowPartition` | Outer graph; contains `StreamingDataflowPartition` node |
| `fashion_cnn_dataflow_model.onnx` | Inner dataflow graph | Only `MVAU_hls`, `ConvolutionInputGenerator_rtl`, `StreamingMaxPool_hls` |
| `fashion_cnn_folded.onnx` | After PE/SIMD assignment | Parallelism parameters set on each HW node |
| `fashion_cnn_hlssynth.onnx` | After HLS synthesis | RTL generated; timing/resource metadata populated |
| `fashion_cnn_zynq.onnx` | After Vivado build | Points to final `.bit` and `.hwh` files |
| `preproc.onnx` | Preprocessing subgraph | UINT8 → float, merged into main model |

---

## 9. Deployment Package Contents

### `deploy.zip`

```
deploy.zip
├── resizer.bit          # 4 MB FPGA bitfile (Zynq xc7z020)
├── resizer.hwh          # 280 KB hardware handoff (PS/PL interface description)
└── driver/
    ├── driver.py        # FINNExampleOverlay — main inference API
    ├── driver_base.py   # DMA transfer, packing, unpacking logic
    ├── validate.py      # Batch accuracy evaluation on FashionMNIST test set
    ├── finn/util/
    │   └── data_packing.py    # Bit-packing utilities for quantized tensors
    └── qonnx/
        ├── core/datatype.py   # DataType enum (UINT8, INT2, etc.)
        └── util/basic.py      # Utility functions
```

### `deploy_pynq.zip`

Superset of `deploy.zip`, additionally containing:
- `03_pynq_deploy.ipynb` — the on-board inference demo notebook
- Test images: `tshirt.png`, `sneaker.png`, `bag_1.png`, `bag_2.png`, `ankle.png`, `trouser.png`, `jean.png`

---

## 10. FINN Compiler Concepts Explained

### Quantization (W2A2)
"W2A2" means 2-bit weights and 2-bit activations. Each weight value is one of {-1, 0, 1} (or a scaled variant), and each activation is one of 4 integer levels. During inference, a multiply-accumulate becomes a tiny integer addition — on an FPGA, this is implemented as a LUT-based popcount, not a DSP multiply.

### MultiThreshold
FINN's core inference primitive. After streamlining, every activation function (ReLU + quantization) becomes a comparison against a set of pre-computed integer thresholds. A MultiThreshold node outputs the number of thresholds the input exceeds. This is a trivially synthesizable comparator tree.

### MVAU (Matrix-Vector Activation Unit)
The hardware block that implements a quantized fully-connected or convolutional layer. It multiplies a weight matrix by an input vector using popcount arithmetic and then applies the MultiThreshold. The key configuration parameters are **PE** (parallel output neurons) and **SIMD** (parallel input channels), which trade off throughput for area.

### ConvolutionInputGenerator (SWG — Sliding Window Generator)
Convolutions are lowered to matrix-vector operations, but this requires turning the 2D feature map into overlapping patches. The SWG generates these patches in the correct order for the MVAU, using a small line buffer (stored in BRAM or distributed RAM). This is often the most BRAM-intensive component.

### Dataflow Partition
FINN separates the graph into a "parent" (CPU-side) and a "child" (FPGA-side). Operations like the TopK argmax that run on the ARM CPU stay in the parent. Everything between the DMA transfers goes into the FPGA partition.

### DWC (Data Width Converter)
Each MVAU produces outputs at a specific bit-width (determined by SIMD × bitwidth). The next layer consumes at a different bit-width. FINN automatically inserts DWC nodes to adapt between them, avoiding any software intervention in the data path.

### FIFO
Between each pair of layers, FINN inserts a shallow FIFO buffer. This decouples the pipeline stages so that a slow layer doesn't block a fast one, preventing deadlock in the streaming dataflow architecture.

### ZynqBuild
The final compilation step. It runs Vivado IP Integrator to stitch all HLS IP blocks together, hooks up the DMA engine to the ARM PS via AXI, runs full place-and-route for the xc7z020, and outputs the `.bit` + `.hwh` pair.

---

## 11. Troubleshooting

**QONNX export mismatch (`np.allclose` fails)**
Ensure the model architecture in `01_export_finn.ipynb` exactly matches what was used for training in `MicroCNN_train_cell__3_.ipynb`. Any difference in layer ordering, quantization bit-widths, or padding will cause a mismatch.

**Streamlining fails / leftover Mul nodes**
This usually means a `Mul` node from BatchNorm scaling was not fully absorbed. The hardware build notebook explicitly removes residual `Mul` nodes before `TopK` as a failsafe (see Cell 4, "STEP 3 — Mul Cleanup").

**HLS synthesis fails with "interface mismatch"**
Check that PE divides MH evenly for all MVAU layers. The resource estimate cell (Cell 7) checks this and will flag broken layers with ❌.

**Vivado runs out of BRAM / LUTs**
The Pynq-Z2 has limited resources. If the build fails at implementation:
- Reduce SIMD values in the folding cell.
- Move weight storage from `block` (BRAM) to `distributed` (LUTs) for smaller layers — or vice versa for larger ones.
- Increase `CLK_NS` (e.g., `20.0` instead of `10.0`) to give Vivado more timing slack.

**Docker networking issues / Xilinx tools not found**
Ensure `FINN_XILINX_PATH` points to the correct directory (the one containing `Vivado/` and/or `Vitis_HLS/` subdirectories) before launching `./run-docker.sh`.

**PYNQ board not reachable over SSH**
Confirm the board's IP address with `hostname -I` on the board's local terminal. Make sure SSH public key was copied with `ssh-copy-id` from within the FINN Docker container, not from your host terminal.

---

## 12. References

- **FINN Documentation:** https://finn.readthedocs.io/en/latest/
- **FINN GitHub:** https://github.com/Xilinx/finn
- **FINN Getting Started:** https://finn.readthedocs.io/en/latest/getting_started.html
- **Brevitas (QAT framework):** https://github.com/Xilinx/brevitas
- **QONNX:** https://github.com/fastmachinelearning/qonnx
- **PYNQ Project:** http://www.pynq.io/
- **FashionMNIST Dataset:** https://github.com/zalandoresearch/fashion-mnist
- **Netron (ONNX visualizer):** https://netron.app/
- **FINN tutorial video:** https://www.youtube.com/watch?v=zw2aG4PhzmA
- **QAT guidelines (FINN/hls4ml):** https://bit.ly/finn-hls4ml-qat-guidelines

---

*README written for the project at: https://github.com/sankaranarayanan23/Image-classification-in-PYNQ-Z2-*
