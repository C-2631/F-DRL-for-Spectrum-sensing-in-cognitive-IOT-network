# Federated Deep Reinforcement Learning for Spectrum Sensing in Cognitive IoT Networks

![Python](https://img.shields.io/badge/Python-100%25-blue)
![License](https://img.shields.io/badge/License-Open-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [File Documentation](#file-documentation)
- [System Components](#system-components)
- [Performance Metrics](#performance-metrics)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 📖 Project Overview

This repository implements a **Federated Deep Reinforcement Learning (DRL)** system for **spectrum sensing** in **Cognitive IoT Networks**. The system combines:

- **LSTM-based Neural Networks**: For spectrum modulation classification
- **Dueling DQN Agents**: For intelligent spectrum channel selection
- **Federated Learning**: Distributed training across edge servers and IoT clients
- **Cognitive Radio Technology**: Spectrum awareness and dynamic channel access

### Use Case
The system enables IoT devices in a wireless network to collectively learn optimal spectrum sensing strategies without sharing raw sensor data, protecting privacy while improving collective decision-making.

### Research Focus
- Non-IID (non-identical distributed) data partitioning using Dirichlet allocation
- Hierarchical federated learning (Central → Edge → Clients)
- Multi-agent DRL for spectrum access
- Collision avoidance in dynamic spectrum access

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Federated Learning** | Distributed model training across multiple edge servers and IoT clients |
| **LSTM Spectrum Sensor** | BiLSTM with Conv1D feature extraction and multi-head attention for modulation classification |
| **Dueling DQN** | Deep Q-Network with dueling architecture for channel selection |
| **Prioritized Replay Buffer** | Importance sampling for efficient DQN training |
| **Non-IID Data Support** | Dirichlet-sampled data distribution to simulate realistic IoT heterogeneity |
| **Hierarchical Aggregation** | FedAvg aggregation at both edge and central levels |
| **Comprehensive Logging** | JSON-based metrics tracking and visualization |
| **Multi-Metric Visualization** | Automatic plotting of accuracy, reward, collision rates |
| **GPU/CPU Hybrid** | Automatic CUDA support with memory optimization |

---

## 🏗️ Architecture

### System Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│         CENTRAL SERVER (192.168.1.13)                   │
│  - Maintains global LSTM & DQN models                   │
│  - Aggregates edge contributions (FedAvg)               │
│  - Manages FL communication rounds                       │
│  - Logs and visualizes global metrics                   │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    ┌────▼─────────────┐     ┌──────▼──────────────┐
    │ EDGE SERVER 1    │     │ EDGE SERVER N       │
    │ (192.168.1.14)   │     │ (192.168.1.x)       │
    │ - Aggregates     │     │ - Aggregates        │
    │   local clients   │     │   local clients     │
    │ - Syncs with     │     │ - Syncs with        │
    │   central        │     │   central           │
    └────┬─────────────┘     └──────┬──────────────┘
         │                          │
    ┌────▼────┐              ┌──────▼──────┐
    │ IoT      │              │ IoT Client  │
    │ Client 1 │              │ N           │
    │(RPi_1)   │              │ (RPi_N)     │
    └──────────┘              └─────────────┘
```

### Data Flow (Single Round)

1. **Central** publishes global LSTM & DQN weights
2. **Edges** receive and broadcast to local clients
3. **Clients** train locally on non-IID data, compute metrics
4. **Clients** send updated weights and reports to edges
5. **Edges** aggregate client contributions, send to central
6. **Central** aggregates all edge contributions (FedAvg)
7. Repeat for `FL_COMM_ROUNDS` iterations

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+**
- **CUDA 11.x** (optional, for GPU acceleration)
- **8GB RAM minimum** (16GB recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/C-2631/F-DRL-for-Spectrum-sensing-in-cognitive-IOT-network.git
cd F-DRL-for-Spectrum-sensing-in-cognitive-IOT-network
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib h5py
```

### Step 4: Prepare Dataset

Download RadioML 2018.01A dataset:

```bash
wget https://www.deepsig.io/datasets/radioml_2018_01a.h5 -O RadioML.hdf5
```

Or use the fallback: If `RadioML.hdf5` is unavailable, the system auto-generates synthetic data.

---

## ⚙️ Configuration

### Global Configuration in `fl_shared.py`

Edit the configuration parameters at the top of `fl_shared.py`:

#### Dataset Parameters
```python
RADIOML_HDF5_PATH = 'RadioML.hdf5'          # Path to RadioML dataset
NUM_MODULATIONS = 24                        # Number of modulation classes
NUM_IQ_SAMPLES = 1024                       # IQ samples per signal
SNR_RANGE = list(range(-20, 32, 2))        # Signal-to-Noise Ratio range
TRAIN_SNR_MIN = -6                          # Min SNR for training
TRAIN_SNR_MAX = 30                          # Max SNR for training
MAX_SAMPLES_PER_CLASS = 500                 # Cap per modulation class
```

#### LSTM Model Parameters
```python
LSTM_HIDDEN_DIM = 96                        # LSTM hidden dimension
LSTM_NUM_LAYERS = 1                         # Number of LSTM layers
LSTM_DROPOUT = 0.15                         # Dropout rate
LSTM_LR = 1e-4                              # Learning rate
LSTM_BATCH_SIZE = 12                        # Batch size
LSTM_WEIGHT_DECAY = 1e-4                    # L2 regularization
```

#### DQN Parameters
```python
NUM_CHANNELS = 4                            # Number of spectrum channels
DQN_HIDDEN_DIM = 64                         # DQN hidden dimension
DQN_LR = 5e-4                               # Learning rate
DQN_GAMMA = 0.99                            # Discount factor
DQN_EPSILON_START = 1.0                     # Initial exploration rate
DQN_EPSILON_END = 0.02                      # Min exploration rate
DQN_EPSILON_DECAY = 0.997                   # Decay per step
DQN_BUFFER_CAPACITY = 6000                  # Replay buffer size
DQN_BATCH_SIZE = 16                         # Training batch size
DQN_TARGET_UPDATE_FREQ = 200                # Target network update frequency
DQN_WARMUP_STEPS = 200                      # Steps before training starts
```

#### Reward Structure
```python
PU_COLLISION_PENALTY = -2.0                 # Penalty for hitting Primary User
SU_ACCESS_REWARD = 2.0                      # Reward for accessing free channel
SENSING_COST = 0.05                         # Cost of spectrum sensing
CORRECT_SENSING_BONUS = 0.5                 # Bonus for accurate detection
```

#### Federated Learning Parameters
```python
FL_COMM_ROUNDS = 80                         # Total communication rounds
FL_LOCAL_LSTM_EPOCHS = 1                    # Local LSTM epochs per round
FL_LOCAL_DQN_STEPS = 150                    # Local DQN steps per round
DIRICHLET_ALPHA = 0.5                       # Non-IID parameter (lower = more heterogeneous)
```

#### Network Topology
```python
TOPOLOGY = {
    'central': {
        'id': 'central',
        'host': '192.168.1.13'
    },
    'edges': [
        {
            'id': 'edge_1',
            'host': '192.168.1.14',
            'port': 9001,
            'clients': ['rpi_1']
        }
    ],
    'clients': {
        'rpi_1': {
            'type': 'RPi',
            'edge': 'edge_1',
            'ip': '192.168.1.15'
        }
    }
}
```

---

## 📊 Usage Guide

### Single-Machine Simulation

Run a complete federated learning simulation on one machine:

#### Step 1: Start Central Server
```bash
python central_server.py
```

Expected output:
```
Initializing Central Server...
Generating Initial Random Weights...
Central Server ready to train for 80 rounds.
Topology Edges: ['edge_1']

[ROUND 1] Publishing Global Models...
Waiting for all Edge weights...
```

#### Step 2: Start Edge Server (in new terminal)
```bash
python edge_server.py edge_1
```

Expected output:
```
[edge_1] Edge Server Initialized. Awaiting clients: ['rpi_1']
[edge_1] Waiting for Global Model Round 1 from Central Server...
```

#### Step 3: Start IoT Client (in new terminal)
```bash
python iot_client.py rpi_1 edge_1
```

Expected output:
```
[rpi_1] Initializing IoT Client ...
[rpi_1] Data Partition Loaded: 450 samples.
[rpi_1] Connecting to Edge: edge_1
[rpi_1] Waiting for round 1 global models from Edge...
```

All processes will run synchronously through 80 rounds. Monitor all terminals for progress.

### Multi-Client Setup

To add more clients, update `TOPOLOGY` in `fl_shared.py`:

```python
TOPOLOGY = {
    'central': {...},
    'edges': [
        {
            'id': 'edge_1',
            'host': '192.168.1.14',
            'port': 9001,
            'clients': ['rpi_1', 'rpi_2', 'rpi_3']  # Add clients here
        }
    ],
    'clients': {
        'rpi_1': {'type': 'RPi', 'edge': 'edge_1', 'ip': '192.168.1.15'},
        'rpi_2': {'type': 'RPi', 'edge': 'edge_1', 'ip': '192.168.1.16'},
        'rpi_3': {'type': 'RPi', 'edge': 'edge_1', 'ip': '192.168.1.17'},
    }
}
```

Then run additional client instances:
```bash
python iot_client.py rpi_2 edge_1  # Terminal 4
python iot_client.py rpi_3 edge_1  # Terminal 5
```

### Output Structure

After training completes:

```
FL_DRL_Outputs_v2/
├── logs/
│   ├── central_history.json          # Global metrics across all rounds
│   ├── edge_edge_1_history.json      # Edge aggregation metrics
│   ├── client_rpi_1_history.json     # Client training history
│   └── client_rpi_N_history.json
└── plots/
    ├── central_metrics_overview.png  # 6-subplot global metrics
    ├── central_edge_comparisons.png  # Edge vs edge comparison
    ├── edge_edge_1_metrics_overview.png
    ├── edge_edge_1_accuracy_trend.png
    ├── client_rpi_1_metrics_overview.png
    ├── client_rpi_1_dqn_trend.png
    └── ...
```

---

## 📁 File Documentation

### 1. **fl_shared.py** (15,055 bytes)
**Core Shared Module** - Contains all shared utilities and model definitions

#### Key Components:

**Global Configuration**
- Dataset paths, SNR ranges, modulation classes
- LSTM & DQN hyperparameters
- Reward structure and transition probabilities
- Network topology definition

**Data Loading & Preprocessing**
- `load_radioml2018()`: Loads RadioML HDF5 dataset with SNR filtering
- `train_test_split()`: 80-20 train-test split with reproducible seeding
- `partition_non_iid()`: Dirichlet-sampled non-IID data partitioning
- `IQDataset`: PyTorch Dataset for IQ samples with normalization
- `create_loader()`: DataLoader factory with batch configuration

**Models**
- `LSTMSpectrumSensor`: BiLSTM + Conv1D + Multi-head Attention for modulation classification
  - Input: IQ samples (2D signal)
  - Output: Logits (24 classes) + Features
  - Architecture: Conv1D (3 layers) → LSTM (bidirectional) → Attention → MLP
  
- `DuelingDQN`: Dueling architecture for channel selection
  - Input: State (LSTM features + channel states + SNR)
  - Output: Q-values for each channel
  - Architecture: Shared trunk → Value stream + Advantage stream
  
- `PrioritizedReplayBuffer`: Experience replay with TD-error-based prioritization
  - Improves sample efficiency for DQN training
  
- `DQNAgent`: Complete DQN training loop
  - ε-greedy exploration → exploitation
  - Soft target network updates (polyak averaging)
  - Gradient clipping for stability

**Environment**
- `CognitiveRadioEnv`: Markov Decision Process for spectrum access
  - State: LSTM features + channel occupancy + SNR
  - Action: Select one of 4 channels
  - Reward: Collision penalty vs. access reward

**Federated Learning Utilities**
- `fedavg()`: Weighted averaging aggregation (FedAvg algorithm)
  - Weights by sample count for weighted averaging
  
- `wait_for_file_and_load()`: Synchronization primitive for file-based IPC
- `dict_save()`: PyTorch checkpoint saving

---

### 2. **central_server.py** (8,706 bytes)
**Central Aggregation Server** - Orchestrates federated learning rounds

#### Class: `CentralServer`

**Initialization**
```python
def __init__(self, server_ip='192.168.1.1'):
    self.global_lstm = None          # Global LSTM model
    self.global_dqn = None           # Global DQN model
    self.history = {...}             # Metrics tracking
    self.edge_history = {}           # Per-edge history
```

**Key Methods**

| Method | Purpose |
|--------|---------|
| `aggregate_edges(pkgs, reps)` | FedAvg aggregation of edge contributions; returns global metrics |
| `log_round(rnd, rep)` | Records metrics for round and per-edge performance |
| `save_history(log_dir)` | Saves JSON history file for post-analysis |
| `plot_metrics(plot_dir)` | Generates 6 comprehensive matplotlib figures |
| `print_summary(rnd, total, rep)` | Console output of round performance |

**Tracked Metrics**
- LSTM accuracy (classification accuracy across modulations)
- LSTM loss (cross-entropy loss)
- DQN reward (average reward per episode)
- Collision rate (% collisions with primary users)
- Total samples (training data volume)
- Per-edge breakdowns

**Main Loop**
```
1. Initialize global models with random weights
2. For each round:
   a. Save global models to sync_dir
   b. Wait for all edges to contribute
   c. Aggregate using FedAvg
   d. Log and print metrics
3. Save final models and plots
```

**Output Visualizations**
- Global LSTM Accuracy (%)
- Global LSTM Loss
- Global DQN Reward
- Collision Rate (%)
- Total Samples per Round
- Edge Accuracy Comparison
- Edge DQN Reward Comparison (additional plot)
- Edge Collision Comparison (additional plot)

---

### 3. **edge_server.py** (8,065 bytes)
**Edge Aggregation Server** - Aggregates local clients and coordinates with central

#### Class: `FedEdgeServer`

**Initialization**
```python
def __init__(self, edge_id, client_ids):
    self.edge_id = edge_id              # Unique edge identifier
    self.client_ids = client_ids        # List of assigned clients
    self.peers = []                     # Peer edge servers (future)
    self.lstm_weights = None            # Current LSTM weights
    self.dqn_weights = None             # Current DQN weights
    self.history = {...}                # Metrics tracking
```

**Key Methods**

| Method | Purpose |
|--------|---------|
| `aggregate_clients(reports, lstm_wts, dqn_wts)` | FedAvg local client contributions |
| `set_peers(peer_list)` | Configure peer edges for mesh federation |
| `log_round(rnd, rep)` | Record per-round metrics |
| `save_history(log_dir)` | Save JSON history for edge |
| `plot_metrics(plot_dir)` | Generate edge-specific visualizations |

**Tracked Metrics**
- Average LSTM loss (across local clients)
- Average LSTM accuracy
- Average DQN reward
- Average collision rate
- Divergence score (reserved for mesh sync)
- Total samples

**Communication Protocol**
```
Round 1:
  Wait ← central_global_rnd1.pth (global models)
  Broadcast → edge_1_global_rnd1.pth (to clients)
  Wait ← client_*_rnd1.pth (all clients)
  Aggregate → edge weights
  Send → agg_edge_1_rnd1.pth (to central)
```

**Visualizations**
- LSTM Accuracy trend
- LSTM Loss trend
- DQN Reward trend
- Collision Rate trend
- Total Samples
- Combined Score (60% accuracy + 30% reward + 10% no-collision)
- Accuracy with 3-step moving average

---

### 4. **iot_client.py** (11,330 bytes)
**IoT Edge Device Client** - Performs local training on IoT hardware

#### Class: `FLClient`

**Initialization**
```python
def __init__(self, client_id, edge_id, X, Y, Z, client_type='RPi'):
    self.client_id = client_id          # Unique client ID
    self.edge_id = edge_id              # Parent edge server
    self.client_type = client_type      # Device type (RPi, ESP32)
    self.X, self.Y, self.Z = X, Y, Z   # Local non-IID dataset
    self.num_samples = len(X)
    
    # Models
    self.lstm = LSTMSpectrumSensor()    # For feature extraction
    self.dqn = DQNAgent(...)            # For spectrum selection
    self.env = CognitiveRadioEnv()      # Simulated environment
    
    # Training
    self.lstm_optimizer = Adam(...)     # LSTM optimizer
    self.scaler = GradScaler()          # Mixed precision scaling
```

**Key Methods**

| Method | Purpose |
|--------|---------|
| `train_lstm()` | Train LSTM on local data for 1 epoch, return loss & accuracy |
| `train_dqn()` | Train DQN for N steps using LSTM features, return reward & collision rate |
| `train_round()` | Execute both LSTM and DQN training, compile report |
| `get_lstm_weights()` / `set_lstm_weights()` | Weight management |
| `get_dqn_weights()` / `set_dqn_weights()` | Weight management |
| `log_round(rnd, report)` | Record metrics |
| `save_history()` | Save JSON history |
| `plot_metrics()` | Generate client visualizations |

**LSTM Training Details**
- Optimizer: Adam with lr=1e-4, weight_decay=1e-4
- Loss: CrossEntropyLoss
- Mixed precision: Automatic scaling for GPU/CPU
- Batch size: 12 samples
- Epochs: 1 (local epoch per round)

**DQN Training Details**
- Steps per round: 150
- Epsilon decay: 0.997 per step
- Prioritized replay buffer with TD-error weighting
- Soft target network update (1% policy → 99% target)
- Gradient clipping: max norm 10.0

**Tracked Metrics**
- LSTM loss and accuracy
- DQN reward (mean across episodes)
- Collision rate
- Epsilon decay schedule

**Communication Protocol**
```
Round 1:
  Wait ← edge_1_global_rnd1.pth
  Load models & train locally (LSTM + DQN)
  Send → rpi_1_rnd1.pth (to edge)
  
Payload structure:
{
  'client_id': 'rpi_1',
  'report': {...},  # Metrics
  'lstm_weights': {...},  # State dict
  'dqn_weights': {...}    # State dict
}
```

**Visualizations**
- LSTM Accuracy trend
- LSTM Loss trend
- DQN Reward trend
- Collision Rate trend
- Epsilon decay
- Combined Score
- DQN reward with 3-step moving average

---

## 🔧 System Components

### 1. LSTM Spectrum Sensor Architecture

```
Input (B×2×1024): IQ signal samples
  ↓
Conv1D Block 1: 2→32 channels, kernel=7, BatchNorm, GELU, MaxPool
  ↓
Conv1D Block 2: 32→64 channels, kernel=5, BatchNorm, GELU
  ↓
Conv1D Block 3: 64→128 channels, kernel=3, BatchNorm, GELU, MaxPool(4)
  ↓
BiLSTM: 128→96 (bidirectional, 1 layer)
  ↓
Multi-Head Attention: 192 heads, 8 attention heads
  ↓
LayerNorm + Residual
  ↓
MLP Classifier: 192→256→128→24
  ↓
Output: Logits (24 modulations) + Features (192-dim)
```

**Purpose**: Extract discriminative features from raw IQ signals for modulation classification

---

### 2. Dueling DQN Architecture

```
Input State (B × 513): [LSTM features(192) + Channel states(4) + SNR(1) + padding]
  ↓
Shared Trunk:
  Linear(513→64) + LayerNorm + GELU
  Linear(64→64) + LayerNorm + GELU
  ↓
Value Stream:        Advantage Stream:
  Linear(64→32)       Linear(64→32)
  GELU                GELU
  Linear(32→1)        Linear(32→4)
  ↓                   ↓
  Combine: Q = V + (A - mean(A))
  ↓
Output: Q-values for 4 channels
```

**Purpose**: Learn optimal channel selection policy via reward signals

---

### 3. Cognitive Radio Environment

**State Space**
```python
[
  lstm_features (192),      # Modulation classification features
  channel_states (4),       # 0=idle, 1=occupied
  snr (1),                  # Normalized signal-to-noise ratio
] → Total: 197 dims
```

**Action Space**
- Discrete: Select one of 4 spectrum channels to access

**Reward Structure**
```
if collision_with_primary_user:
    r = -2.0 (penalty) + 0.5 (bonus if all detected idle)
else:
    r = +2.0 (access reward)
r -= 0.05 (sensing cost)
```

**Channel Dynamics** (Markov)
- State transitions follow PU_TRANSITION_PROBS
- Per-channel: (p00, p11) = probability of staying in each state

---

## 📈 Performance Metrics

### Key Metrics Tracked

1. **LSTM Classification Accuracy**: % correct modulation predictions
2. **LSTM Loss**: Cross-entropy loss on local data
3. **DQN Reward**: Mean reward across DQN episodes
4. **Collision Rate**: % spectrum accesses that collide with primary users
5. **Total Samples**: Training data volume per round
6. **Combined Score**: Weighted metric (60% accuracy + 30% reward efficiency + 10% no-collision)

### Aggregation Methods

| Level | Aggregation Method |
|-------|-------------------|
| Client → Edge | FedAvg (weighted by sample count) |
| Edge → Central | FedAvg (weighted by sample count) |
| Metrics | Mean across contributors |

### Expected Performance Progression

- **Early Rounds (1-20)**: Rapid accuracy improvement, collision rate decreases
- **Mid Rounds (20-50)**: Convergence of accuracy, stable DQN reward
- **Late Rounds (50-80)**: Fine-tuning, potential overfitting if not regularized

---

## 🔬 Advanced Configuration

### Custom Network Topology

Add multiple edges with different clients:

```python
TOPOLOGY = {
    'central': {'id': 'central', 'host': '192.168.1.13'},
    'edges': [
        {
            'id': 'edge_1',
            'host': '192.168.1.14',
            'port': 9001,
            'clients': ['rpi_1', 'rpi_2']
        },
        {
            'id': 'edge_2',
            'host': '192.168.1.24',
            'port': 9002,
            'clients': ['rpi_3', 'rpi_4']
        }
    ],
    'clients': {
        'rpi_1': {'type': 'RPi', 'edge': 'edge_1', 'ip': '192.168.1.15'},
        'rpi_2': {'type': 'RPi', 'edge': 'edge_1', 'ip': '192.168.1.16'},
        'rpi_3': {'type': 'RPi', 'edge': 'edge_2', 'ip': '192.168.1.25'},
        'rpi_4': {'type': 'RPi', 'edge': 'edge_2', 'ip': '192.168.1.26'},
    }
}
```

### Hyperparameter Tuning

**For faster convergence:**
```python
FL_COMM_ROUNDS = 40           # Reduce rounds
LSTM_LR = 5e-4                # Increase learning rate
DQN_LR = 1e-3                 # Increase learning rate
DIRICHLET_ALPHA = 1.0         # More IID data
```

**For better generalization:**
```python
LSTM_DROPOUT = 0.25           # Increase regularization
DIRICHLET_ALPHA = 0.1         # More heterogeneous data
FL_LOCAL_LSTM_EPOCHS = 2      # More local training
```

**For spectrum sensing accuracy:**
```python
MAX_SAMPLES_PER_CLASS = 1000   # More training data
TRAIN_SNR_MIN = -20           # Wider SNR range
TRAIN_SNR_MAX = 32
```

### GPU Optimization

Adjust CUDA memory fraction in `fl_shared.py`:

```python
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.75)  # 75% of GPU VRAM
```

---

## 🐛 Troubleshooting

### Issue: "ResourceExhaustedError: Out of memory"

**Solution:**
```python
# In fl_shared.py, reduce model size:
LSTM_HIDDEN_DIM = 64          # Instead of 96
DQN_HIDDEN_DIM = 32           # Instead of 64
LSTM_BATCH_SIZE = 8           # Instead of 12
DQN_BATCH_SIZE = 8            # Instead of 16
```

### Issue: Processes hang at "Waiting for..."

**Diagnosis:**
- Ensure all processes started: Central → Edge → Clients
- Check sync_dir exists and is writable
- Verify TOPOLOGY configuration matches running clients

**Solution:**
```bash
# Check sync_dir
ls -la sync_dir/

# Clean up stale files
rm -rf sync_dir/
rm -rf FL_DRL_Outputs_v2/

# Restart all processes
```

### Issue: Poor accuracy or high collision rate

**Root causes:**
1. Non-IID data too heterogeneous (DIRICHLET_ALPHA too low)
2. Learning rates too high (instability)
3. Insufficient local training (FL_LOCAL_LSTM_EPOCHS too low)

**Solutions:**
```python
# More homogeneous data
DIRICHLET_ALPHA = 1.0

# Conservative learning rates
LSTM_LR = 5e-5
DQN_LR = 2e-4

# More local epochs
FL_LOCAL_LSTM_EPOCHS = 2
FL_LOCAL_DQN_STEPS = 300
```

### Issue: "RadioML.hdf5 not found"

The system automatically falls back to synthetic data generation. No action needed.

---

## 📝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Test on both CPU and GPU
- Update README for new features
- Maintain backward compatibility

---

## 📚 References

- **Federated Learning**: [McMahan et al., 2016](https://arxiv.org/abs/1602.05629)
- **Dueling DQN**: [Wang et al., 2015](https://arxiv.org/abs/1511.06581)
- **Prioritized Experience Replay**: [Schaul et al., 2015](https://arxiv.org/abs/1511.05952)
- **RadioML Dataset**: [O'Shea & Hoydis, 2016](https://arxiv.org/abs/1602.04105)
- **Cognitive Radio Networks**: [Akyildiz et al., 2006](https://dx.doi.org/10.1016/j.comnet.2006.05.001)

---

## 📄 License

This project is open-source and available under the MIT License. See LICENSE file for details.

---

## 👨‍💻 Author

**C-2631**

For questions or issues, please open a GitHub Issue.

---

## 🙏 Acknowledgments

- RadioML team for the modulation classification dataset
- PyTorch team for the deep learning framework
- Federated learning research community for algorithmic innovations

---

**Last Updated**: May 26, 2026  
**Status**: Active Development
