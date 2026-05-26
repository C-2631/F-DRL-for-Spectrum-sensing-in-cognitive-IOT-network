import os, sys, copy, time, random, json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py

# -- Global Configuration --
# -- Global Configuration --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RADIOML_HDF5_PATH = 'RadioML.hdf5' 
NUM_MODULATIONS = 24
NUM_IQ_SAMPLES = 1024
SNR_RANGE = list(range(-20, 32, 2))
MODULATION_CLASSES = [
    'OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
    '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM',
    '128QAM','256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC',
    'AM-DSB-SC','FM','GMSK','OQPSK',
]

TRAIN_SNR_MIN = -6
TRAIN_SNR_MAX = 30

MAX_SAMPLES_PER_CLASS = 500

LSTM_HIDDEN_DIM = 96
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.15
LSTM_LR = 1e-4
LSTM_BATCH_SIZE = 12
LSTM_WEIGHT_DECAY = 1e-4

NUM_CHANNELS = 4

DQN_HIDDEN_DIM = 64
DQN_LR = 5e-4
DQN_GAMMA = 0.99

DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.02
DQN_EPSILON_DECAY = 0.997

DQN_BUFFER_CAPACITY = 6000
DQN_BATCH_SIZE = 16
DQN_TARGET_UPDATE_FREQ = 200
DQN_WARMUP_STEPS = 200

PU_COLLISION_PENALTY = -2.0
SU_ACCESS_REWARD = 2.0
SENSING_COST = 0.05
CORRECT_SENSING_BONUS = 0.5

PU_TRANSITION_PROBS = [
    (0.80, 0.30),
    (0.70, 0.40),
    (0.90, 0.20),
    (0.60, 0.50)
]

FL_COMM_ROUNDS = 80
FL_LOCAL_LSTM_EPOCHS = 1
FL_LOCAL_DQN_STEPS = 150

DIRICHLET_ALPHA = 0.5

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

SYNC_DIR = Path('sync_dir')
SYNC_DIR.mkdir(parents=True, exist_ok=True)

# -- Data Loading & Partitioning --
def load_radioml2018(path=RADIOML_HDF5_PATH, snr_min=TRAIN_SNR_MIN, snr_max=TRAIN_SNR_MAX, max_per_class=MAX_SAMPLES_PER_CLASS):
    try:
        with h5py.File(str(path), 'r') as f:
            Y_all = np.array(f['Y'])
            Z_all = np.array(f['Z']).flatten().astype(np.float32)
            Y_int = np.argmax(Y_all, axis=1).astype(np.int64)

            mask = (Z_all >= snr_min) & (Z_all <= snr_max)
            valid_idx = np.where(mask)[0]
            Y_valid = Y_int[valid_idx]

            rng = np.random.default_rng(42)
            keep_local = []
            for c in range(NUM_MODULATIONS):
                idx = np.where(Y_valid == c)[0]
                if len(idx) > max_per_class:
                    idx = rng.choice(idx, max_per_class, replace=False)
                keep_local.extend(idx.tolist())
            keep_local = np.array(keep_local)
            final_idx = np.sort(valid_idx[keep_local])

            X_dataset = f['X']
            X = np.array(X_dataset[final_idx]) # type: ignore
            Y = Y_int[final_idx]
            Z = Z_all[final_idx]
        perm = np.random.permutation(len(X))
        X, Y, Z = X[perm], Y[perm], Z[perm]
        return X.astype(np.float32), Y, Z, MODULATION_CLASSES
    except Exception as e:
        print(f"[DATASET] Could not load HDF5 ({e}) - generating random fallback.")
        rng = np.random.default_rng(42)
        N = max_per_class * NUM_MODULATIONS // 4 
        X = rng.standard_normal((N, 1024, 2)).astype(np.float32)
        Y = rng.integers(0, NUM_MODULATIONS, N).astype(np.int64)
        Z = rng.choice([s for s in SNR_RANGE if snr_min <= s <= snr_max], N).astype(np.float32)
        return X, Y, Z, MODULATION_CLASSES

def train_test_split(X, Y, Z, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    sp = int(len(X) * (1 - test_ratio))
    tr, te = idx[:sp], idx[sp:]
    return X[tr], Y[tr], Z[tr], X[te], Y[te], Z[te]

def partition_non_iid(X, Y, Z, num_clients, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(num_clients)]
    for c in range(NUM_MODULATIONS):
        cls = np.where(Y == c)[0]; rng.shuffle(cls)
        props = rng.dirichlet(np.full(num_clients, alpha))
        props = (props * len(cls)).astype(int)
        props[0] += len(cls) - props.sum()
        s = 0
        for k in range(num_clients):
            client_idx[k].extend(cls[s:s+props[k]].tolist()); s += props[k]
    result = []
    for k in range(num_clients):
        idx = np.array(client_idx[k]); rng.shuffle(idx)
        result.append((X[idx], Y[idx], Z[idx]))
    return result

class IQDataset(Dataset):
    def __init__(self, X, Y, Z=None):
        X = X.copy()
        mean = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2), keepdims=True) + 1e-8
        X = ((X - mean) / std).astype(np.float32)
        self.X = torch.from_numpy(X).permute(0, 2, 1)
        self.Y = torch.from_numpy(Y).long()
        self.Z = torch.from_numpy(Z).float() if Z is not None else None

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        d = {'x': self.X[i], 'y': self.Y[i]}
        if self.Z is not None: d['snr'] = self.Z[i]
        return d

def create_loader(X, Y, Z=None, batch_size=128, shuffle=True):
    return DataLoader(IQDataset(X, Y, Z), batch_size=batch_size, shuffle=shuffle, num_workers=0)

def fedavg(weights_list, sample_counts=None):
    if len(weights_list) == 1: return copy.deepcopy(weights_list[0])
    if sample_counts is None: sample_counts = [1] * len(weights_list)
    total = sum(sample_counts)
    ratios = [n / total for n in sample_counts]
    avg = {}
    for key in weights_list[0]:
        avg[key] = torch.zeros_like(weights_list[0][key], dtype=torch.float32)
        for w, r in zip(weights_list, ratios):
            avg[key] += w[key].float() * r
    return avg

# -- Models --
class LSTMSpectrumSensor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS, num_classes=NUM_MODULATIONS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_directions = 2
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        feat = hidden_dim * 2
        self.attn = nn.MultiheadAttention(feat, 8, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(feat)
        self.classifier = nn.Sequential(
            nn.Linear(feat, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        z = self.conv(x).permute(0, 2, 1)
        z, _ = self.lstm(z)
        z_attn, _ = self.attn(z, z, z)
        features = self.attn_norm(z + z_attn).mean(dim=1)
        return self.classifier(features), features

    def get_feature_dim(self):
        return self.hidden_dim * self.num_directions

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim=NUM_CHANNELS, hidden_dim=DQN_HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.adv = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, action_dim))

    def forward(self, x):
        t = self.trunk(x)
        V = self.value(t)
        A = self.adv(t)
        return V + (A - A.mean(dim=1, keepdim=True))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity, self.alpha, self.beta_start, self.beta_frames = capacity, alpha, beta_start, beta_frames
        self.frame, self.pos, self.buffer = 1, 0, []
        self.priorities = np.zeros(capacity, np.float32)

    @property
    def beta(self): return min(1.0, self.beta_start + self.frame * (1 - self.beta_start) / self.beta_frames)

    def push(self, s, a, r, s2, d):
        mp = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity: self.buffer.append((s, a, r, s2, d))
        else: self.buffer[self.pos] = (s, a, r, s2, d)
        self.priorities[self.pos] = mp
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        p = self.priorities[:n] ** self.alpha
        p /= p.sum()
        idx = np.random.choice(n, batch_size, p=p, replace=False)
        samp = [self.buffer[i] for i in idx]
        w = (n * p[idx]) ** (-self.beta)
        w /= w.max()
        self.frame += 1
        s, a, r, s2, d = zip(*samp)
        return (np.array(s, np.float32), np.array(a, np.int64), np.array(r, np.float32), 
                np.array(s2, np.float32), np.array(d, np.float32), idx, np.array(w, np.float32))

    def update_priorities(self, idx, prios):
        for i, p in zip(idx, prios): self.priorities[i] = p + 1e-6
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim=NUM_CHANNELS):
        self.action_dim = action_dim
        self.policy = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.policy.parameters(), lr=DQN_LR, eps=1.5e-4)
        self.memory = PrioritizedReplayBuffer(DQN_BUFFER_CAPACITY)
        self.epsilon = DQN_EPSILON_START
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon: return random.randrange(self.action_dim)
        with torch.no_grad():
            return self.policy(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).argmax(1).item()

    def decay_epsilon(self):
        self.epsilon = max(DQN_EPSILON_END, self.epsilon * DQN_EPSILON_DECAY)

    def optimize(self):
        if len(self.memory) < max(DQN_WARMUP_STEPS, DQN_BATCH_SIZE): return 0.0
        s, a, r, s2, d, idx, w = self.memory.sample(DQN_BATCH_SIZE)
        s = torch.FloatTensor(s).to(DEVICE); a = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
        r = torch.FloatTensor(r).unsqueeze(1).to(DEVICE); s2 = torch.FloatTensor(s2).to(DEVICE)
        d = torch.FloatTensor(d).unsqueeze(1).to(DEVICE); w = torch.FloatTensor(w).unsqueeze(1).to(DEVICE)
        curr_q = self.policy(s).gather(1, a)
        with torch.no_grad():
            best_a = self.policy(s2).argmax(1, keepdim=True)
            tgt_q = r + DQN_GAMMA * self.target(s2).gather(1, best_a) * (1 - d)
        td_err = (curr_q - tgt_q).abs().detach().cpu().numpy().flatten()
        self.memory.update_priorities(idx, td_err)
        loss = (w * F.smooth_l1_loss(curr_q, tgt_q, reduction='none')).mean()
        self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0); self.opt.step()
        self.steps += 1
        if self.steps % DQN_TARGET_UPDATE_FREQ == 0:
            for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
                tp.data.copy_(0.01 * pp.data + 0.99 * tp.data)
        return loss.item()

    def get_weights(self):
        return copy.deepcopy(self.policy.state_dict())

    def set_weights(self, w):
        current = self.policy.state_dict()
        compatible = {}
        skipped = []
        for key, value in w.items():
            if key in current and current[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)

        if skipped:
            print(f"[DQN] Skipping incompatible checkpoint keys: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")

        current.update(compatible)
        self.policy.load_state_dict(current)
        self.target.load_state_dict(current)

class CognitiveRadioEnv:
    def __init__(self, lstm_feature_dim=512):
        self.lstm_feature_dim = lstm_feature_dim
        self.obs_dim = lstm_feature_dim + NUM_CHANNELS + 1
        self.channel_states = np.zeros(NUM_CHANNELS, np.float32)
        self.lstm_features = np.zeros(lstm_feature_dim, np.float32)
        self.current_snr = 10.0
        self.step_count = 0
        self.total_collisions = 0
        self.total_reward = 0.0

    def set_lstm_features(self, features, snr=10.0):
        self.lstm_features = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
        self.current_snr = float(snr) / 30.0

    def reset(self):
        self.channel_states = np.random.choice([0.0, 1.0], size=NUM_CHANNELS).astype(np.float32)
        self.step_count = 0; self.total_collisions = 0; self.total_reward = 0.0
        return self._obs()

    def _obs(self):
        return np.concatenate([self.lstm_features, self.channel_states, [self.current_snr]]).astype(np.float32)

    def step(self, action):
        for c in range(NUM_CHANNELS):
            p00, p11 = PU_TRANSITION_PROBS[c]
            curr = int(self.channel_states[c])
            if curr == 0 and random.random() > p00: self.channel_states[c] = 1.0
            elif curr == 1 and random.random() > p11: self.channel_states[c] = 0.0
        r = 0.0
        collision = (self.channel_states[action] == 1.0)
        if collision:
            self.total_collisions += 1
            r = PU_COLLISION_PENALTY
            if all(s == 1.0 for s in self.channel_states): r += CORRECT_SENSING_BONUS
        else: r = SU_ACCESS_REWARD
        r -= SENSING_COST
        self.total_reward += r
        self.step_count += 1
        done = self.step_count >= DQN_WARMUP_STEPS
        return self._obs(), r, done, {}

    @property
    def collision_rate(self): return self.total_collisions / max(self.step_count, 1)

def wait_for_file_and_load(filepath):
    while not os.path.exists(filepath): time.sleep(1)
    try: return torch.load(filepath, map_location='cpu', weights_only=False)
    except: time.sleep(0.5); return torch.load(filepath, map_location='cpu', weights_only=False)

def dict_save(data, filepath):
    torch.save(data, filepath)
