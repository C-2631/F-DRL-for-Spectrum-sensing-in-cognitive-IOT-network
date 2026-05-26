import os, sys, copy, time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import amp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)


# Import shared configuration and definitions
from fl_shared import *

class FLClient:
    """Simulates one IoT edge device (RPi or ESP32) in the distributed FL system."""
    def __init__(self, client_id, edge_id, X, Y, Z, client_type='RPi'):
        # --- PyTorch performance tuning ---
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(2)
        torch.cuda.empty_cache()

        self.client_id = client_id
        self.edge_id = edge_id
        self.client_type = client_type
        self.X, self.Y, self.Z = X, Y, Z
        self.num_samples = len(X)
        self.device = DEVICE

        self.lstm = LSTMSpectrumSensor().to(self.device)
        feat = self.lstm.get_feature_dim() 
        self.dqn = DQNAgent(feat + NUM_CHANNELS + 1)
        self.env = CognitiveRadioEnv(feat)

        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
        self.crit = nn.CrossEntropyLoss()
        self.scaler = amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
        self.history = {
            'round': [], 'lstm_loss': [], 'lstm_accuracy': [], 'dqn_reward': [],
            'collision_rate': [], 'epsilon': [],
        }

    def set_lstm_weights(self, w):
        self.lstm.load_state_dict(copy.deepcopy(w))

    def set_dqn_weights(self, w):
        self.dqn.set_weights(w)

    def get_lstm_weights(self):
        return copy.deepcopy(self.lstm.state_dict())

    def get_dqn_weights(self):
        return self.dqn.get_weights()

    def train_lstm(self):
        """Train the BiLSTM model on the client's local data."""
        loader = create_loader(self.X, self.Y, self.Z, LSTM_BATCH_SIZE)
        self.lstm.train()
        tl, tc, tn = 0.0, 0, 0

        for epoch in range(FL_LOCAL_LSTM_EPOCHS):
            for i, b in enumerate(loader):
                x = b['x'].to(self.device)
                y = b['y'].to(self.device)

                self.lstm_optimizer.zero_grad()

                with amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                    logits, _ = self.lstm(x)
                    loss = self.crit(logits, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.lstm_optimizer)
                self.scaler.update()

                tl += loss.item() * len(y)
                tc += (logits.argmax(1) == y).sum().item()
                tn += len(y)

                if i % 25 == 0:
                    print(f'  [Client {self.client_id}] LSTM Epoch {epoch+1}/{FL_LOCAL_LSTM_EPOCHS}, Batch {i}/{len(loader)}, Loss: {loss.item():.4f}')
                    time.sleep(0.08)

        return tl / max(tn, 1), tc / max(tn, 1)

    def train_dqn(self):
        """Train the Dueling DQN agent using the trained LSTM as a sensor."""
        self.lstm.eval()
        rng = np.random.default_rng()
        state = self.env.reset()
        ep_r, ep_rs = 0.0, []

        for step in range(FL_LOCAL_DQN_STEPS):
            if step % 25 == 0:
                print(f'  [Client {self.client_id}] DQN Step {step}/{FL_LOCAL_DQN_STEPS}')
                time.sleep(0.08)

            with torch.no_grad():
                i = rng.integers(0, len(self.X))
                iq = torch.FloatTensor(self.X[i]).unsqueeze(0).permute(0, 2, 1).to(self.device)
                _, f = self.lstm(iq)
                self.env.set_lstm_features(f[0])

            state = self.env._obs()
            a = self.dqn.select_action(state)
            s2, r, done, _ = self.env.step(a)

            self.dqn.memory.push(state, a, r, s2, float(done))
            self.dqn.optimize()

            ep_r += r

            if done:
                ep_rs.append(ep_r)
                ep_r = 0.0
                state = self.env.reset()

        self.dqn.decay_epsilon()
        return (float(np.mean(ep_rs)) if ep_rs else ep_r), self.env.collision_rate

    def train_round(self):
        ll, la = self.train_lstm()
        dr, cr = self.train_dqn()

        return {
            'client_id': self.client_id,
            'client_type': self.client_type,
            'num_samples': self.num_samples,
            'lstm_loss': ll,
            'lstm_accuracy': la,
            'dqn_reward': dr,
            'collision_rate': cr,
            'epsilon': self.dqn.epsilon,
        }

    def log_round(self, rnd, report):
        self.history['round'].append(rnd)
        self.history['lstm_loss'].append(report['lstm_loss'])
        self.history['lstm_accuracy'].append(report['lstm_accuracy'])
        self.history['dqn_reward'].append(report['dqn_reward'])
        self.history['collision_rate'].append(report['collision_rate'])
        self.history['epsilon'].append(report['epsilon'])

    def save_history(self, log_dir):
        out_file = log_dir / f"client_{self.client_id}_history.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print(f"[{self.client_id}] History saved -> {out_file}")

    def plot_metrics(self, plot_dir):
        rounds = self.history['round']
        if not rounds:
            print(f"[{self.client_id}] No rounds to plot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes.ravel()

        ax[0].plot(rounds, [x * 100 for x in self.history['lstm_accuracy']], marker='o', color='tab:blue')
        ax[0].set_title(f'{self.client_id}: LSTM Accuracy (%)')
        ax[0].set_xlabel('Round')
        ax[0].grid(alpha=0.3)

        ax[1].plot(rounds, self.history['lstm_loss'], marker='o', color='tab:red')
        ax[1].set_title(f'{self.client_id}: LSTM Loss')
        ax[1].set_xlabel('Round')
        ax[1].grid(alpha=0.3)

        ax[2].plot(rounds, self.history['dqn_reward'], marker='o', color='tab:green')
        ax[2].set_title(f'{self.client_id}: DQN Reward')
        ax[2].set_xlabel('Round')
        ax[2].grid(alpha=0.3)

        ax[3].plot(rounds, [x * 100 for x in self.history['collision_rate']], marker='o', color='tab:purple')
        ax[3].set_title(f'{self.client_id}: Collision Rate (%)')
        ax[3].set_xlabel('Round')
        ax[3].grid(alpha=0.3)

        ax[4].plot(rounds, self.history['epsilon'], marker='o', color='tab:orange')
        ax[4].set_title(f'{self.client_id}: Epsilon Decay')
        ax[4].set_xlabel('Round')
        ax[4].grid(alpha=0.3)

        combined = []
        for a, r, c in zip(self.history['lstm_accuracy'], self.history['dqn_reward'], self.history['collision_rate']):
            combined.append((0.6 * a) + (0.3 * max(r, 0.0) / (abs(r) + 1.0)) + (0.1 * (1.0 - c)))
        ax[5].plot(rounds, [x * 100 for x in combined], marker='o', color='tab:brown')
        ax[5].set_title(f'{self.client_id}: Combined Score (%)')
        ax[5].set_xlabel('Round')
        ax[5].grid(alpha=0.3)

        fig.tight_layout()
        out_file = plot_dir / f"client_{self.client_id}_metrics_overview.png"
        fig.savefig(out_file, dpi=100)
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4.5))
        if len(rounds) >= 3:
            win = 3
            smooth = np.convolve(self.history['dqn_reward'], np.ones(win) / win, mode='valid')
            ax2.plot(rounds[win - 1:], smooth, marker='o', color='tab:cyan', label='DQN reward MA(3)')
        ax2.plot(rounds, self.history['dqn_reward'], alpha=0.4, color='tab:green', label='DQN reward raw')
        ax2.set_title(f'{self.client_id}: DQN Reward Trend (Raw + Moving Avg)')
        ax2.set_xlabel('Round')
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        out_file2 = plot_dir / f"client_{self.client_id}_dqn_trend.png"
        fig2.savefig(out_file2, dpi=100)
        plt.close(fig2)
        print(f"[{self.client_id}] Plots saved -> {out_file} and {out_file2}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python iot_client.py <client_id> [edge_id]")
        print("Example: python iot_client.py rpi_1 edge_1")
        sys.exit(1)

    client_id = sys.argv[1]
    edge_id = sys.argv[2] if len(sys.argv) > 2 else TOPOLOGY['clients'][client_id]['edge']

    outputs_dir = Path('FL_DRL_Outputs_v2')
    log_dir = outputs_dir / 'logs'
    plot_dir = outputs_dir / 'plots'
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{client_id}] Initializing IoT Client ...")
    
    # 1. Load Data & Find partition for this client
    # In a real environment, each device would just have its own dataset file.
    # Here, we partition it deterministically to simulate different data silos.
    X, Y, Z, _ = load_radioml2018()
    Xtr, Ytr, Ztr, Xte, Yte, Zte = train_test_split(X, Y, Z)
    
    all_cids = [cid for e in TOPOLOGY['edges'] for cid in e['clients']]
    num_clients = len(all_cids)
    parts = partition_non_iid(Xtr, Ytr, Ztr, num_clients, DIRICHLET_ALPHA)
    
    # Find index of this client
    idx = all_cids.index(client_id)
    cX, cY, cZ = parts[idx]
    
    client_type = TOPOLOGY['clients'][client_id]['type']
    print(f"[{client_id}] Data Partition Loaded: {len(cX)} samples.")
    
    client = FLClient(client_id, edge_id, cX, cY, cZ, client_type=client_type)

    print(f"[{client_id}] Connecting to Edge: {edge_id}")
    
    for round_num in range(1, FL_COMM_ROUNDS + 1):
        print(f"\n[{client_id}] Waiting for round {round_num} global models from Edge...")
        
        # In actual networking, this would be a socket read.
        # For our local simulation, we wait until edge_{edge_id}_global_rnd{round_num}.pth appears
        global_model_file = SYNC_DIR / f"{edge_id}_global_rnd{round_num}.pth"
        models = wait_for_file_and_load(global_model_file)
        
        print(f"[{client_id}] Received Global Models. Training locally...")
        client.set_lstm_weights(models['lstm'])
        client.set_dqn_weights(models['dqn'])

        report = client.train_round()
        client.log_round(round_num, report)
        
        print(f"[{client_id}] Round {round_num} complete. Acc: {report['lstm_accuracy']:.3f}, DQN: {report['dqn_reward']:+.1f}, Col: {report['collision_rate']:.3f}")
        
        # Package weights and send to Edge (save local file)
        client_pkg = {
            'client_id': client_id,
            'report': report,
            'lstm_weights': client.get_lstm_weights(),
            'dqn_weights': client.get_dqn_weights()
        }
        
        out_file = SYNC_DIR / f"{client_id}_rnd{round_num}.pth"
        dict_save(client_pkg, out_file)
        print(f"[{client_id}] Sent weights to Edge {edge_id} -> {out_file.name}")

    client.save_history(log_dir)
    client.plot_metrics(plot_dir)
    print(f"[{client_id}] FL Training completely finished!")
