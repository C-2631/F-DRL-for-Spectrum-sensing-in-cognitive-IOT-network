import os, sys, copy, time
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark = False
torch.set_num_threads(2)
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)

from fl_shared import *

class FedEdgeServer:
    """Edge device (192.168.1.11 / .12) — aggregates its clients, syncs mesh."""
    def __init__(self, edge_id, client_ids):
        self.edge_id = edge_id
        self.client_ids = client_ids
        self.peers = []
        self.lstm_weights = None
        self.dqn_weights = None
        self.history = {
            'round': [], 'avg_lstm_loss': [], 'avg_lstm_acc': [],
            'avg_dqn_reward': [], 'avg_collision': [], 'divergence': [], 'total_samples': [],
        }

    def set_peers(self, peer_list):
        self.peers = peer_list

    def aggregate_clients(self, reports, lstm_wts, dqn_wts):
        counts = [r['num_samples'] for r in reports]
        self.lstm_weights = fedavg(lstm_wts, counts)
        self.dqn_weights = fedavg(dqn_wts, counts)
        return {
            'edge_id': self.edge_id,
            'num_clients': len(reports),
            'total_samples': sum(counts),
            'avg_lstm_loss': np.mean([r['lstm_loss'] for r in reports]),
            'avg_lstm_acc': np.mean([r['lstm_accuracy'] for r in reports]),
            'avg_dqn_reward': np.mean([r['dqn_reward'] for r in reports]),
            'avg_collision': np.mean([r['collision_rate'] for r in reports]),
        }

    def log_round(self, rnd, rep):
        self.history['round'].append(rnd)
        self.history['avg_lstm_loss'].append(rep['avg_lstm_loss'])
        self.history['avg_lstm_acc'].append(rep['avg_lstm_acc'])
        self.history['avg_dqn_reward'].append(rep['avg_dqn_reward'])
        self.history['avg_collision'].append(rep['avg_collision'])
        self.history['total_samples'].append(rep['total_samples'])
        self.history['divergence'].append(0.0)

    def save_history(self, log_dir):
        out_file = log_dir / f"edge_{self.edge_id}_history.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print(f"[{self.edge_id}] History saved -> {out_file}")

    def plot_metrics(self, plot_dir):
        rounds = self.history['round']
        if not rounds:
            print(f"[{self.edge_id}] No rounds to plot.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes.ravel()

        ax[0].plot(rounds, [x * 100 for x in self.history['avg_lstm_acc']], marker='o', color='tab:blue')
        ax[0].set_title(f'{self.edge_id}: LSTM Accuracy (%)')
        ax[0].set_xlabel('Round')
        ax[0].grid(alpha=0.3)

        ax[1].plot(rounds, self.history['avg_lstm_loss'], marker='o', color='tab:red')
        ax[1].set_title(f'{self.edge_id}: LSTM Loss')
        ax[1].set_xlabel('Round')
        ax[1].grid(alpha=0.3)

        ax[2].plot(rounds, self.history['avg_dqn_reward'], marker='o', color='tab:green')
        ax[2].set_title(f'{self.edge_id}: DQN Reward')
        ax[2].set_xlabel('Round')
        ax[2].grid(alpha=0.3)

        ax[3].plot(rounds, [x * 100 for x in self.history['avg_collision']], marker='o', color='tab:purple')
        ax[3].set_title(f'{self.edge_id}: Collision Rate (%)')
        ax[3].set_xlabel('Round')
        ax[3].grid(alpha=0.3)

        ax[4].plot(rounds, self.history['total_samples'], marker='o', color='tab:brown')
        ax[4].set_title(f'{self.edge_id}: Total Samples')
        ax[4].set_xlabel('Round')
        ax[4].grid(alpha=0.3)

        combined = []
        for a, r, c in zip(self.history['avg_lstm_acc'], self.history['avg_dqn_reward'], self.history['avg_collision']):
            combined.append((0.6 * a) + (0.3 * max(r, 0.0) / (abs(r) + 1.0)) + (0.1 * (1.0 - c)))
        ax[5].plot(rounds, [x * 100 for x in combined], marker='o', color='tab:orange')
        ax[5].set_title(f'{self.edge_id}: Combined Score (%)')
        ax[5].set_xlabel('Round')
        ax[5].grid(alpha=0.3)

        fig.tight_layout()
        out_file = plot_dir / f"edge_{self.edge_id}_metrics_overview.png"
        fig.savefig(out_file, dpi=100)
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4.5))
        if len(rounds) >= 3:
            win = 3
            smooth = np.convolve(self.history['avg_lstm_acc'], np.ones(win) / win, mode='valid')
            ax2.plot(rounds[win - 1:], [x * 100 for x in smooth], marker='o', color='tab:cyan', label='LSTM acc MA(3)')
        ax2.plot(rounds, [x * 100 for x in self.history['avg_lstm_acc']], alpha=0.4, color='tab:blue', label='LSTM acc raw')
        ax2.set_title(f'{self.edge_id}: Accuracy Trend (Raw + Moving Avg)')
        ax2.set_xlabel('Round')
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        out_file2 = plot_dir / f"edge_{self.edge_id}_accuracy_trend.png"
        fig2.savefig(out_file2, dpi=100)
        plt.close(fig2)
        print(f"[{self.edge_id}] Plots saved -> {out_file} and {out_file2}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python edge_server.py <edge_id>")
        print("Example: python edge_server.py edge_1")
        sys.exit(1)

    edge_id = sys.argv[1]

    outputs_dir = Path('FL_DRL_Outputs_v2')
    log_dir = outputs_dir / 'logs'
    plot_dir = outputs_dir / 'plots'
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify which clients belong to this edge
    clients = [c for c in TOPOLOGY['edges'] if c['id'] == edge_id][0]['clients']
    
    edge = FedEdgeServer(edge_id, clients)
    
    print(f"[{edge_id}] Edge Server Initialized. Awaiting clients: {clients}")
    
    for round_num in range(1, FL_COMM_ROUNDS + 1):
        print(f"\n[{edge_id}] Waiting for Global Model Round {round_num} from Central Server...")
        
        # Read from Central
        central_file = SYNC_DIR / f"central_global_rnd{round_num}.pth"
        models = wait_for_file_and_load(central_file)
        
        edge.lstm_weights = models['lstm']
        edge.dqn_weights = models['dqn']
        
        # Broadcast to local clients
        print(f"[{edge_id}] Broadcasting models to clients: {clients}")
        out_file = SYNC_DIR / f"{edge_id}_global_rnd{round_num}.pth"
        dict_save({'lstm': edge.lstm_weights, 'dqn': edge.dqn_weights}, out_file)
        
        # Wait for clients
        print(f"[{edge_id}] Waiting for client weights...")
        reports = []
        lstm_wts = []
        dqn_wts = []
        for cid in clients:
            client_file = SYNC_DIR / f"{cid}_rnd{round_num}.pth"
            pkg = wait_for_file_and_load(client_file)
            
            reports.append(pkg['report'])
            lstm_wts.append(pkg['lstm_weights'])
            dqn_wts.append(pkg['dqn_weights'])
            
        print(f"[{edge_id}] Received all client weights. Aggregating locally...")
        er = edge.aggregate_clients(reports, lstm_wts, dqn_wts)
        edge.log_round(round_num, er)
        
        print(f"[{edge_id}] Accuracy: {er['avg_lstm_acc']:.3f} | DQN: {er['avg_dqn_reward']:+.1f}")
        
        # Send aggregated weights to Central
        edge_pkg = {
            'edge_id': edge_id,
            'report': er,
            'lstm_weights': edge.lstm_weights,
            'dqn_weights': edge.dqn_weights
        }
        edge_out = SYNC_DIR / f"agg_{edge_id}_rnd{round_num}.pth"
        dict_save(edge_pkg, edge_out)
        print(f"[{edge_id}] Sent aggregated weights to Central -> {edge_out.name}")

    edge.save_history(log_dir)
    edge.plot_metrics(plot_dir)
    print(f"[{edge_id}] Edge Server finished successfully!")
