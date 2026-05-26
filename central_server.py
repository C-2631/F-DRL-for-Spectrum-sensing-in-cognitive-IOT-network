import os, sys, copy, time, json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

import torch
torch.backends.cudnn.benchmark = False
torch.set_num_threads(2)
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.55)

from fl_shared import *

class CentralServer:
    """Main PC (192.168.1.1) — global FedAvg, logging, checkpointing."""
    def __init__(self, server_ip='192.168.1.1'):
        self.server_ip = server_ip
        self.global_lstm = None
        self.global_dqn = None
        self.t0 = datetime.now()
        self.history = {
            'round': [], 'global_lstm_loss': [], 'global_lstm_acc': [],
            'global_dqn_reward': [], 'global_collision': [], 'total_samples': [],
        }
        self.edge_history = {}

    def aggregate_edges(self, pkgs, reps):
        self.global_lstm = fedavg([p['lstm_weights'] for p in pkgs],
                                  [r['total_samples'] for r in reps])
        self.global_dqn = fedavg([p['dqn_weights'] for p in pkgs],
                                 [r['total_samples'] for r in reps])
        return {
            'global_lstm_loss': np.mean([r['avg_lstm_loss'] for r in reps]),
            'global_lstm_acc': np.mean([r['avg_lstm_acc'] for r in reps]),
            'global_dqn_reward': np.mean([r['avg_dqn_reward'] for r in reps]),
            'global_collision': np.mean([r['avg_collision'] for r in reps]),
            'total_samples': sum(r['total_samples'] for r in reps),
            'edge_summaries': reps,
        }

    def log_round(self, rnd, rep):
        self.history['round'].append(rnd)
        self.history['global_lstm_loss'].append(rep['global_lstm_loss'])
        self.history['global_lstm_acc'].append(rep['global_lstm_acc'])
        self.history['global_dqn_reward'].append(rep['global_dqn_reward'])
        self.history['global_collision'].append(rep['global_collision'])
        self.history['total_samples'].append(rep['total_samples'])

        for es in rep['edge_summaries']:
            eid = es['edge_id']
            if eid not in self.edge_history:
                self.edge_history[eid] = {
                    'round': [], 'avg_lstm_acc': [], 'avg_dqn_reward': [], 'avg_collision': []
                }
            self.edge_history[eid]['round'].append(rnd)
            self.edge_history[eid]['avg_lstm_acc'].append(es['avg_lstm_acc'])
            self.edge_history[eid]['avg_dqn_reward'].append(es['avg_dqn_reward'])
            self.edge_history[eid]['avg_collision'].append(es['avg_collision'])

    def save_history(self, log_dir):
        payload = {'global': self.history, 'edges': self.edge_history}
        out_file = log_dir / 'central_history.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        print(f"[CENTRAL] History saved -> {out_file}")

    def plot_metrics(self, plot_dir):
        rounds = self.history['round']
        if not rounds:
            print('[CENTRAL] No rounds to plot.')
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        ax = axes.ravel()

        ax[0].plot(rounds, [x * 100 for x in self.history['global_lstm_acc']], marker='o', color='tab:blue')
        ax[0].set_title('Global LSTM Accuracy (%)')
        ax[0].set_xlabel('Round')
        ax[0].grid(alpha=0.3)

        ax[1].plot(rounds, self.history['global_lstm_loss'], marker='o', color='tab:red')
        ax[1].set_title('Global LSTM Loss')
        ax[1].set_xlabel('Round')
        ax[1].grid(alpha=0.3)

        ax[2].plot(rounds, self.history['global_dqn_reward'], marker='o', color='tab:green')
        ax[2].set_title('Global DQN Reward')
        ax[2].set_xlabel('Round')
        ax[2].grid(alpha=0.3)

        ax[3].plot(rounds, [x * 100 for x in self.history['global_collision']], marker='o', color='tab:purple')
        ax[3].set_title('Global Collision Rate (%)')
        ax[3].set_xlabel('Round')
        ax[3].grid(alpha=0.3)

        ax[4].plot(rounds, self.history['total_samples'], marker='o', color='tab:brown')
        ax[4].set_title('Total Samples per Round')
        ax[4].set_xlabel('Round')
        ax[4].grid(alpha=0.3)

        for eid, h in self.edge_history.items():
            ax[5].plot(h['round'], [x * 100 for x in h['avg_lstm_acc']], marker='o', label=eid)
        ax[5].set_title('Edge Accuracy Comparison (%)')
        ax[5].set_xlabel('Round')
        ax[5].legend()
        ax[5].grid(alpha=0.3)

        fig.tight_layout()
        out_main = plot_dir / 'central_metrics_overview.png'
        fig.savefig(out_main, dpi=100)
        plt.close(fig)

        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
        for eid, h in self.edge_history.items():
            ax2[0].plot(h['round'], h['avg_dqn_reward'], marker='o', label=eid)
            ax2[1].plot(h['round'], [x * 100 for x in h['avg_collision']], marker='o', label=eid)
        ax2[0].set_title('Edge DQN Reward Comparison')
        ax2[1].set_title('Edge Collision Comparison (%)')
        ax2[0].set_xlabel('Round')
        ax2[1].set_xlabel('Round')
        ax2[0].grid(alpha=0.3)
        ax2[1].grid(alpha=0.3)
        ax2[0].legend()
        ax2[1].legend()
        fig2.tight_layout()
        out_extra = plot_dir / 'central_edge_comparisons.png'
        fig2.savefig(out_extra, dpi=100)
        plt.close(fig2)
        print(f"[CENTRAL] Plots saved -> {out_main} and {out_extra}")

    def print_summary(self, rnd, total, rep):
        sep = '=' * 65
        print(f'\n{sep}')
        print(f"  CENTRAL SERVER ({self.server_ip}) — Round {rnd:02d}/{total}")
        print(sep)
        print(f"  LSTM Loss      : {rep['global_lstm_loss']:.4f}")
        print(f"  LSTM Accuracy  : {rep['global_lstm_acc']*100:.2f}%")
        print(f"  DQN Reward     : {rep['global_dqn_reward']:+.2f}")
        print(f"  Collision Rate : {rep['global_collision']*100:.2f}%")
        print(f"  Total Samples  : {rep['total_samples']:,}")
        for es in rep['edge_summaries']:
            print(f"    {es['edge_id']}: LSTM acc {es['avg_lstm_acc']:.3f} | DQN {es['avg_dqn_reward']:+.1f} | col {es['avg_collision']:.3f}")

if __name__ == "__main__":
    print("Initializing Central Server...")

    outputs_dir = Path('FL_DRL_Outputs_v2')
    log_dir = outputs_dir / 'logs'
    plot_dir = outputs_dir / 'plots'
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up previous simulation runs
    if SYNC_DIR.exists():
        print("Deleting existing 'sync_dir' for a fresh start...")
        shutil.rmtree(SYNC_DIR)
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    
    central = CentralServer()
    
    # Needs initial weights
    print("Generating Initial Random Weights...")
    lstm_init = LSTMSpectrumSensor().state_dict()
    dqn_init = DQNAgent(192 + NUM_CHANNELS + 1).get_weights()
    central.global_lstm = lstm_init
    central.global_dqn = dqn_init
    
    edges_ids = [e['id'] for e in TOPOLOGY['edges']]
    
    print(f"Central Server ready to train for {FL_COMM_ROUNDS} rounds.")
    print(f"Topology Edges: {edges_ids}")
    
    for round_num in range(1, FL_COMM_ROUNDS + 1):
        print(f"\n[ROUND {round_num}] Publishing Global Models...")
        central_out = SYNC_DIR / f"central_global_rnd{round_num}.pth"
        dict_save({'lstm': central.global_lstm, 'dqn': central.global_dqn}, central_out)
        
        # Wait for all edges
        print("Waiting for all Edge weights...")
        reps = []
        pkgs = []
        for eid in edges_ids:
            edge_file = SYNC_DIR / f"agg_{eid}_rnd{round_num}.pth"
            edge_pkg = wait_for_file_and_load(edge_file)
            
            reps.append(edge_pkg['report'])
            pkgs.append(edge_pkg)
            
        print("All Edge weights received. Aggregating globally...")
        g_rep = central.aggregate_edges(pkgs, reps)
        central.log_round(round_num, g_rep)
        central.print_summary(round_num, FL_COMM_ROUNDS, g_rep)
        
    print("Training completely finished. Final Global Model Saved.")
    final_out = SYNC_DIR / "central_global_final.pth"
    dict_save({'lstm': central.global_lstm, 'dqn': central.global_dqn}, final_out)

    central.save_history(log_dir)
    central.plot_metrics(plot_dir)
    
    print("To clean up the simulation, delete the 'sync_dir' folder.")
