"""
ai/benchmark.py

Expanded 10-Graph MATLAB Benchmark Suite for Gesture AI Toolkit.
Generates 10 comprehensive, publication-quality MATLAB plots in result/:

1. 01_training_validation_curves.png (Epoch Loss & Accuracy)
2. 02_roc_curves.png (2x3 Subplot Grid for per-class ROC curves)
3. 03_precision_recall_curves.png (2x3 Subplot Grid for per-class PR curves)
4. 04_confusion_matrix.png (Raw Count & Normalized % Heatmaps)
5. 05_per_class_metrics_bar.png (Precision, Recall, F1 Bar Breakdown)
6. 06_inference_speed_benchmark.png (Inference Latency & Throughput FPS vs Batch Size)
7. 07_landmark_importance_sensitivity.png (Ranked Feature Importance for 21 Landmarks)
8. 08_model_calibration_reliability.png (Probability Reliability & Confidence Histogram)
9. 09_noise_robustness_curve.png (Landmark Perturbation & Jitter Robustness Curve)
10. 10_gesture_embedding_tsne.png (2D t-SNE Gesture Cluster Manifold Projection)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import GestureNet

# MATLAB default styling
plt.style.use('default')

MATLAB_COLORS = [
    '#0072BD',  # MATLAB Blue
    '#D95319',  # MATLAB Orange
    '#EDB120',  # MATLAB Yellow
    '#7E29B0',  # MATLAB Purple
    '#77AC30',  # MATLAB Green
    '#4DBEEE',  # MATLAB Cyan
    '#A2142F'   # MATLAB Dark Red
]

MATLAB_MARKERS = ['o', 's', '^', 'd', 'v', 'x', '*']

LANDMARK_NAMES = [
    "0: Wrist", "1: Thumb CMC", "2: Thumb MCP", "3: Thumb IP", "4: Thumb Tip",
    "5: Index MCP", "6: Index PIP", "7: Index DIP", "8: Index Tip",
    "9: Middle MCP", "10: Middle PIP", "11: Middle DIP", "12: Middle Tip",
    "13: Ring MCP", "14: Ring PIP", "15: Ring DIP", "16: Ring Tip",
    "17: Pinky MCP", "18: Pinky PIP", "19: Pinky DIP", "20: Pinky Tip"
]


def apply_matlab_axes(ax, title="", xlabel="", ylabel="", grid=True):
    """Applies classic MATLAB figure formatting."""
    ax.set_facecolor('white')
    ax.set_aspect('auto')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    if title:
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, fontweight='bold')

    if grid:
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    
    ax.tick_params(direction='in', top=True, right=True, labelsize=8.5)


class ExpandedMatlabBenchmarkSuite:
    """
    Evaluates GestureNet and generates 10 comprehensive MATLAB figures.
    """
    def __init__(self, dataset_path: str, model_path: str, output_dir: str = "result"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = LabelEncoder()
        
        self._load_data()
        self._load_model()

    def _load_data(self):
        df = pd.read_csv(self.dataset_path)
        X_raw = df.iloc[:, 1:].values
        y_raw = df.iloc[:, 0].values

        y_encoded = self.encoder.fit_transform(y_raw)
        self.classes = list(self.encoder.classes_)
        self.num_classes = len(self.classes)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_raw, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
        )

        self.X_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_bin = label_binarize(self.y_test, classes=range(self.num_classes))

    def _load_model(self):
        self.model = GestureNet(num_classes=self.num_classes)
        self.model.load_model(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def record_training_history(self, epochs: int = 50) -> dict:
        train_dataset = TensorDataset(torch.tensor(self.X_train, dtype=torch.float32), torch.tensor(self.y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32), torch.tensor(self.y_test, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        fresh_model = GestureNet(num_classes=self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(fresh_model.parameters(), lr=0.001)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            fresh_model.train()
            t_loss, t_corr, t_tot = 0.0, 0, 0
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = fresh_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                t_loss += loss.item() * bx.size(0)
                preds = torch.argmax(out, dim=1)
                t_corr += (preds == by).sum().item()
                t_tot += by.size(0)
            
            fresh_model.eval()
            v_loss, v_corr, v_tot = 0.0, 0, 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    out = fresh_model(bx)
                    loss = criterion(out, by)
                    v_loss += loss.item() * bx.size(0)
                    preds = torch.argmax(out, dim=1)
                    v_corr += (preds == by).sum().item()
                    v_tot += by.size(0)
                    
            history['train_loss'].append(t_loss / t_tot)
            history['val_loss'].append(v_loss / v_tot)
            history['train_acc'].append(t_corr / t_tot)
            history['val_acc'].append(v_corr / v_tot)

        return history

    def run_inference(self, jitter_sigma: float = 0.08):
        np.random.seed(42)
        noise = np.random.normal(0, jitter_sigma, self.X_test.shape)
        X_jittered = torch.tensor(self.X_test + noise, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(X_jittered)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        self.y_prob = probabilities
        self.y_pred = preds
        return probabilities, preds

    # 1. Training & Validation Loss & Accuracy Curves
    def plot_training_curves(self, history=None, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "01_training_validation_curves.png")

        if history is None:
            history = self.record_training_history(epochs=50)

        epochs = range(1, len(history["train_loss"]) + 1)
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]
        train_acc = [a * 100 if a <= 1.0 else a for a in history["train_acc"]]
        val_acc = [a * 100 if a <= 1.0 else a for a in history["val_acc"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

        axes[0].plot(epochs, train_loss, color=MATLAB_COLORS[0], linestyle='-', marker='o', markevery=4, markersize=4, label='Training Loss')
        axes[0].plot(epochs, val_loss, color=MATLAB_COLORS[1], linestyle='--', marker='s', markevery=4, markersize=4, label='Validation Loss')
        apply_matlab_axes(axes[0], title="Training vs Validation Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
        axes[0].legend(loc='upper right', fontsize=9, frameon=True, edgecolor='black', facecolor='white')

        axes[1].plot(epochs, train_acc, color=MATLAB_COLORS[0], linestyle='-', marker='o', markevery=4, markersize=4, label='Training Acc')
        axes[1].plot(epochs, val_acc, color=MATLAB_COLORS[1], linestyle='--', marker='s', markevery=4, markersize=4, label='Validation Acc')
        apply_matlab_axes(axes[1], title="Training vs Validation Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
        axes[1].legend(loc='lower right', fontsize=9, frameon=True, edgecolor='black', facecolor='white')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    # 2. Multi-Class ROC Subplot Grid
    def plot_roc_curves(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "02_roc_curves.png")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_bin[:, i], self.y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=300)
        axes = axes.flatten()

        for i in range(self.num_classes):
            c = MATLAB_COLORS[i % len(MATLAB_COLORS)]
            axes[i].plot(fpr[i], tpr[i], color=c, linewidth=2, label=f'AUC = {roc_auc[i]:.4f}')
            axes[i].plot([0, 1], [0, 1], 'r:', linewidth=1.2, label='Random')
            axes[i].set_xlim([-0.02, 1.02])
            axes[i].set_ylim([-0.02, 1.05])
            apply_matlab_axes(axes[i], title=f'ROC: Gesture "{self.classes[i]}"', xlabel='False Positive Rate', ylabel='True Positive Rate')
            axes[i].legend(loc='lower right', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')

        plt.suptitle("Multi-Class Receiver Operating Characteristic (ROC) Subplot Grid", fontsize=13, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

        return roc_auc

    # 3. Multi-Class Precision-Recall Subplot Grid
    def plot_precision_recall_curves(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "03_precision_recall_curves.png")

        precision = dict()
        recall = dict()
        ap_score = dict()

        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y_bin[:, i], self.y_prob[:, i])
            ap_score[i] = average_precision_score(self.y_bin[:, i], self.y_prob[:, i])

        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), dpi=300)
        axes = axes.flatten()

        for i in range(self.num_classes):
            c = MATLAB_COLORS[i % len(MATLAB_COLORS)]
            axes[i].plot(recall[i], precision[i], color=c, linewidth=2, label=f'AP = {ap_score[i]:.4f}')
            axes[i].set_xlim([-0.02, 1.02])
            axes[i].set_ylim([-0.02, 1.05])
            apply_matlab_axes(axes[i], title=f'PR: Gesture "{self.classes[i]}"', xlabel='Recall', ylabel='Precision')
            axes[i].legend(loc='lower left', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')

        plt.suptitle("Multi-Class Precision-Recall (PR) Subplot Grid", fontsize=13, fontweight='bold', y=0.99)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

        return ap_score

    # 4. Confusion Matrix Heatmap
    def plot_confusion_matrix(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "04_confusion_matrix.png")

        cm_raw = confusion_matrix(self.y_test, self.y_pred)
        cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=300)

        sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=self.classes, yticklabels=self.classes, cbar=True,
                    linewidths=0.5, linecolor='black',
                    annot_kws={"size": 9.5, "fontweight": "bold"})
        apply_matlab_axes(axes[0], title="Confusion Matrix (Instance Counts)",
                          xlabel="Predicted Gesture", ylabel="True Gesture", grid=False)

        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', ax=axes[1],
                    xticklabels=self.classes, yticklabels=self.classes, cbar=True,
                    linewidths=0.5, linecolor='black',
                    annot_kws={"size": 9.5, "fontweight": "bold"})
        apply_matlab_axes(axes[1], title="Confusion Matrix (Normalized %)",
                          xlabel="Predicted Gesture", ylabel="True Gesture", grid=False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    # 5. Per-Class Metrics Bar Chart
    def plot_per_class_metrics(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "05_per_class_metrics_bar.png")

        prec, rec, f1, support = precision_recall_fscore_support(self.y_test, self.y_pred, average=None)

        fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)
        x = np.arange(len(self.classes))
        width = 0.22

        ax.bar(x - width, prec, width, label='Precision', color=MATLAB_COLORS[0], edgecolor='black')
        ax.bar(x, rec, width, label='Recall', color=MATLAB_COLORS[1], edgecolor='black')
        ax.bar(x + width, f1, width, label='F1-Score', color=MATLAB_COLORS[2], edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(self.classes, fontsize=9.5, fontweight='bold')
        ax.set_ylim([0, 1.15])
        apply_matlab_axes(ax, title="Per-Class Performance Metrics Breakdown", xlabel="Gesture Class", ylabel="Score (0.0 - 1.0)")

        for i in range(len(self.classes)):
            ax.text(i, f1[i] + 0.02, f"{f1[i]:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.legend(loc='lower right', fontsize=9, frameon=True, edgecolor='black', facecolor='white')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    # 6. CPU Hardware Inference Speed & Throughput Benchmark
    def plot_inference_speed(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "06_inference_speed_benchmark.png")

        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        latencies_per_sample = []
        throughputs = []

        dummy_feature = torch.randn(1, 63).to(self.device)
        for _ in range(50):
            _ = self.model(dummy_feature)

        for b in batch_sizes:
            input_batch = torch.randn(b, 63).to(self.device)
            times = []
            for _ in range(200):
                start = time.perf_counter()
                _ = self.model(input_batch)
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

            avg_batch_time = np.mean(times)
            per_sample_latency = avg_batch_time / b
            throughput = (b * 1000.0) / avg_batch_time

            latencies_per_sample.append(per_sample_latency)
            throughputs.append(throughput)

        fig, ax1 = plt.subplots(figsize=(9.5, 5), dpi=300)

        c1 = MATLAB_COLORS[1]
        ax1.plot(batch_sizes, latencies_per_sample, 'o-', color=c1, linewidth=2, markersize=6, label='Latency (ms/sample)')
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(batch_sizes)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        apply_matlab_axes(ax1, title="CPU Inference Latency & Throughput Benchmark",
                           xlabel="Batch Size", ylabel="Latency per Sample (ms)")
        ax1.yaxis.label.set_color(c1)
        ax1.tick_params(axis='y', labelcolor=c1)

        ax2 = ax1.twinx()
        c2 = MATLAB_COLORS[0]
        ax2.plot(batch_sizes, throughputs, 's--', color=c2, linewidth=2, markersize=6, label='Throughput (FPS)')
        ax2.set_ylabel("Throughput (Frames / sec)", fontsize=10, fontweight='bold', color=c2)
        ax2.tick_params(axis='y', labelcolor=c2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    # 7. Hand Landmark Feature Importance & Motion Sensitivity [NEW]
    def plot_landmark_importance(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "07_landmark_importance_sensitivity.png")

        # Gradient-based Loss Sensitivity across all 21 MediaPipe hand landmarks (groups of 3 coordinates x,y,z)
        X_in = torch.tensor(self.X_test, dtype=torch.float32, requires_grad=True).to(self.device)
        y_in = torch.tensor(self.y_test, dtype=torch.long).to(self.device)

        logits = self.model(X_in)
        loss = nn.CrossEntropyLoss()(logits, y_in)
        loss.backward()

        grads = X_in.grad.abs().mean(dim=0).cpu().numpy()
        lm_importance = np.array([grads[i*3:(i+1)*3].sum() for i in range(21)])

        # Scale relative importance percentage
        rel_importance = (lm_importance / np.max(lm_importance)) * 100.0
        sorted_idx = np.argsort(rel_importance)

        fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
        y_pos = np.arange(21)

        colors = []
        for i in sorted_idx:
            name = LANDMARK_NAMES[i]
            if "Tip" in name:
                colors.append(MATLAB_COLORS[0])  # Blue for Tip
            elif "MCP" in name or "CMC" in name:
                colors.append(MATLAB_COLORS[1])  # Orange for MCP/CMC
            elif "Wrist" in name:
                colors.append(MATLAB_COLORS[3])  # Purple for Wrist
            else:
                colors.append(MATLAB_COLORS[4])  # Green for PIP/DIP joints

        bars = ax.barh(y_pos, rel_importance[sorted_idx], color=colors, edgecolor='black', height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([LANDMARK_NAMES[i] for i in sorted_idx], fontsize=8.5, fontweight='bold')
        ax.set_xlim([0, 115])
        apply_matlab_axes(ax, title="21 MediaPipe Hand Landmark Feature Sensitivity Ranking",
                          xlabel="Relative Feature Importance Score (%)", ylabel="Landmark Joint")

        for bar, val in zip(bars, rel_importance[sorted_idx]):
            ax.text(val + 1.2, bar.get_y() + bar.get_height()/2.0, f"{val:.1f}%",
                    va='center', fontsize=7.5, fontweight='bold')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=MATLAB_COLORS[0], edgecolor='black', label='Fingertip Landmarks'),
            Patch(facecolor=MATLAB_COLORS[1], edgecolor='black', label='MCP / Base Joints'),
            Patch(facecolor=MATLAB_COLORS[4], edgecolor='black', label='PIP / DIP Interphalangeal Joints'),
            Patch(facecolor=MATLAB_COLORS[3], edgecolor='black', label='Wrist Landmark')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    # 8. Model Confidence Calibration & Reliability Diagram [NEW]
    def plot_confidence_calibration(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "08_model_calibration_reliability.png")

        top_probs = np.max(self.y_prob, axis=1)
        correct_mask = (self.y_pred == self.y_test)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

        # Reliability Diagram
        bins = np.linspace(0, 1.0, 11)
        bin_accs = []
        for i in range(len(bins) - 1):
            mask = (top_probs >= bins[i]) & (top_probs < bins[i+1])
            bin_accs.append(np.mean(correct_mask[mask]) if np.sum(mask) > 0 else 0)

        axes[0].plot([0, 1], [0, 1], 'r--', label='Ideal Calibration', linewidth=1.5)
        axes[0].bar(bins[:-1], bin_accs, width=0.08, align='edge', color=MATLAB_COLORS[0],
                    edgecolor='black', alpha=0.8, label='GestureNet')
        apply_matlab_axes(axes[0], title="Probability Reliability Diagram",
                           xlabel="Predicted Top-1 Confidence", ylabel="Empirical Accuracy")
        axes[0].legend(loc='upper left', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')

        # Confidence Histogram
        axes[1].hist(top_probs[correct_mask], bins=15, alpha=0.7, color=MATLAB_COLORS[4], label='Correct Predictions', edgecolor='black')
        axes[1].hist(top_probs[~correct_mask], bins=15, alpha=0.7, color=MATLAB_COLORS[6], label='Incorrect Predictions', edgecolor='black')
        apply_matlab_axes(axes[1], title="Top-1 Confidence Probability Distribution",
                           xlabel="Max Class Confidence", ylabel="Sample Count")
        axes[1].legend(loc='upper left', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    # 9. Landmark Noise & Camera Jitter Robustness Curve [NEW]
    def plot_noise_robustness(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "09_noise_robustness_curve.png")

        sigmas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        accuracies = []

        np.random.seed(42)
        for sig in sigmas:
            noise = np.random.normal(0, sig, self.X_test.shape)
            inputs = torch.tensor(self.X_test + noise, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = torch.argmax(self.model(inputs), dim=1).cpu().numpy()
            acc = accuracy_score(self.y_test, preds) * 100.0
            accuracies.append(acc)

        fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)
        ax.plot(sigmas, accuracies, color=MATLAB_COLORS[0], linestyle='-', marker='o', linewidth=2, markersize=6, label='GestureNet Accuracy')
        
        apply_matlab_axes(ax, title="Landmark Coordinate Noise & Camera Jitter Robustness",
                          xlabel="Gaussian Noise Std Dev (Sigma)", ylabel="Validation Accuracy (%)")
        ax.set_ylim([60, 103])

        ax.annotate(f'Clean Baseline: {accuracies[0]:.1f}%', xy=(0, accuracies[0]),
                    xytext=(0.02, 68), arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=8.5, bbox=dict(boxstyle='square,pad=0.3', facecolor='#ffffcc', edgecolor='black'))

        ax.legend(loc='lower left', fontsize=9, frameon=True, edgecolor='black', facecolor='white')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

        return dict(zip(sigmas, accuracies))

    # 10. 2D t-SNE Gesture Feature Space Cluster Projection [NEW]
    def plot_tsne_clusters(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.output_dir, "10_gesture_embedding_tsne.png")

        print("Computing 2D t-SNE feature manifold projection...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_embedded = tsne.fit_transform(self.X_test)

        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)

        for i in range(self.num_classes):
            mask = (self.y_test == i)
            c = MATLAB_COLORS[i % len(MATLAB_COLORS)]
            m = MATLAB_MARKERS[i % len(MATLAB_MARKERS)]
            scatter_kwargs = {'color': c, 'marker': m, 's': 30, 'alpha': 0.8, 'label': f'Class: {self.classes[i]}'}
            if m not in ['x', '*', '+']:
                scatter_kwargs['edgecolors'] = 'black'
                scatter_kwargs['linewidths'] = 0.3
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], **scatter_kwargs)

        apply_matlab_axes(ax, title="2D t-SNE Feature Space Manifold Projection of 21 Hand Landmarks",
                          xlabel="t-SNE Dimension 1", ylabel="t-SNE Dimension 2")

        ax.legend(loc='upper right', fontsize=8.5, frameon=True, edgecolor='black', facecolor='white')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved MATLAB plot: {save_path}")

    def generate_all_plots(self):
        print("\n=== Generating 10 Comprehensive MATLAB Benchmark Visualizations ===")
        
        self.run_inference(jitter_sigma=0.08)

        self.plot_training_curves()
        roc_auc = self.plot_roc_curves()
        ap_scores = self.plot_precision_recall_curves()
        self.plot_confusion_matrix()
        self.plot_per_class_metrics()
        self.plot_inference_speed()
        self.plot_landmark_importance()
        self.plot_confidence_calibration()
        noise_dict = self.plot_noise_robustness()
        self.plot_tsne_clusters()

        overall_acc = accuracy_score(self.y_test, self.y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(self.y_test, self.y_pred, average='macro')

        report = {
            "total_benchmark_graphs": 10,
            "overall_accuracy": float(overall_acc),
            "macro_averages": {
                "precision": float(macro_p),
                "recall": float(macro_r),
                "f1_score": float(macro_f1)
            },
            "roc_auc_scores": {k: float(v) for k, v in roc_auc.items()},
            "average_precision_scores": {k: float(v) for k, v in ap_scores.items()},
            "noise_robustness_sigmas": {str(k): float(v) for k, v in noise_dict.items()}
        }

        report_json_path = os.path.join(self.output_dir, "benchmark_report.json")
        with open(report_json_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Saved numerical benchmark summary report to {report_json_path}\n")


def main():
    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "gesture_data.csv")
    model_path = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
    output_dir = os.path.join(PROJECT_ROOT, "result")

    suite = ExpandedMatlabBenchmarkSuite(dataset_path=dataset_path, model_path=model_path, output_dir=output_dir)
    suite.generate_all_plots()


if __name__ == "__main__":
    main()
