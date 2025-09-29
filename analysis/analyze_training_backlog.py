#!/usr/bin/env python3
"""
Training Backlog Analysis Script for Experiment 03
Analyzes DQN training episodes and creates visualizations
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# Add project root to path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Use non-interactive backend for headless PDF generation
import matplotlib
matplotlib.use("Agg")


def load_training_backlog(json_path: str) -> Dict[str, Any]:
    """Load training backlog from JSON file"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def analyze_training_episodes(backlog: List[Dict]) -> Dict[str, Any]:
    """Analyze training episodes and return statistics"""
    if not backlog:
        return {}
    
    episodes = len(backlog)
    total_steps = sum(len(ep["training_steps"]) for ep in backlog)
    
    # Extract metrics over episodes
    episode_rewards = [ep["total_reward"] for ep in backlog]
    edge_ratios = [ep["edge_action_ratio"] for ep in backlog]
    cloud_ratios = [ep["cloud_action_ratio"] for ep in backlog]
    local_ratios = [ep["local_action_ratio"] for ep in backlog]
    success_rates = [ep["success_rate"] for ep in backlog]
    avg_latencies = [ep["avg_latency"] for ep in backlog]
    avg_energies = [ep["avg_energy"] for ep in backlog]
    epsilons = [ep["training_steps"][-1]["epsilon"] for ep in backlog]
    
    # Calculate trends
    reward_trend = np.polyfit(range(episodes), episode_rewards, 1)[0] if episodes > 1 else 0
    edge_trend = np.polyfit(range(episodes), edge_ratios, 1)[0] if episodes > 1 else 0
    success_trend = np.polyfit(range(episodes), success_rates, 1)[0] if episodes > 1 else 0
    
    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "episode_rewards": episode_rewards,
        "edge_ratios": edge_ratios,
        "cloud_ratios": cloud_ratios,
        "local_ratios": local_ratios,
        "success_rates": success_rates,
        "avg_latencies": avg_latencies,
        "avg_energies": avg_energies,
        "epsilons": epsilons,
        "trends": {
            "reward_trend": reward_trend,
            "edge_trend": edge_trend,
            "success_trend": success_trend,
        },
        "final_stats": {
            "final_reward": episode_rewards[-1] if episode_rewards else 0,
            "final_edge_ratio": edge_ratios[-1] if edge_ratios else 0,
            "final_success_rate": success_rates[-1] if success_rates else 0,
            "final_epsilon": epsilons[-1] if epsilons else 0,
        }
    }


def plot_training_analysis(analysis: Dict[str, Any], output_path: str = "training_analysis.pdf"):
    """Create training analysis plots"""
    if not analysis or analysis["episodes"] == 0:
        print("No training data to plot")
        return
    
    episodes = list(range(1, analysis["episodes"] + 1))
    
    with plt.style.context('default'):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("DQN Training Analysis - Experiment 03", fontsize=16)
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(episodes, analysis["episode_rewards"], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Action Distribution
        axes[0, 1].plot(episodes, analysis["edge_ratios"], 'g-o', label='Edge', linewidth=2, markersize=4)
        axes[0, 1].plot(episodes, analysis["cloud_ratios"], 'r-o', label='Cloud', linewidth=2, markersize=4)
        axes[0, 1].plot(episodes, analysis["local_ratios"], 'b-o', label='Local', linewidth=2, markersize=4)
        axes[0, 1].set_title("Action Distribution")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Action Ratio")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Success Rate
        axes[0, 2].plot(episodes, analysis["success_rates"], 'purple', marker='o', linewidth=2, markersize=4)
        axes[0, 2].set_title("Success Rate")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Success Rate")
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Average Latency
        axes[1, 0].plot(episodes, analysis["avg_latencies"], 'orange', marker='o', linewidth=2, markersize=4)
        axes[1, 0].set_title("Average Latency")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Latency (s)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Average Energy
        axes[1, 1].plot(episodes, analysis["avg_energies"], 'brown', marker='o', linewidth=2, markersize=4)
        axes[1, 1].set_title("Average Energy")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Energy (J)")
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Epsilon Decay
        axes[1, 2].plot(episodes, analysis["epsilons"], 'red', marker='o', linewidth=2, markersize=4)
        axes[1, 2].set_title("Epsilon Decay")
        axes[1, 2].set_xlabel("Episode")
        axes[1, 2].set_ylabel("Epsilon")
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training analysis plots saved to: {output_path}")


def print_training_summary(analysis: Dict[str, Any]):
    """Print training summary statistics"""
    if not analysis or analysis["episodes"] == 0:
        print("No training data available")
        return
    
    print("\n" + "="*60)
    print("DQN TRAINING SUMMARY - EXPERIMENT 03")
    print("="*60)
    print(f"Total Episodes: {analysis['episodes']}")
    print(f"Total Training Steps: {analysis['total_steps']}")
    print(f"Steps per Episode: {analysis['total_steps'] // analysis['episodes']}")
    
    print(f"\nFINAL STATISTICS:")
    print(f"  Final Reward: {analysis['final_stats']['final_reward']:.3f}")
    print(f"  Final Edge Ratio: {analysis['final_stats']['final_edge_ratio']:.1%}")
    print(f"  Final Success Rate: {analysis['final_stats']['final_success_rate']:.1%}")
    print(f"  Final Epsilon: {analysis['final_stats']['final_epsilon']:.3f}")
    
    print(f"\nTRENDS (slope):")
    print(f"  Reward Trend: {analysis['trends']['reward_trend']:.3f} {'(improving)' if analysis['trends']['reward_trend'] > 0 else '(declining)'}")
    print(f"  Edge Action Trend: {analysis['trends']['edge_trend']:.3f} {'(increasing)' if analysis['trends']['edge_trend'] > 0 else '(decreasing)'}")
    print(f"  Success Rate Trend: {analysis['trends']['success_trend']:.3f} {'(improving)' if analysis['trends']['success_trend'] > 0 else '(declining)'}")
    
    print(f"\nEPISODE BREAKDOWN:")
    for i, ep in enumerate(analysis['episode_rewards'], 1):
        print(f"  Episode {i:2d}: Reward={ep:7.3f}, Edge={analysis['edge_ratios'][i-1]:.1%}, Success={analysis['success_rates'][i-1]:.1%}")


def main():
    """Main analysis function"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_backlog.py <summary_json_path>")
        print("Example: python analyze_training_backlog.py data/test_exp03_summary.json")
        return
    
    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found")
        return
    
    print(f"Loading training backlog from: {json_path}")
    data = load_training_backlog(json_path)
    
    if "training_backlog" not in data:
        print("Error: No training backlog found in the JSON file")
        return
    
    backlog = data["training_backlog"]
    print(f"Found {len(backlog)} training episodes")
    
    # Analyze training data
    analysis = analyze_training_episodes(backlog)
    
    # Print summary
    print_training_summary(analysis)
    
    # Create plots
    output_dir = os.path.dirname(json_path)
    plot_path = os.path.join(output_dir, "training_analysis.pdf")
    plot_training_analysis(analysis, plot_path)
    
    # Save detailed analysis
    analysis_path = os.path.join(output_dir, "training_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Detailed analysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()
