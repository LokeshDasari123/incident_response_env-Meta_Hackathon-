#!/usr/bin/env python3
"""
Generate training visualization plots from JSONL logs
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_latest_log():
    """Find and load the latest training log"""
    log_dir = Path("data/training_logs")
    logs = sorted(log_dir.glob("training_*.jsonl"))
    if not logs:
        print("❌ No training logs found in data/training_logs/")
        return None, None
    
    latest = logs[-1]
    print(f"📊 Loading: {latest.name}")
    
    episodes = []
    with open(latest) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    
    return episodes, latest.stem

def plot_reward_curve(episodes, log_name):
    """Plot reward over episodes"""
    ep_nums = [e['episode'] for e in episodes]
    rewards = [e['reward'] for e in episodes]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Raw rewards
    ax.scatter(ep_nums, rewards, alpha=0.5, s=30, label='Episode Reward', color='blue')
    
    # Moving average (window=5)
    if len(rewards) >= 5:
        mavg = np.convolve(rewards, np.ones(5)/5, mode='valid')
        ax.plot(range(2, len(rewards)-2), mavg, 'r-', linewidth=2, label='5-Episode Moving Avg')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Training Reward Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = f"data/training_logs/{log_name}_reward_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()

def plot_task_performance(episodes, log_name):
    """Plot performance by task difficulty"""
    task_rewards = defaultdict(list)
    for e in episodes:
        task = e.get('task', e.get('task_id', 'unknown'))
        task_rewards[task].append(e['reward'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tasks = ['easy', 'medium', 'hard', 'expert']
    task_list = [t for t in tasks if t in task_rewards]
    avgs = [np.mean(task_rewards[t]) for t in task_list]
    bests = [np.max(task_rewards[t]) for t in task_list]
    
    x = np.arange(len(task_list))
    width = 0.35
    
    ax.bar(x - width/2, avgs, width, label='Average Reward', color='skyblue')
    ax.bar(x + width/2, bests, width, label='Best Reward', color='orange')
    
    ax.set_xlabel('Task Difficulty', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Performance by Task Difficulty', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plot_path = f"data/training_logs/{log_name}_task_performance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()

def plot_challenger_stats(episodes, log_name):
    """Plot challenger win rate over time"""
    ep_nums = []
    challenger_win_rates = []
    
    window = 5
    for i in range(len(episodes) - window + 1):
        window_episodes = episodes[i:i+window]
        wins = sum(1 for e in window_episodes if e.get('reward', 0) > 0.75)
        ep_nums.append(window_episodes[-1]['episode'])
        challenger_win_rates.append(wins / window * 100)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ep_nums, challenger_win_rates, 'o-', linewidth=2, markersize=6, color='red')
    ax.fill_between(ep_nums, challenger_win_rates, alpha=0.3, color='red')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Challenger Win Rate (%)', fontsize=12)
    ax.set_title(f'Challenger Success Rate (5-Episode Window)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plot_path = f"data/training_logs/{log_name}_challenger_stats.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()

def print_summary(episodes):
    """Print training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    task_rewards = defaultdict(list)
    total_challenger_wins = 0
    
    for e in episodes:
        task = e.get('task', e.get('task_id', 'unknown'))
        task_rewards[task].append(e['reward'])
        total_challenger_wins += e.get('challenger_wins', 0)
    
    for task in ['easy', 'medium', 'hard', 'expert']:
        if task in task_rewards:
            rewards = task_rewards[task]
            print(f"\n{task.upper()}:")
            print(f"  Episodes: {len(rewards)}")
            print(f"  Avg Reward: {np.mean(rewards):.3f}")
            print(f"  Best: {np.max(rewards):.3f}")
            print(f"  Worst: {np.min(rewards):.3f}")
    
    print(f"\nTotal Challenger Wins: {total_challenger_wins}")
    print("="*60 + "\n")

if __name__ == "__main__":
    episodes, log_name = load_latest_log()
    
    if episodes:
        print_summary(episodes)
        plot_reward_curve(episodes, log_name)
        plot_task_performance(episodes, log_name)
        plot_challenger_stats(episodes, log_name)
        print("\n✅ All plots generated!")
    else:
        print("❌ Failed to load training logs")
