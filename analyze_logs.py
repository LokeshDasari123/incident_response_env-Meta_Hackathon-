import json
from pathlib import Path

log_file = Path('data/training_logs/training_20260426_072639.jsonl')
episodes = []
with open(log_file) as f:
    for line in f:
        if line.strip():
            episodes.append(json.loads(line))

print(f"Total episodes: {len(episodes)}")
print("\n" + "="*70)
print("CHECKING IF LLM IS BEING USED (checking for CoT phases)")
print("="*70)

# Check first 3 episodes
for idx in [0, 1, 2, len(episodes)-1]:
    ep = episodes[idx]
    print(f"\nEpisode {ep['episode']}:")
    print(f"  Reward: {ep['reward']:.3f}")
    print(f"  Root Cause: {ep.get('root_cause', 'N/A')}")
    print(f"  Confidence: {ep.get('confidence', 'N/A')}")
    print(f"  Has cot_phases: {'cot_phases' in ep}")
    
    if 'cot_phases' in ep:
        for phase in ep.get('cot_phases', [])[:2]:
            print(f"    - {phase.get('phase')}: model={phase.get('model', 'N/A')[:40]}")

# Check RC accuracy
print(f"\n" + "="*70)
print("RC (ROOT CAUSE) ACCURACY ANALYSIS")
print("="*70)
rc_values = [ep.get('root_cause_correct', 0) for ep in episodes]
rc_rate = sum(rc_values) / len(rc_values) * 100 if rc_values else 0
print(f"Root cause correct rate: {rc_rate:.1f}%")
print(f"Episodes with RC=100%: {sum(1 for ep in episodes if ep.get('root_cause_correct') == 1)}")
print(f"Episodes with RC=0%: {sum(1 for ep in episodes if ep.get('root_cause_correct') == 0)}")

# Check task performance
print(f"\n" + "="*70)
print("TASK PERFORMANCE")
print("="*70)
for task in ['easy', 'medium', 'hard']:
    task_eps = [ep for ep in episodes if ep.get('task') == task]
    if task_eps:
        rewards = [ep['reward'] for ep in task_eps]
        rc_correct = sum(1 for ep in task_eps if ep.get('root_cause_correct') == 1)
        print(f"{task.upper():8} {len(task_eps):2} eps | avg={sum(rewards)/len(rewards):.3f} | RC correct: {rc_correct}/{len(task_eps)} ({rc_correct/len(task_eps)*100:.0f}%)")
