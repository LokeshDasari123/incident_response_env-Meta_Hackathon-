"""
Diagnostic: Does the RL loop ACTUALLY increase complexity when the agent gets better?
Tests: skill rises -> rewards rise -> curriculum promotes -> params scale harder
"""
import sys, math
sys.path.insert(0, '.')

from training.curriculum import CurriculumController
from envs.multi_agent_env import MultiAgentIncidentEnv

GROUND_TRUTH = {
    "easy":   {"root_cause_service": "payments-db", "root_cause_type": "misconfiguration", "severity": "P0", "affected_services": ["payments-db","payments-api","checkout-ui"], "remediation_action": "fix_config"},
    "medium": {"root_cause_service": "user-service", "root_cause_type": "network_partition", "severity": "P1", "affected_services": ["user-service","auth-service","api-gateway","storefront-ui"], "remediation_action": "fix_config"},
    "hard":   {"root_cause_service": "payments-db", "root_cause_type": "memory_leak", "severity": "P0", "affected_services": ["payments-db","cache-service","order-service","api-gateway","storefront-ui"], "remediation_action": "restart_service"},
}

cc = CurriculumController(promote_threshold=0.65, promote_window=5)

print("=" * 120)
print("RL LOOP DIAGNOSTIC — Does complexity increase when agent is capable?")
print("=" * 120)
print(f"{'ep':>3} | {'diff':>6} | {'skill':>5} | {'reward':>6} | {'noise':>5} | {'adv_bgt':>7} | {'adv_cun':>7} | {'flt_bgt':>7} | {'flt_agg':>7} | {'mon_rel':>7} | {'inj':>3} | {'dec':>3}")
print("-" * 120)

for ep in range(30):
    task = cc.current_difficulty
    params = cc.get_env_params()
    gt = GROUND_TRUTH[task]

    env = MultiAgentIncidentEnv(
        monitor_reliability=params["monitor_reliability"],
        monitor_noise=params["monitor_noise"],
        fault_budget=params["fault_budget"],
        fault_aggression=params["fault_aggression"],
        adversary_budget=params["adversary_budget"],
        adversary_cunning=params["adversary_cunning"],
        seed=ep * 7,
    )

    # Skill increases with episode (simulating RL learning curve)
    skill = 1.0 / (1.0 + math.exp(-10 * (ep / 30 - 0.40)))
    ceiling = {"easy": 0.90, "medium": 0.78, "hard": 0.62}.get(task, 0.75)
    skill *= ceiling

    env.reset(task_id=task, seed=ep * 7, responder_skill=skill, ground_truth=gt)

    max_steps = {"easy": 10, "medium": 15, "hard": 20}[task]
    best_r = 0.0
    for step in range(1, min(max_steps + 1, 6)):
        _, reward, done, info = env.step()
        best_r = max(best_r, reward)
        if done:
            break

    cc.record_reward(best_r, ep)
    s = cc.state
    inj = len(env.injector.get_injections())
    dec = len(env.adversary.get_deceptions())

    print(
        f"{ep:3d} | {task:>6} | {skill:.3f} | {best_r:.3f}  | {s.noise_multiplier:.2f}  | "
        f"{s.adversary_budget:>7d} | {s.adversary_cunning:>7.3f} | {s.fault_budget:>7d} | "
        f"{s.fault_aggression:>7.3f} | {s.monitor_reliability:>7.3f} | {inj:3d} | {dec:3d}"
    )

    transitions = cc.get_transition_log()
    if transitions and transitions[-1].get("episode") == ep:
        t = transitions[-1]
        print(f"   >>> CURRICULUM {t['type'].upper()}: {t['from']} -> {t['to']} (avg_reward={t['avg_reward']:.3f})")

    env.close()

print()
print("=" * 80)
print("FINAL STATE")
print("=" * 80)
print(f"Final difficulty:  {cc.current_difficulty}")
print(f"Total transitions: {len(cc.get_transition_log())}")
for t in cc.get_transition_log():
    print(f"  {t['type']:>10}: {t['from']} -> {t['to']} at episode {t['episode']} (avg={t['avg_reward']:.3f})")
print(f"Intra-progress:    {cc._intra_progress:.3f}")
