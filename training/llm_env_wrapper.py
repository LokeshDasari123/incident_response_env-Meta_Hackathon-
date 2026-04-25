"""
training/llm_env_wrapper.py
---------------------------
LLM-Environment Bridge for GRPO training.
Converts observations to rich prompts and parses LLM completions
into IncidentAction objects with composite reward shaping.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple


class LLMEnvWrapper:
    """
    Bridges the multi-agent environment with LLM training.

    Responsibilities:
    - observation_to_prompt(): converts obs + memory + messages to text
    - parse_response(): extracts IncidentAction from LLM output
    - shape_reward(): adds format/reasoning/memory bonuses to env reward
    """

    SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE triaging a production incident in a multi-agent system.

    You have access to:
    - ALERTS: active monitoring alerts
    - METRICS: real-time service metrics
    - TOPOLOGY: service dependency call graph
    - TIMELINE: chronological incident events
    - MONITOR SIGNALS: anomaly reports from the monitoring agent
    - MEMORY: your investigation history from previous steps

    APPROACH:
    1. Read the call graph topology — traverse INWARD from edge services
    2. The root cause is the DEEPEST service showing abnormal metrics
    3. Cross-reference monitor agent signals with your own analysis
    4. Use your memory to avoid repeating failed hypotheses
    5. List ONLY services in the cascade chain — NEVER include red herrings

    FAULT TYPE RULES:
    - misconfiguration = config_change in timeline + RT spike immediately after
    - memory_leak = memory_utilization > 0.85 AND restart_count > 0
    - network_partition = providerRPC_MCR drops to 0 or near-zero
    - crash_loop = restart_count > 2 AND service cycling

    SEVERITY RULES:
    - P0 = cascading failure 2+ services OR payment/checkout impacted
    - P1 = single service degraded with user-facing impact
    - P2 = internal service issue, no user impact
    - P3 = monitoring noise only

    REMEDIATION: config_change → fix_config | memory_leak → restart_service |
                 network_partition → fix_config | crash_loop → restart_service

    STAKEHOLDER MESSAGE: ALWAYS include for P0/P1 with service name + impact + ETA.

    Respond ONLY with valid JSON:
    {
      "root_cause_service": "<exact service name>",
      "root_cause_type": "<fault type>",
      "severity": "<P0|P1|P2|P3>",
      "affected_services": ["<cascade chain only>"],
      "remediation_action": "<action>",
      "stakeholder_message": "<required for P0/P1>",
      "confidence": <0.0-1.0>,
      "reasoning": "<step by step analysis>"
    }
    """).strip()

    def observation_to_prompt(
        self,
        obs: Dict[str, Any],
        memory_summary: str = "",
        monitor_signals: Optional[List[Dict]] = None,
        agent_messages: Optional[List[Dict]] = None,
    ) -> str:
        """Convert observation + context to a rich LLM prompt."""
        alerts   = obs.get("alerts", [])
        metrics  = obs.get("metrics", {})
        topology = obs.get("topology", [])
        timeline = obs.get("timeline", [])
        task_id  = obs.get("task_id", "unknown")
        step     = obs.get("step", 0)
        max_steps = obs.get("max_steps", 10)
        tp       = obs.get("time_pressure", 0.0)
        sla_in   = obs.get("sla_breach_in_steps")
        score    = obs.get("current_score", 0.0)

        # SLA warning
        sla_warn = ""
        if sla_in is not None and sla_in <= 3:
            sla_warn = f"\n⚠️ SLA BREACH IN {sla_in} STEPS — ACT NOW\n"

        # Monitor signals section
        monitor_section = ""
        if monitor_signals:
            monitor_section = "\n=== MONITOR AGENT SIGNALS ===\n"
            for sig in monitor_signals[-3:]:
                monitor_section += (
                    f"  → {sig.get('top_service', '?')}: "
                    f"anomaly_score={sig.get('anomaly_score', 0):.2f} "
                    f"({sig.get('reason', 'unknown')})\n"
                )

        # Memory section
        memory_section = ""
        if memory_summary:
            memory_section = f"\n=== YOUR INVESTIGATION MEMORY ===\n{memory_summary}\n"

        # Investigation results
        inv_section = ""
        inv_results = obs.get("investigation_results", [])
        if inv_results:
            inv_section = "\n=== INVESTIGATION RESULTS ===\n"
            for ir in inv_results:
                inv_section += f"  [{ir.get('finding_type', '?')}] {ir.get('finding', '')}\n"

        # Previous actions
        prev_section = ""
        prev_actions = obs.get("previous_actions", [])
        if prev_actions:
            prev_section = "\n=== PREVIOUS ATTEMPTS ===\n"
            for pa in prev_actions[-2:]:
                prev_section += (
                    f"  Step {pa.get('step', '?')}: "
                    f"{pa.get('action', {}).get('root_cause_service', '?')} "
                    f"→ reward={pa.get('score', 0):.2f}\n"
                )

        prompt = textwrap.dedent(f"""
INCIDENT TRIAGE | Task: {task_id} | Step: {step}/{max_steps} | SLA Pressure: {tp:.0%} | Score: {score:.2f}
{sla_warn}
=== ALERTS ({len(alerts)}) ===
{json.dumps(alerts[:10], indent=2)}

=== METRICS ===
{json.dumps(metrics, indent=2)}

=== CALL GRAPH TOPOLOGY ===
{json.dumps(topology, indent=2)}

=== TIMELINE ===
{json.dumps(timeline[:15], indent=2)}
{monitor_section}{memory_section}{inv_section}{prev_section}
JSON only. Include ALL affected_services in cascade chain. Write stakeholder_message for P0/P1.
        """).strip()

        return prompt

    def parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse LLM response into action dict.
        Handles markdown code fences and malformed JSON.
        """
        fallback = {
            "root_cause_service": "unknown",
            "root_cause_type":    "unknown",
            "severity":           "P2",
            "affected_services":  [],
            "remediation_action": "investigate_further",
            "stakeholder_message": None,
            "confidence":          0.0,
            "reasoning":          "Parse failed.",
        }

        if not text:
            return fallback

        # Strip markdown code fences
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return fallback

    def shape_reward(
        self,
        env_reward: float,
        action: Dict[str, Any],
        raw_text: str = "",
        memory_used: bool = False,
        monitor_integrated: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Composite reward shaping for LLM training.
        Adds bonuses for format compliance, reasoning quality, and collaboration.
        """
        bonuses = {}

        # Format compliance: valid JSON = +0.03
        try:
            json.loads(raw_text)
            bonuses["format_compliance"] = 0.03
        except Exception:
            bonuses["format_compliance"] = 0.0

        # Reasoning quality: mentions key analysis concepts
        reasoning = (action.get("reasoning") or "").lower()
        reasoning_score = 0.0
        for keyword in ["topology", "traversal", "cascade", "root cause", "metrics"]:
            if keyword in reasoning:
                reasoning_score += 0.01
        bonuses["reasoning_quality"] = min(0.03, reasoning_score)

        # Memory utilization: agent references prior steps
        if memory_used or "previous" in reasoning or "step" in reasoning:
            bonuses["memory_utilization"] = 0.02
        else:
            bonuses["memory_utilization"] = 0.0

        # Multi-agent collaboration: uses monitor signals
        if monitor_integrated or "monitor" in reasoning or "anomaly" in reasoning:
            bonuses["collaboration"] = 0.03
        else:
            bonuses["collaboration"] = 0.0

        # Confidence calibration: penalize overconfident wrong answers
        confidence = action.get("confidence", 0.5)
        if env_reward < 0.3 and confidence > 0.8:
            bonuses["overconfidence_penalty"] = -0.02
        else:
            bonuses["overconfidence_penalty"] = 0.0

        total_bonus = sum(bonuses.values())
        shaped_reward = round(min(1.0, max(0.0, env_reward + total_bonus)), 4)

        return shaped_reward, bonuses
