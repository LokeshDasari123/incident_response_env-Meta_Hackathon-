"""
agents/chain_of_thought.py
--------------------------
4-Phase Chain-of-Thought Reasoning Engine.

This is the core of "long-horizon RL" — the agent solves an incident
through structured multi-step reasoning rather than one-shot guessing.

Phases:
  1. SCAN      → Fast model: "List top 3 suspicious services"
  2. ANALYZE   → Balanced model: pick root cause + fault type
  3. DECIDE    → Strong model (complexity ≥ 0.70): remediation + severity
  4. COMMUNICATE → Fast model: draft stakeholder message

Memory integration:
  - Before each phase, the complexity-gated STM/LTM context is injected
  - Phase outputs are fed forward into the next phase's prompt
  - The full chain produces a single merged IncidentAction dict

Falls back gracefully if any phase fails or API key is missing.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS   = 512
SCAN_MAX_TOKENS      = 256   # scan is short
COMMUNICATE_TOKENS   = 256
FALLBACK_ACTION: Dict[str, Any] = {
    "root_cause_service":  "unknown",
    "root_cause_type":     "unknown",
    "severity":            "P2",
    "affected_services":   [],
    "remediation_action":  "investigate_further",
    "stakeholder_message": None,
    "confidence":          0.0,
    "reasoning":           "Chain-of-thought failed — fallback.",
    "cot_phases":          [],
}


class ChainOfThought:
    """
    4-phase chain-of-thought reasoning engine.
    Uses memory context from ProgressiveMemorySystem each phase.
    """

    # ── Phase prompts ──────────────────────────────────────────────────────────

    SCAN_SYSTEM = textwrap.dedent("""
    You are an SRE triaging a production incident. Your task is PHASE 1: SCAN.
    Quickly identify the top 3 suspicious services from the alerts and metrics.
    Do NOT make a final diagnosis yet — just narrow the search space.

    Respond ONLY with valid JSON (no markdown):
    {
      "candidates": ["<svc1>", "<svc2>", "<svc3>"],
      "top_candidate": "<most likely root cause service>",
      "reasoning": "<1-2 sentence scan rationale based on metrics/alerts>"
    }
    """).strip()

    ANALYZE_SYSTEM = textwrap.dedent("""
    You are an SRE triaging a production incident. Your task is PHASE 2: ANALYZE.
    You have already scanned and found candidate services. Now determine the root
    cause service and fault type by traversing the call graph topology INWARD.

    FAULT TYPE RULES:
    - misconfiguration = config_change in timeline + RT spike immediately after
    - memory_leak = memory_utilization > 0.85 AND restart_count > 0
    - network_partition = providerRPC_MCR drops to 0 or near-zero
    - crash_loop = restart_count > 2 AND service cycling

    Respond ONLY with valid JSON (no markdown):
    {
      "root_cause_service": "<exact service name>",
      "root_cause_type": "<misconfiguration|memory_leak|network_partition|crash_loop|resource_exhaustion|dependency_failure|unknown>",
      "confidence": <0.0-1.0>,
      "reasoning": "<topology traversal step by step>"
    }
    """).strip()

    DECIDE_SYSTEM = textwrap.dedent("""
    You are an SRE triaging a production incident. Your task is PHASE 3: DECIDE.
    You know the root cause. Now determine severity, affected services, and remediation.

    SEVERITY RULES:
    - P0 = cascading failure 2+ services OR payment/checkout impacted
    - P1 = single service degraded with user-facing impact
    - P2 = internal service issue, no user impact
    - P3 = monitoring noise only

    REMEDIATION RULES:
    - config_change caused it → fix_config
    - memory_leak / crash_loop → restart_service
    - network_partition → fix_config
    - unknown cascade → escalate

    Respond ONLY with valid JSON (no markdown):
    {
      "severity": "<P0|P1|P2|P3>",
      "affected_services": ["<cascade chain ONLY — no red herrings>"],
      "remediation_action": "<rollback|restart_service|scale_up|fix_config|reroute_traffic|escalate|investigate_further>",
      "reasoning": "<severity + remediation justification>"
    }
    """).strip()

    COMMUNICATE_SYSTEM = textwrap.dedent("""
    You are an SRE triaging a production incident. Your task is PHASE 4: COMMUNICATE.
    Draft a stakeholder message for P0/P1 incidents. Skip (return null) for P2/P3.

    The message MUST mention: affected service, user impact, and ETA.
    Example: "payments-db memory_leak causing payment failures. Applying restart. ETA 10 min."

    Respond ONLY with valid JSON (no markdown):
    {
      "stakeholder_message": "<message or null>"
    }
    """).strip()

    def __init__(self, router: Any) -> None:
        """
        Args:
            router: ComplexityRouter instance (for per-phase model selection)
        """
        self._router = router
        self._phase_log: List[Dict[str, Any]] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        obs: Dict[str, Any],
        complexity: float,
        memory_context: str = "",
        debate_challenge: Optional[str] = None,
        episode: int = 0,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Run the full 4-phase chain. Returns a merged IncidentAction dict.
        Falls back gracefully if API calls fail or keys are missing.
        """
        self._phase_log = []
        task_id = obs.get("task_id", "unknown")

        has_api = any(
            k in os.environ and os.environ[k] and not os.environ[k].startswith("your_")
            for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "HF_TOKEN")
        )
        mode = "Live API" if has_api else "Rule-Based/Fallback"
        prefix = f"[Ep {episode}][Step {step}]"

        print(f"  {prefix}[CoT] Starting 4-phase reasoning | mode={mode} | complexity={complexity:.3f}", flush=True)

        # ── Phase 1: SCAN ──────────────────────────────────────────────────────
        scan_result  = self._phase_scan(obs, complexity, memory_context)
        candidates   = scan_result.get("candidates", [])
        top_candidate = scan_result.get("top_candidate", "unknown")
        print(f"  {prefix}[CoT][SCAN] candidates={candidates} top={top_candidate}", flush=True)

        # ── Phase 2: ANALYZE ───────────────────────────────────────────────────
        analyze_result = self._phase_analyze(obs, complexity, memory_context, scan_result)
        rc_service    = analyze_result.get("root_cause_service", top_candidate)
        rc_type       = analyze_result.get("root_cause_type", "unknown")
        confidence    = float(analyze_result.get("confidence", 0.5))
        print(f"  {prefix}[CoT][ANALYZE] rc={rc_service} type={rc_type} conf={confidence:.0%}", flush=True)

        # ── Phase 3: DECIDE ────────────────────────────────────────────────────
        decide_result = self._phase_decide(
            obs, complexity, memory_context, scan_result, analyze_result, debate_challenge
        )
        severity      = decide_result.get("severity", "P2")
        affected      = decide_result.get("affected_services", [rc_service])
        remediation   = decide_result.get("remediation_action", "investigate_further")
        print(f"  {prefix}[CoT][DECIDE] severity={severity} action={remediation} "
              f"affected={len(affected)} svcs", flush=True)

        # ── Phase 4: COMMUNICATE ───────────────────────────────────────────────
        comm_result   = self._phase_communicate(
            obs, complexity, rc_service, rc_type, severity, affected, remediation
        )
        stakeholder_msg = comm_result.get("stakeholder_message")
        print(f"  {prefix}[CoT][COMM] msg={'OK' if stakeholder_msg else 'null'}", flush=True)

        # ── Merge into final action ────────────────────────────────────────────
        merged_reasoning = " → ".join([
            scan_result.get("reasoning", ""),
            analyze_result.get("reasoning", ""),
            decide_result.get("reasoning", ""),
        ])

        action = {
            "root_cause_service":  rc_service,
            "root_cause_type":     rc_type,
            "severity":            severity,
            "affected_services":   list(dict.fromkeys(affected)),  # deduplicate
            "remediation_action":  remediation,
            "stakeholder_message": stakeholder_msg,
            "confidence":          round(confidence, 2),
            "reasoning":           merged_reasoning[:300],
            "cot_phases": self._phase_log,
        }
        return action

    # ── Phase implementations ──────────────────────────────────────────────────

    def _phase_scan(
        self,
        obs: Dict[str, Any],
        complexity: float,
        memory_context: str,
    ) -> Dict[str, Any]:
        """PHASE 1: Fast model scans for candidate services."""
        fallback = {
            "candidates": list(obs.get("metrics", {}).keys())[:3],
            "top_candidate": "unknown",
            "reasoning": "Scan fallback (rule-based).",
        }

        user_content = self._build_scan_prompt(obs, memory_context)
        result = self._call_phase(
            phase_name    = "scan",
            system        = self.SCAN_SYSTEM,
            user          = user_content,
            complexity    = complexity,
            force_tier    = "fast",         # always use fast model for scan
            max_tokens    = SCAN_MAX_TOKENS,
            fallback      = fallback,
        )
        return result

    def _phase_analyze(
        self,
        obs: Dict[str, Any],
        complexity: float,
        memory_context: str,
        scan_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """PHASE 2: Balanced model analyzes root cause."""
        # For low complexity, skip the API call and use rule-based
        if complexity < 0.25:
            metrics = obs.get("metrics", {})
            # Simple rule: pick the service with highest CPU or memory
            worst = max(
                metrics.items(),
                key=lambda kv: kv[1].get("cpu_utilization", 0) + kv[1].get("memory_utilization", 0),
                default=("unknown", {}),
            )
            return {
                "root_cause_service": worst[0],
                "root_cause_type":    "unknown",
                "confidence":         0.4,
                "reasoning":          "Low complexity — rule-based analysis.",
            }

        fallback = {
            "root_cause_service": scan_result.get("top_candidate", "unknown"),
            "root_cause_type":    "unknown",
            "confidence":         0.3,
            "reasoning":          "Analyze fallback.",
        }

        user_content = self._build_analyze_prompt(obs, memory_context, scan_result)
        result = self._call_phase(
            phase_name = "analyze",
            system     = self.ANALYZE_SYSTEM,
            user       = user_content,
            complexity = complexity,
            force_tier = None,           # let router decide (fast or balanced)
            max_tokens = DEFAULT_MAX_TOKENS,
            fallback   = fallback,
        )
        return result

    def _phase_decide(
        self,
        obs: Dict[str, Any],
        complexity: float,
        memory_context: str,
        scan_result: Dict[str, Any],
        analyze_result: Dict[str, Any],
        debate_challenge: Optional[str],
    ) -> Dict[str, Any]:
        """PHASE 3: Strong model (if hard) decides severity + remediation."""
        rc  = analyze_result.get("root_cause_service", "unknown")
        ft  = analyze_result.get("root_cause_type", "unknown")

        # Rule-based defaults
        rem_map = {
            "misconfiguration":    "fix_config",
            "memory_leak":         "restart_service",
            "network_partition":   "fix_config",
            "crash_loop":          "restart_service",
            "resource_exhaustion": "scale_up",
            "dependency_failure":  "escalate",
        }
        fallback = {
            "severity":            "P1",
            "affected_services":   scan_result.get("candidates", [rc]),
            "remediation_action":  rem_map.get(ft, "investigate_further"),
            "reasoning":           "Decide fallback (rule-based remediation map).",
        }

        # For low-medium complexity skip strong model
        force_tier = "strong" if complexity >= 0.70 else None

        user_content = self._build_decide_prompt(
            obs, memory_context, scan_result, analyze_result, debate_challenge
        )
        result = self._call_phase(
            phase_name = "decide",
            system     = self.DECIDE_SYSTEM,
            user       = user_content,
            complexity = complexity,
            force_tier = force_tier,
            max_tokens = DEFAULT_MAX_TOKENS,
            fallback   = fallback,
        )
        return result

    def _phase_communicate(
        self,
        obs: Dict[str, Any],
        complexity: float,
        rc_service: str,
        rc_type: str,
        severity: str,
        affected: List[str],
        remediation: str,
    ) -> Dict[str, Any]:
        """PHASE 4: Fast model drafts stakeholder message."""
        # P2/P3 → no message needed, skip API call
        if severity in ("P2", "P3"):
            self._phase_log.append({
                "phase": "communicate", "skipped": True,
                "reason": f"severity={severity}, no stakeholder msg needed",
            })
            return {"stakeholder_message": None}

        fallback = {
            "stakeholder_message": (
                f"{rc_service} {rc_type.replace('_',' ')} causing cascade "
                f"to {len(affected)} services. Severity: {severity}. "
                f"Action: {remediation.replace('_',' ')}. ETA: ~10 minutes."
            )
        }

        user_content = (
            f"Incident: {rc_service} experiencing {rc_type}. "
            f"Severity: {severity}. Affected: {', '.join(affected[:4])}. "
            f"Planned action: {remediation}. "
            "Write the stakeholder message now."
        )
        result = self._call_phase(
            phase_name = "communicate",
            system     = self.COMMUNICATE_SYSTEM,
            user       = user_content,
            complexity = complexity,
            force_tier = "fast",
            max_tokens = COMMUNICATE_TOKENS,
            fallback   = fallback,
        )
        return result

    # ── LLM caller ────────────────────────────────────────────────────────────

    def _call_phase(
        self,
        phase_name: str,
        system:     str,
        user:       str,
        complexity: float,
        force_tier: Optional[str],
        max_tokens: int,
        fallback:   Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Make one LLM call for a phase.
        Routes to the appropriate model tier, falls back to rule-based if unavailable.
        """
        t0 = time.time()

        if force_tier:
            tier  = force_tier
            _, base_url, model, api_key = self._router.route(
                complexity, ltm_recommendation=force_tier
            )
        else:
            tier, base_url, model, api_key = self._router.route(complexity)

        # Check API key availability
        if not api_key or api_key.startswith("your_"):
            # Try cascading fallback: strong → balanced → fast → rule-based
            fallback_tiers = [t for t in ("balanced", "fast") if t != tier]
            for ft in fallback_tiers:
                _, fb_url, fb_model, fb_key = self._router.route(0.0, ltm_recommendation=ft)
                if fb_key and not fb_key.startswith("your_"):
                    base_url, model, api_key, tier = fb_url, fb_model, fb_key, ft
                    break
            else:
                # All API keys missing — use rule-based fallback
                elapsed = time.time() - t0
                self._phase_log.append({
                    "phase": phase_name, "tier": "rule_based",
                    "model": "none", "latency_ms": round(elapsed * 1000, 1),
                    "tokens": 0, "fallback": True,
                })
                return fallback

        # Make the API call
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key)
            resp = client.chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature = 0.15,
                max_tokens  = max_tokens,
                stream      = False,
            )
            raw_text    = (resp.choices[0].message.content or "").strip()
            tokens_used = getattr(resp.usage, "total_tokens", 0)
            result      = self._parse_json(raw_text, fallback)

            elapsed = time.time() - t0
            self._phase_log.append({
                "phase":      phase_name,
                "tier":       tier,
                "model":      model,
                "latency_ms": round(elapsed * 1000, 1),
                "tokens":     tokens_used,
                "fallback":   False,
                "complexity": round(complexity, 3),
            })
            return result

        except Exception as exc:
            elapsed = time.time() - t0
            self._phase_log.append({
                "phase":      phase_name,
                "tier":       tier,
                "model":      model,
                "latency_ms": round(elapsed * 1000, 1),
                "tokens":     0,
                "fallback":   True,
                "error":      str(exc)[:80],
            })
            return fallback

    @staticmethod
    def _parse_json(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM JSON output, stripping markdown fences."""
        if not text:
            return fallback
        if "```" in text:
            parts = text.split("```")
            text  = parts[1] if len(parts) > 1 else parts[0]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return fallback

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _build_scan_prompt(obs: Dict[str, Any], memory_ctx: str) -> str:
        alerts   = obs.get("alerts",  [])[:8]
        metrics  = obs.get("metrics", {})
        failing  = {
            k: {"cpu": round(v.get("cpu_utilization", 0), 2),
                "mem": round(v.get("memory_utilization", 0), 2),
                "status": v.get("status", "?")}
            for k, v in metrics.items() if not v.get("is_healthy", True)
        }
        parts = [
            f"TASK: {obs.get('task_id','?')} | STEP: {obs.get('step',0)}/{obs.get('max_steps',10)}",
            f"FAILING SERVICES: {json.dumps(failing, indent=2)}",
            f"ALERTS ({len(alerts)}): {json.dumps(alerts, indent=2)}",
        ]
        if memory_ctx:
            parts.append(f"MEMORY CONTEXT:\n{memory_ctx}")
        parts.append("Identify the top 3 suspicious services and the most likely root cause.")
        return "\n\n".join(parts)

    @staticmethod
    def _build_analyze_prompt(
        obs: Dict[str, Any],
        memory_ctx: str,
        scan: Dict[str, Any],
    ) -> str:
        topology = obs.get("topology", [])
        timeline = obs.get("timeline", [])[:10]
        metrics  = {k: v for k, v in obs.get("metrics", {}).items()
                    if k in scan.get("candidates", [])}
        parts = [
            f"TASK: {obs.get('task_id','?')} | STEP: {obs.get('step',0)}/{obs.get('max_steps',10)}",
            f"SCAN RESULT: candidates={scan.get('candidates',[])} "
            f"top={scan.get('top_candidate','?')}",
            f"CANDIDATE METRICS:\n{json.dumps(metrics, indent=2)}",
            f"TOPOLOGY:\n{json.dumps(topology, indent=2)}",
            f"TIMELINE:\n{json.dumps(timeline, indent=2)}",
        ]
        if memory_ctx:
            parts.append(f"MEMORY CONTEXT:\n{memory_ctx}")
        parts.append("Determine root cause service and fault type. Traverse topology inward.")
        return "\n\n".join(parts)

    @staticmethod
    def _build_decide_prompt(
        obs: Dict[str, Any],
        memory_ctx: str,
        scan: Dict[str, Any],
        analyze: Dict[str, Any],
        debate: Optional[str],
    ) -> str:
        topology = obs.get("topology", [])
        parts = [
            f"TASK: {obs.get('task_id','?')} | STEP: {obs.get('step',0)}/{obs.get('max_steps',10)} "
            f"| TIME PRESSURE: {obs.get('time_pressure',0):.0%}",
            f"ANALYSIS: root_cause={analyze.get('root_cause_service','?')} "
            f"type={analyze.get('root_cause_type','?')} "
            f"confidence={analyze.get('confidence',0):.0%}",
            f"TOPOLOGY:\n{json.dumps(topology, indent=2)}",
        ]
        if memory_ctx:
            parts.append(f"MEMORY CONTEXT:\n{memory_ctx}")
        if debate:
            parts.append(f"⚔️ CHALLENGER CHALLENGE:\n{debate}\n"
                         "Address this challenge in your severity/remediation decision.")
        parts.append(
            "Determine severity, affected services (cascade chain only), and remediation action."
        )
        return "\n\n".join(parts)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_phase_log(self) -> List[Dict[str, Any]]:
        return self._phase_log

    def get_phase_summary(self) -> Dict[str, Any]:
        """Summary for training logs."""
        phases_used = [p["phase"] for p in self._phase_log]
        tiers_used  = list({p.get("tier", "rule_based") for p in self._phase_log})
        total_tokens = sum(p.get("tokens", 0) for p in self._phase_log)
        total_latency = sum(p.get("latency_ms", 0) for p in self._phase_log)
        fallbacks = sum(1 for p in self._phase_log if p.get("fallback", False))
        return {
            "phases_completed":    len(self._phase_log),
            "phases_used":         phases_used,
            "tiers_used":          tiers_used,
            "total_tokens":        total_tokens,
            "total_latency_ms":    round(total_latency, 1),
            "api_fallbacks":       fallbacks,
        }
