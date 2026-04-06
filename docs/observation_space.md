# Observation Space

Each step the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| alerts | list[Alert] | Active monitoring alerts with service, metric, value, threshold |
| metrics | dict | Current CPU/memory/latency per service |
| topology | list[Edge] | Call graph: upstream->downstream with latencies |
| timeline | list[Event] | Chronological incident events |
| time_pressure | float | SLA breach urgency 0.0-1.0 |
| sla_breach_in_steps | int | Steps until SLA breach (hard task) |
| previous_actions | list | History with scores for adaptation |
