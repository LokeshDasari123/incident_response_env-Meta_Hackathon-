# Action Space

The agent submits an IncidentAction each step:

| Field | Type | Values |
|-------|------|--------|
| root_cause_service | string | Any service name from topology |
| root_cause_type | enum | misconfiguration, memory_leak, network_partition, crash_loop, resource_exhaustion, auth_failure, dependency_failure, unknown |
| severity | enum | P0, P1, P2, P3 |
| affected_services | list[string] | All impacted services |
| remediation_action | enum | rollback, restart_service, scale_up, fix_config, increase_connection_pool, flush_cache, reroute_traffic, escalate, investigate_further |
| stakeholder_message | string | Required for P0/P1 |
| confidence | float | 0.0-1.0 |
| reasoning | string | Chain of thought |
