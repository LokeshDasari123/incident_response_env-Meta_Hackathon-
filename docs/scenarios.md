# Scenarios

## Easy: Change-Induced Single Service Failure
- Fault: ConfigMap misconfiguration on payments-db
- Cascade: payments-db -> payments-api -> checkout-ui
- Red herring: CPU spike on worker-node-4
- Max steps: 10

## Medium: Test-Induced Hidden Dependency Cascade  
- Fault: DNS resolution failure for user-service
- Cascade: user-service -> auth-service -> api-gateway -> storefront-ui
- Red herrings: cache-service memory warning, worker-node-4 CPU
- Max steps: 15

## Hard: Process-Induced Cascading Failure with SLA Pressure
- Fault: Memory leak + crash-loop on payments-db
- Cascade: payments-db -> cache-service -> order-service -> api-gateway -> storefront-ui
- Red herrings: network-switch-03 latency, worker-node-7 CPU
- SLA breach at step 6
- Max steps: 20
