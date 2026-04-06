# Architecture

## System Design
Client -> HTTP/WebSocket -> FastAPI Server -> IncidentResponseEnv -> Graders -> Reward

## Components
- models/: Pydantic typed models (Action, Observation, Reward, State)
- scenarios/: JSON scenario definitions with ground truth
- graders/: Multi-dimensional scoring (root cause, action, severity, communication, speed)
- envs/: Core environment logic (reset/step/state)
- server/: FastAPI server with WebSocket support
- client/: HTTP and WebSocket clients
- data/: Alibaba cluster trace processed patterns

## Data Sources
- Alibaba Cluster Trace v2021 (microservices)
- Microsoft AIOpsLab fault taxonomy
- Google SRE Book incidents (Ch 13-16)
