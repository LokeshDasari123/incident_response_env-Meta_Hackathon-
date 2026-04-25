# SENTINEL — Incident Response OpenEnv
## Complete Setup & Run Guide

---

## 📋 Project Overview

**SENTINEL** is a sophisticated OpenEnv environment for AI-powered incident response triage in production systems. It features:

- **Multi-Agent Architecture**: Responder agent vs. Challenger agent with adversarial debate
- **4-Tier Curriculum**: Easy → Medium → Hard → Expert difficulty levels
- **Microservice Simulation**: Cascading failure scenarios across service topologies
- **Dual Stack**: Python FastAPI backend + Next.js React frontend

### Project Structure
```
incident_response_env/        # Main Python backend (OpenEnv)
└── server/                   # FastAPI application
└── envs/                     # RL environment definitions
└── agents/                   # Multi-agent logic
└── training/                 # RL training loop (GRPO)
└── graders/                  # Reward/scoring system

my-app/                       # Frontend (Next.js)
└── app/                      # React pages & components
└── public/                   # Static assets
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- **Python 3.11+**
- **Node.js 18+**
- **npm or yarn**
- **LLM API credentials** (OpenAI, Groq, or compatible endpoint)

### Step 1: Backend Setup

```powershell
# Navigate to project root
cd d:\MetaHack\incident_response_env-Meta_Hackathon-

# Create virtual environment (if not already done)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements.txt

# Copy environment template and configure
copy .env.example .env
# Edit .env with your API keys:
#   - API_BASE_URL: LLM endpoint (e.g., https://api.groq.com/openai/v1)
#   - API_KEY: Your LLM API key
#   - MODEL_NAME: LLM model ID (e.g., llama-3.3-70b-versatile)
```

### Step 2: Frontend Setup

```powershell
# In a new terminal, navigate to frontend
cd d:\MetaHack\my-app

# Install dependencies
npm install

# Build (optional, for production)
npm run build
```

### Step 3: Run the Application

**Terminal 1 — Backend (FastAPI Server)**
```powershell
cd d:\MetaHack\incident_response_env-Meta_Hackathon-
.\.venv\Scripts\Activate.ps1
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```
✅ Backend running at: `http://localhost:7860`

**Terminal 2 — Frontend (Next.js Dev Server)**
```powershell
cd d:\MetaHack\my-app
npm run dev
```
✅ Frontend running at: `http://localhost:3000`

---

## 🧪 Testing the Application

### 1. **Health Check** (Verify Backend)

```powershell
curl http://localhost:7860/health
```

Expected response:
```json
{"status": "healthy", "version": "1.0.0"}
```

### 2. **Run Built-in Tests**

#### Unit Tests
```powershell
cd d:\MetaHack\incident_response_env-Meta_Hackathon-
pytest tests/unit/ -v
```

#### Integration Tests
```powershell
pytest tests/integration/ -v
```

#### E2E Tests
```powershell
pytest tests/e2e/ -v
```

#### All Tests with Coverage
```powershell
pytest tests/ -v --cov=. --cov-report=html
# Coverage report: htmlcov/index.html
```

### 3. **Test Individual Components**

#### Test Environment Validation
```powershell
python scripts/validate_env.py
```

#### Test Full Inference Pipeline
```powershell
python scripts/test_full_inference.py
```

#### Test RL Loop Diagnostics
```powershell
python test_rl_loop.py
```
This runs 30 episodes and shows if complexity increases as agent improves.

#### Test E2E Pillars (Core Functionality)
```powershell
python scripts/test_e2e_pillars.py
```

---

## 🎮 Interactive Testing

### Method 1: Direct Python Client

```python
from client.http_client import IncidentEnvClient

# Initialize client
client = IncidentEnvClient("http://localhost:7860")

# Reset environment (start new scenario)
obs = client.reset(task_id="easy")
print(f"Initial observation: {obs}")

# Submit an action
action = {
    "root_cause_service": "payments-db",
    "root_cause_type": "misconfiguration",
    "severity": "P0",
    "affected_services": ["payments-db", "payments-api", "checkout-ui"],
    "remediation_action": "fix_config",
    "communication": "Restarting payments-db with corrected config"
}

obs, reward, done, info = client.step(action)
print(f"Reward: {reward}, Done: {done}")
print(f"Debate challenge: {obs.get('debate_challenge')}")
```

### Method 2: WebSocket Client (Real-time)

```python
import asyncio
from client.websocket_client import IncidentEnvWebsocketClient

async def run():
    client = IncidentEnvWebsocketClient("ws://localhost:7860/ws")
    
    async with client:
        # Reset
        obs = await client.reset(task_id="medium")
        print(f"Observation: {obs}")
        
        # Step
        action = {...}  # Your action
        obs, reward, done, info = await client.step(action)
        print(f"Reward: {reward}")

asyncio.run(run())
```

### Method 3: Test Script (Baseline Agent)

```powershell
# Run inference with a mock LLM agent
python inference.py
```

Expected output:
```
[START] task=easy env=incident-response model=llama-3.3-70b-versatile
[STEP]  step=1 action={"root_cause_service":"payments-db",...} reward=0.72 done=false error=null
[STEP]  step=2 action={"root_cause_service":"payments-db",...} reward=0.85 done=true error=null
[END]   success=true steps=2 score=0.850 rewards=0.72,0.85
```

---

## 🏋️ Training (Advanced)

### Run GRPO Training Loop

```powershell
cd d:\MetaHack\incident_response_env-Meta_Hackathon-

# Train with curriculum (easy → medium → hard)
python training/train_grpo.py \
  --task easy \
  --episodes 100 \
  --batch_size 8 \
  --learning_rate 1e-5 \
  --model_name llama-3.3-70b-versatile
```

**Parameters**:
- `--task`: Difficulty level (easy, medium, hard, expert)
- `--episodes`: Training episodes
- `--batch_size`: Batch size for GRPO
- `--learning_rate`: Learning rate
- `--model_name`: LLM model identifier

### Monitor Training Progress

```powershell
# View training logs
Get-Content data/training_logs/latest_summary.json | ConvertFrom-Json | Format-Table

# Plot reward curves
python scripts/plot_curves.py data/training_logs/reward_curves.json
```

---

## 📊 API Endpoints (OpenEnv Spec)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new scenario |
| `/step` | POST | Submit action & advance environment |
| `/state` | GET | Get current environment state |
| `/ws` | WS | WebSocket for real-time communication |

### Example: Reset Endpoint

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "dynamic": true, "seed": 42}'
```

Response:
```json
{
  "observation": {
    "services": [...],
    "alerts": [...],
    "metrics": {...}
  },
  "max_steps": 10
}
```

### Example: Step Endpoint

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "root_cause_service": "payments-db",
      "root_cause_type": "misconfiguration",
      "severity": "P0",
      "affected_services": ["payments-db", "payments-api"],
      "remediation_action": "fix_config",
      "communication": "Fixing config..."
    }
  }'
```

Response:
```json
{
  "observation": {...},
  "reward": 0.75,
  "done": false,
  "info": {
    "debate_challenge": "You identified payments-db but...",
    "debate_strategy": "topology_challenge"
  }
}
```

---

## 🧠 Task Difficulty Levels

### Easy (10 steps max)
- **Scenario**: Single service failure (ConfigMap update)
- **Expected Score**: 0.75 (GPT-4), 0.15 (Random)
- **Example**: `payments-db` fails → `payments-api` → `checkout-ui`

### Medium (15 steps max)
- **Scenario**: Hidden dependency cascade (DNS failure)
- **Expected Score**: 0.52 (GPT-4), 0.10 (Random)
- **Example**: 4 services affected, 2 red herrings

### Hard (20 steps max)
- **Scenario**: Cascading failure with SLA pressure
- **Expected Score**: 0.31 (GPT-4), 0.05 (Random)
- **Example**: Memory leak + crash-loop, SLA breach at step 6

### Expert (25 steps max) ⚡
- **Scenario**: Multi-vector concurrent failure
- **Expected Score**: 0.25 (GPT-4), 0.03 (Random)
- **Example**: Certificate expiry + memory leak, 7 services, SLA breach at step 4

---

## 🔍 Debugging & Troubleshooting

### Issue: Backend won't start

```powershell
# Check Python version
python --version  # Should be 3.11+

# Verify dependencies
pip list | grep fastapi

# Check port availability
netstat -ano | findstr :7860

# Clear Python cache
Remove-Item -Recurse -Force "__pycache__"
Remove-Item -Recurse -Force ".pytest_cache"
```

### Issue: Frontend won't compile

```powershell
# Clear Next.js cache
cd d:\MetaHack\my-app
Remove-Item -Recurse -Force ".next"
rm package-lock.json
npm install
npm run dev
```

### Issue: LLM API errors

```powershell
# Test LLM connectivity
python -c "
from openai import OpenAI
client = OpenAI(api_key='your_key', base_url='https://api.groq.com/openai/v1')
response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print(response)
"
```

### Issue: Tests failing

```powershell
# Run with verbose output
pytest tests/ -vv -s

# Run specific test
pytest tests/unit/test_env.py::test_reset -v

# Show print statements
pytest tests/ -s
```

---

## 📈 Performance Monitoring

### Monitor Backend Logs

```powershell
# Tail logs in real-time
Get-Content server.log -Tail 50 -Wait

# Set debug log level
# In server/app.py: logger.setLevel(logging.DEBUG)
```

### Monitor Frontend Performance

Open browser DevTools (F12) → Performance tab → Record → Interact → Stop

### Benchmark Environment

```powershell
python scripts/benchmark_env.py --tasks easy,medium,hard --episodes 10
```

---

## 🤝 Contributing & Development

### Code Style

```powershell
# Format code
black .

# Lint
ruff check .

# Type checking
mypy src/
```

### Add New Test

```powershell
# Create test file
echo "
def test_new_feature():
    assert True
" > tests/unit/test_new_feature.py

# Run it
pytest tests/unit/test_new_feature.py -v
```

---

## 📚 Additional Resources

- **API Docs**: http://localhost:7860/docs (Swagger UI)
- **ReDoc Docs**: http://localhost:7860/redoc
- **README**: [README.md](./README.md)
- **Architecture**: [docs/architecture.md](./docs/architecture.md)
- **Action Space**: [docs/action_space.md](./docs/action_space.md)
- **Observation Space**: [docs/observation_space.md](./docs/observation_space.md)

---

## ✅ Verification Checklist

- [ ] Python 3.11+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with API keys
- [ ] Backend running on port 7860
- [ ] Frontend running on port 3000
- [ ] Health check passes: `curl http://localhost:7860/health`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Can reset environment via API
- [ ] Can submit actions and receive rewards
- [ ] Frontend loads without errors

---

## 🎯 Next Steps

1. **Understand the Environment**: Read [docs/architecture.md](./docs/architecture.md)
2. **Try a Task**: Use `inference.py` or the HTTP client to run a scenario
3. **Write a Simple Agent**: Create an agent in `agents/` that solves "easy"
4. **Train a Model**: Run `training/train_grpo.py` for curriculum learning
5. **Deploy**: Use Docker (`docker build -t sentinel . && docker run -p 7860:7860 sentinel`)

---

**Happy troubleshooting! 🚀**
