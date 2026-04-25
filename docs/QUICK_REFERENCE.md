# SENTINEL Quick Reference Card

## ⚡ Most Common Commands

### Setup (First Time Only)
```powershell
# Backend setup
cd d:\MetaHack\incident_response_env-Meta_Hackathon-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Frontend setup
cd d:\MetaHack\my-app
npm install
```

### Run Application (Daily)
```powershell
# Terminal 1: Backend
cd d:\MetaHack\incident_response_env-Meta_Hackathon-
.\.venv\Scripts\Activate.ps1
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Terminal 2: Frontend
cd d:\MetaHack\my-app
npm run dev
```

### Test Everything
```powershell
cd d:\MetaHack\incident_response_env-Meta_Hackathon-
.\.venv\Scripts\Activate.ps1

# Quick health check
curl http://localhost:7860/health

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Test Specific Components
```powershell
# Validate environment
python scripts/validate_env.py

# Test RL loop
python test_rl_loop.py

# Test full inference
python scripts/test_full_inference.py

# Test E2E pillars
python scripts/test_e2e_pillars.py
```

### Interactive Testing
```powershell
# Start Python REPL
python

# Test via Python
from client.http_client import IncidentEnvClient
client = IncidentEnvClient("http://localhost:7860")
obs = client.reset(task_id="easy")
action = {
    "root_cause_service": "payments-db",
    "root_cause_type": "misconfiguration",
    "severity": "P0",
    "affected_services": ["payments-db", "payments-api"],
    "remediation_action": "fix_config",
    "communication": "Fixing config..."
}
obs, reward, done, info = client.step(action)
print(f"Reward: {reward}")
```

## 📡 Key Endpoints

| URL | Purpose |
|-----|---------|
| `http://localhost:7860` | Backend API |
| `http://localhost:7860/health` | Health check |
| `http://localhost:7860/docs` | Swagger API docs |
| `http://localhost:3000` | Frontend |
| `ws://localhost:7860/ws` | WebSocket |

## 🎮 Task Difficulties

| Difficulty | Steps | Expected Score | Root Cause |
|-----------|-------|-----------------|-----------|
| **Easy** | 10 | 0.75 (GPT-4) | Single service failure |
| **Medium** | 15 | 0.52 (GPT-4) | Hidden dependency cascade |
| **Hard** | 20 | 0.31 (GPT-4) | Memory leak + SLA pressure |
| **Expert** | 25 | 0.25 (GPT-4) | Multi-vector concurrent |

## 🔧 Common Issues & Fixes

```powershell
# Port already in use
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Clear cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall

# Clear frontend cache
rm -Recurse .next, node_modules
npm install

# Check Python version
python --version  # Must be 3.11+

# Activate virtual env
.\.venv\Scripts\Activate.ps1
```

## 📂 Key Files

- `server/app.py` - FastAPI server entry point
- `envs/incident_env.py` - Main environment
- `agents/base_agent.py` - Agent interface
- `training/train_grpo.py` - Training script
- `inference.py` - Baseline inference
- `my-app/app/page.tsx` - Frontend homepage

## 🚀 Workflow

1. **Setup**: Run setup commands once ✓
2. **Start**: Run both backend & frontend
3. **Test**: Run test suite
4. **Develop**: Make changes (auto-reload enabled)
5. **Deploy**: Build & run Docker container

## 📊 Monitor & Debug

```powershell
# Watch backend logs
Get-Content server.log -Tail 50 -Wait

# View training metrics
Get-Content data/training_logs/latest_summary.json | ConvertFrom-Json

# Run with debug output
pytest tests/ -vv -s

# Profile performance
python -m cProfile -s cumulative scripts/test_full_inference.py
```

## 🎯 Next Steps

- Read full guide: `SETUP_AND_RUN_GUIDE.md`
- Check architecture: `docs/architecture.md`
- Review action space: `docs/action_space.md`
- Try training: `training/train_grpo.py`
- Deploy: Build Docker image

---
**For detailed info, see SETUP_AND_RUN_GUIDE.md**
