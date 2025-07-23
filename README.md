# TAO-WM
Learning Hierarchical Policies From Play Data and a World Model

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:Erfi/TAO-WM.git
cd TAO-WM
export TAO_ROOT=$(pwd)
conda create -n tao_venv python=3.10 -y
conda activate tao_venv
sh install.sh                
```

### 2. For development:
```
pip install -r requirements-dev.txt
pre-commit install
```

### 3. Set environment variables for datasets and logging directory (default is dataset/ and logs/), and set WandB entity (username or team name)
```bash
source scripts/set_path.sh
```
