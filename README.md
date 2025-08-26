# TAO-WM
Learning Hierarchical Policies From Play Data and a World Model

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:Erfi/TAO-WM.git
cd TAO-WM
export TAO_ROOT_DIR=$(pwd)
conda create -n tao_venv python=3.10 -y
conda activate tao_venv
sh install.sh                
```

### 2. For development:
```
pip install -r requirements-dev.txt
pre-commit install
```

### 3. Set environment variables
For datasets and logging directory (default is dataset/ and logs/), and set WandB entity (username or team name)
```bash
source scripts/set_path.sh
```

### 4. Dataset
To download and preprocess datasets, please follow A.0 and A.1 [here](dataset/README.md#a-calvin).

### 5. World Model

#### 5.1 Train a WM
```bash
python scripts/train_wm.py
```

#### 5.2 Featurize the dataset with a learned WM
```bash
python scripts/featurizer.py
```
