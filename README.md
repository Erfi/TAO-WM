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

### 6. Train Taowm 
*(example: Tacorl -> low-level policy)*
```bash
python scripts/train_taowm.py model=tacorl model.train_low_level=true datamodule=tacorl datamodule/datasets=goal_augmented_vision_only datamodule.root_data_dir=/home/basiri/Dev/TAO-WM/dataset/calvin_data
```

### 7. Evaluate Taowm
```bash
python scripts/evaluate_tao.py --dataset_path /home/basiri/Dev/TAO-WM/dataset/calvin_data --train_folder /home/basiri/Dev/TAO-WM/logs/runs/2025-08-25/18-38-21 --start_end_tasks  /home/basiri/Dev/TAO-WM/dataset/calvin_data/start_end_tasks.json --eval_log_dir /home/basiri/Dev/TAO-WM/logs/evaluations --num_sequences 100 --num_tasks_per_seq 5
```