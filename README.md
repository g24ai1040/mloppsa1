# MLOps Assignment 1

This repository contains the solution for **MLOps Assignment 1**.  
It demonstrates usage of **Git branching** and **GitHub Actions (CI/CD)** for ML model training.

---

## Repository Structure

- `requirements.txt` → Python dependencies
- `misc.py` → Generic helper functions (data loading, preprocessing, training, evaluation)
- `train.py` → DecisionTreeRegressor model training script (in `dtree` branch)
- `train2.py` → KernelRidge model training script (in `kernelridge` branch)
- `.github/workflows/ci.yml` → GitHub Actions workflow (runs both models on `kernelridge` pushes)
- `README.md` → Project documentation

---

## Branches

- **main** → Base branch (merged with `dtree`)
- **dtree** → Contains `train.py`, `misc.py`, `requirements.txt`
- **kernelridge** → Contains `train2.py` and GitHub Actions workflow

---

## How to Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>