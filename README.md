# NFL Big Data Bowl 2026 — CAP4621 Group Project

This repo contains our team's code and dashboard for the **NFL Big Data Bowl 2026 Prediction Competition** on Kaggle.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/btabman/UF_CAP4261_F25_TEAM9.git
cd nfl-big-data-2026
```
### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
* Windows:
venv\Scripts\activate
* Mac/Linux:
source venv/bin/activate
```
### 3. Install Required Packages
```bash
pip install -r requirements.txt
```
### 4. Set Up Kaggle API
Ensure Kaggle is installed:
Kaggle --version
Log into Kaggle: https://www.kaggle.com/

Go to Profile → Account → Create New API Token

Move the downloaded kaggle.json file to:
* Windows
C:\Users\<YourName>\.kaggle\kaggle.json  
* Mac/Linux 
~/.kaggle/kaggle.json   

### 5. Download the Competition Data
```bash
kaggle competitions download -c nfl-big-data-bowl-2026-prediction
python -m zipfile -e nfl-big-data-bowl-2026-prediction.zip data/raw
```