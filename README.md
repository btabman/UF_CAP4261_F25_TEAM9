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
```

* Windows:
```bash
venv\Scripts\activate
```
* Mac/Linux:
```bash
source venv/bin/activate
```
### 3. Install Required Packages
```bash
pip install -r requirements.txt
```
### 4. Set Up Kaggle API
Ensure Kaggle is installed:
```
kaggle --version
```
Log into Kaggle: https://www.kaggle.com/

Go to Profile → Account → Create New API Token

Move the downloaded kaggle.json file to:
* Windows
```
C:\Users\<YourName>\.kaggle\kaggle.json
```
* Mac/Linux
```
~/.kaggle/kaggle.json   
```
### 5. Download the Competition Data
```bash
kaggle competitions download -c nfl-big-data-bowl-2026-prediction
python -m zipfile -e nfl-big-data-bowl-2026-prediction.zip data/raw
```
### 6. Clean data set
Use the preprocess.py file to import the training and test data, perform some cleaning, and convert to parquet files. This will greatly inmprove speed compared to repeatedly reading from csv and/or holding the entire data set in memory.


### 7. Visualizing data in Power BI
Have a look at the NFL.pbix file to get familiar with the data. It will show the player movements in the field as well as the ball landing location (shown as brown diamond on the scatter plot). The primary_key is a concatenation of the game_id and play_id fields, which can be selected on the list slicer to visualize different plays. There are also range slicers for the frame_id and output_frame_id, to visualize movement through time for the training set and test set respectively.
If you need to access to Power BI, you can do so with your Gatorlink credentials: https://it.ufl.edu/cloud/collaboration-tools/office-365/?


### 8. Running Models
#### MiniMax.py
This script includes a miniMax algorithm that functions by play. Each play object includes player objects. The Player() class controls the movement of each player as described by physics. The Play() class has all the functions needed to create the game tree, which currently implements alpha-beta prunning. Outside of both classes, is the predict() function, which takes in two polars dataframes: one with input data and another with output data. This function creates the needed Player() and Play() objects, runs the miniMax algorithm, and returns predictions. At the end of the script, there is a section that is in charge of loading datasets, running the predict() function for different plays, and reporting RMSE.


#### player_model.ipynb
First use feature_processing.ipynb to transform the original dataset into an enriched dataset with more physics attributes and clustered formation and play layout variables
The player_model notebook uses the enhanced data to build an attention model by exploring various attention parameters to find an optimal configuration.
If you wish to load a .pt model built on this data structure and run it directly, us ethe code in notebooks/player_from_pt

### Transformer.py
How to run and train the model simply put into the ternminal `python3 src/models/transformer.py'. This will start training a model for yourself. In the config dictionary you can adjust any of the features to your liking/capability of your machine.
