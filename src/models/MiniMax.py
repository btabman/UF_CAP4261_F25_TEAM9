
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from math import sin, cos, radians, floor
import warnings
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)

# global time step 
DT = 0.1

# Player class controls player movement according to physics
class Player():
    # constructor for class
    def __init__(self, id, p90a, x0, y0, s, a, dir, side, role, pos, predict=True):
        self.id = id
        self.max_a = p90a
        self.direction = dir   
        self.coordinate = np.array([x0, y0])
        self.velocity = np.array([s * sin(radians(dir)), s * cos(radians(dir))])
        self.acceleration = np.array([a * sin(radians(dir)), a * cos(radians(dir))])
        self.offense = (side == 'Offense')
        self.receiver = (role == 'Targeted Receiver')
        self.position = pos
        self.coordinates = [self.coordinate.copy()]
        self.predict = predict

    # moves player in a certain direction by a number of timesteps
    def move(self, minimax_direction, timesteps=1):
        new_a = np.array([self.max_a * sin(radians(minimax_direction)), self.max_a * cos(radians(minimax_direction))])
        for t in range(timesteps):
            if t<5:
                self.acceleration = self.acceleration*(1-t/5) + new_a*t/5
            else:
                self.acceleration = new_a
            self.velocity += self.acceleration * DT
            self.coordinate += self.velocity * DT
            self.coordinates.append(self.coordinate.copy())       
        return self.coordinates

    # get the direction in which the player can move
    def legalActions(self, right_dir):
        if not self.predict:
            return [self.direction]
        elif right_dir:
            return [90, 165, 15]
        else:
            return [270, 345, 195]
    
# Play class generates the game tree according to miniMax alpha-beta pruning implementation
class Play():
    # constructor
    def __init__(self, game_id, play_id, players, num_frames, ball_x, ball_y, right_dir, num_moves=0):
        self.game_id = game_id
        self.play_id = play_id
        self.num_frames = num_frames
        self.ball_position = np.array([ball_x, ball_y])
        self.offense = []
        self.defense = []
        for p in players:
            if p.offense:
                self.offense.append(p)
            else:
                self.defense.append(p)
        self.right_dir = right_dir
        self.num_moves = num_moves
    
    # collects trajectory predictions for players where it is required
    def generatePredictions(self):
        pred = pd.DataFrame(columns=['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y'])
        for player in self.offense + self.defense:
            if player.predict:
                new_df = pd.DataFrame({
                    'game_id': self.game_id,
                    'play_id': self.play_id,
                    'nfl_id': player.id,
                    'frame_id': [i for i in range(1,len(player.coordinates))],
                    'x': [pos[0] for pos in player.coordinates[1:]],
                    'y': [pos[1] for pos in player.coordinates[1:]]
                })
                pred = pd.concat([pred, new_df], ignore_index=True)
        return pred

    # generates the successor state for a play
    def generateSuccessor(self, playerIndex, action, num_moves, offense=True):
        nextPlay = Play(self.game_id, self.play_id, copy.deepcopy(self.offense + self.defense), self.num_frames, self.ball_position[0], self.ball_position[1], self.right_dir, num_moves)
        t = floor(self.num_frames/2)
        if self.num_frames % 2 == 1 and num_moves > 1:
            t = int(self.num_frames/2) + 1
            
        if offense:
            nextPlay.offense[playerIndex].move(action, t)
        else:
            nextPlay.defense[playerIndex].move(action, t)
        return nextPlay

    # evaluation function that gives higher values for a receiver being close to the ball landing position, and lower values when a defender is close to an offensive player.
    def evaluationFunction(self):
        total_score = 0
        for player in (self.offense):
            weight = 1
            if player.receiver:
                weight = 50
            dist_to_ball = ((player.coordinate[0] - self.ball_position[0])**2 + (player.coordinate[1]- self.ball_position[1])**2)**0.5
            total_score += -weight*dist_to_ball
            for defender in (self.defense):
                dist_to_defender = ((player.coordinate[0] - defender.coordinate[0])**2 + (player.coordinate[1] - defender.coordinate[1])**2)**0.5
                if dist_to_defender < 2.0:
                    total_score += -100
        return total_score

    # recursive component of miniMax algorithm that takes care of the defensive players
    def minIter(self, defenseIndex, alpha, beta):
        min_value = float('inf')
        min_pred = None
        for action in self.defense[defenseIndex].legalActions(self.right_dir):
            successor = self.generateSuccessor(defenseIndex, action, self.num_moves, False)
            if defenseIndex == len(self.defense) - 1:
                if self.num_moves == 2:
                    value, pred = successor.evaluationFunction(), successor.generatePredictions()
                else:
                    value, pred = successor.maxIter(0, alpha, beta)
            else:
                value, pred = successor.minIter(defenseIndex + 1, alpha, beta)
            if (value < min_value):
                min_pred = pred
                min_value = value
            if min_value <= alpha:
                break
            beta = min(beta, min_value)
        return min_value, min_pred

    # recursive component of miniMax algorithm that takes care of the offensive players
    def maxIter(self, offenseIndex, alpha, beta):      
        max_value = float('-inf')
        max_pred = None
        for action in self.offense[offenseIndex].legalActions(self.right_dir):
            successor = None
            if offenseIndex == 0:
                successor = self.generateSuccessor(offenseIndex, action, (self.num_moves + 1), True)
            else:
                successor = self.generateSuccessor(offenseIndex, action, self.num_moves, True)
            if offenseIndex == len(self.offense) - 1:
                value, pred = successor.minIter(0, alpha, beta)
            else:
                value, pred = successor.maxIter(offenseIndex + 1, alpha, beta)
            if (value > max_value):
                max_pred = pred
                max_value = value
            if max_value >= beta:
                break
            alpha = max(alpha, max_value)
        return max_value, max_pred

    # starts the miniMax algorithm
    def miniMax(self):
        alpha = float('-inf')
        beta = float('inf')
        max, predictions = self.maxIter(0, alpha, beta)
        return predictions
    
# function that creates all required objects to generate the required predictions for a play
def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:

    # Compute 90th percentile acceleration for each player
    acceleration_df = (
        test_input
        .group_by("nfl_id")
        .agg(pl.col("a").quantile(0.9).alias("p90a"))
    )

    # Take latest movement data for each player
    data = (
        test_input
        .group_by(["game_id", "play_id", "nfl_id"])
        .agg(pl.all().last())
    )

    game_id = data["game_id"][0]
    play_id = data["play_id"][0]
    num_frames = int(data["num_frames_output"][0])
    ball_x = data["ball_land_x"][0]
    ball_y = data["ball_land_y"][0]
    right_direction = (data["play_direction"][0] == "right")

    # Build player objects
    players = []
    for player_row in data.iter_rows(named=True):
        player = Player(
            id=player_row["nfl_id"],
            p90a=acceleration_df.filter(pl.col("nfl_id") == player_row["nfl_id"]).select("p90a").item(),
            x0=player_row["x"],
            y0=player_row["y"],
            s=player_row["s"],
            a=player_row["a"],
            dir=player_row["dir"],
            side=player_row["player_side"],
            role=player_row["player_role"],
            pos=player_row["player_position"],
            predict=player_row["player_to_predict"]
        )
        players.append(player)

    # create Play object and run minimax
    play = Play(game_id, play_id, players, num_frames, ball_x, ball_y, right_direction)
    predictions = play.miniMax()  
    
    assert isinstance(predictions, (pd.DataFrame, pl.DataFrame))
    assert len(predictions) == len(test)
    return predictions

# upload data

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "train" / "input_2023_w01.csv"
train_df = pl.read_csv(csv_path)

csv_path = base_dir / "train" / "output_2023_w01.csv"
test_df = pl.read_csv(csv_path)

train_df = train_df.filter(
    (pl.col("game_id") == 2023090700)  & (pl.col("play_id") == 1069)
)

# run predictions

unique_combinations = train_df.select(
    ["game_id", "play_id", "num_frames_output", "ball_land_x", "ball_land_y", "play_direction"]
).unique()

predictions_df = pd.DataFrame(columns=['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y'])
for row in unique_combinations.iter_rows(named=True):
    predictions = predict(
        test=test_df.filter(
            (pl.col("game_id") == row['game_id']) & (pl.col("play_id") == row['play_id'])),
        test_input=train_df.filter(
            (pl.col("game_id") == row['game_id']) & (pl.col("play_id") == row['play_id'])
        ))  
    predictions_df = pd.concat([predictions_df, predictions], ignore_index=True)
predictions_df.to_csv(base_dir / "predictions.csv", index=False)

# calculate RMSE

merged_df = pd.merge(
    test_df.to_pandas(),
    predictions_df,
    on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
    suffixes=('_true', '_pred') 
)

rmse = np.sqrt(0.5/len(merged_df)*(((merged_df['x_true'] - merged_df['x_pred']) ** 2).sum()
                 +((merged_df['y_true'] - merged_df['y_pred']) ** 2).sum()))
print(f"RMSE: {rmse}")

# create trajectory plot

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

# plot true trajectory
plt.scatter(merged_df['x_true'], merged_df['y_true'], alpha=0.6, s=20, label='True', color='blue')

# plot predicted trajectory
plt.scatter(merged_df['x_pred'], merged_df['y_pred'], alpha=0.6, s=20, label='Predicted', color='red')

plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.title('True vs Predicted Trajectories', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(base_dir / 'cross_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("Cross plot saved as 'cross_plot.png'")

