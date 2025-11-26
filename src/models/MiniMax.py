
import pandas as pd
import numpy as np
from pathlib import Path
from math import sin, cos, radians, floor
import warnings
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)

DT = 0.1

class Player():
    def __init__(self, id, p90a, x0, y0, s, a, dir, side, role, pos, predict=True):
        self.id = id
        self.max_a = p90a   
        self.coordinate = np.array([x0, y0])
        self.velocity = np.array([s * cos(radians(dir)), s * sin(radians(dir))])
        self.acceleration = np.array([a * cos(radians(dir)), a * sin(radians(dir))])
        self.offense = (side == 'Offense')
        self.receiver = (role == 'Targeted Receiver')
        self.position = pos
        self.coordinates = [self.coordinate.copy()]
        self.predict = predict

    def move(self, minimax_direction=360, timesteps=1):
        new_a = np.array([self.max_a * cos(radians(minimax_direction)), self.max_a * sin(radians(minimax_direction))])
        for t in range(timesteps):
            if t<5:
                self.acceleration = self.acceleration*(1-t/5) + new_a*t/5
            else:
                self.acceleration = new_a
            self.velocity += self.acceleration * DT
            self.coordinate += self.velocity * DT
            self.coordinates.append(self.coordinate.copy())       
        return self.coordinates
    

class Play():
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
        print(pred)
        return pred
    
    def legalActions(self):
        if self.right_dir:
            return [15, 90, 165]
        else:
            return [345, 270, 195]
    
    def generateSuccessor(self, playerIndex, action, num_moves, offense=True):
        nextPlay = Play(self.game_id, self.play_id, copy.deepcopy(self.offense + self.defense), self.num_frames, self.ball_position[0], self.ball_position[1], self.right_dir, num_moves)
        t = floor(self.num_frames/2)
        print(f"Generating successor for player index {playerIndex} with action {action} at move {num_moves}, offense: {offense}, for timestep {t}")
        if self.num_frames % 2 == 1 and num_moves > 1:
            t = int(self.num_frames/2) + 1
            
        if offense:
            nextPlay.offense[playerIndex].move(action, t)
        else:
            nextPlay.defense[playerIndex].move(action, t)
        return nextPlay
    
    def evaluationFunction(self):
        total_score = 0
        for player in self.offense:
            weight = 1
            if player.receiver:
                weight = 50
            dist_to_ball = ((player.coordinate[0] - self.ball_position[0])**2 + (player.coordinate[1]- self.ball_position[1])**2)**0.5
            total_score += -weight*dist_to_ball
            for defender in self.defense:
                dist_to_defender = ((player.coordinate[0] - defender.coordinate[0] )**2 + (player.coordinate[1] - defender.coordinate[1])**2)**0.5
                if dist_to_defender < 2.0:
                    total_score += -100
        return total_score
    
    def minIter(self, defenseIndex):
        min_value = float('inf')
        min_pred = None
        for action in self.legalActions():
            successor = self.generateSuccessor(defenseIndex, action, self.num_moves, False)
            if defenseIndex == len(self.defense) - 1:
                if self.num_moves == 2:
                    print ("leaf node reached")
                    value, pred = successor.evaluationFunction(), successor.generatePredictions()
                else:
                    value, pred = successor.maxIter(0)
            else:
                value, pred = successor.minIter(defenseIndex + 1)
            if (value < min_value):
                min_pred = pred
                min_value = value
        return min_value, min_pred
    
    def maxIter(self, offenseIndex):      
        max_value = float('-inf')
        max_pred = None
        for action in self.legalActions():
            successor = None
            if offenseIndex == 0:
                # if ((self.num_moves+1)%2==1):
                # print(f"Number of moves increased to {self.num_moves + 1}") 
                successor = self.generateSuccessor(offenseIndex, action, (self.num_moves + 1), True)
            else:
                print(f"Generating successor for offense index {offenseIndex} at move {self.num_moves}")
                successor = self.generateSuccessor(offenseIndex, action, self.num_moves, True)
            if offenseIndex == len(self.offense) - 1:
                value, pred = successor.minIter(0)
            else:
                value, pred = successor.maxIter(offenseIndex + 1)
            if (value > max_value):
                max_pred = pred
                max_value = value
        return max_value, max_pred
    
    def miniMax(self):
        max, predictions = self.maxIter(0)
        return predictions
    

## Process Dataset 

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "train" / "input_2023_w01.csv"
train_df = pd.read_csv(csv_path)

csv_path = base_dir / "train" / "output_2023_w01.csv"
test_df = pd.read_csv(csv_path)


acceleration_df = (
    train_df.groupby("nfl_id", as_index=False)["a"]
    .quantile(0.9)
)

# take last row per (game_id, play_id, nfl_id)
train_df = train_df.groupby(["game_id", "play_id", "nfl_id"], as_index=False).last()


# run predictions

unique_combinations = train_df[['game_id', 'play_id', 'num_frames_output', 'ball_land_x', 'ball_land_y', 'play_direction' ]].drop_duplicates()
predictions_df = pd.DataFrame(columns=['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y'])
for _, row in unique_combinations.iterrows():
    game_id = row['game_id']
    play_id = row['play_id']
    play_data = train_df[(train_df['game_id'] == game_id) & (train_df['play_id'] == play_id)]
    num_frames= int(row['num_frames_output'])
    ball_x = row['ball_land_x']
    ball_y = row['ball_land_y']
    right_direction = (row['play_direction']== 'right')
    
    players = []
    for _, player_row in play_data.iterrows():
        player = Player(
            id=player_row['nfl_id'],
            p90a=acceleration_df.loc[acceleration_df["nfl_id"] == player_row['nfl_id'], "a"].values[0],
            x0=player_row['x'],
            y0=player_row['y'],
            s=player_row['s'],
            a=player_row['a'],
            dir=player_row['dir'],
            side=player_row['player_side'],
            role=player_row['player_role'],
            pos=player_row['player_position'],
            predict=player_row['player_to_predict']
        )
        players.append(player)
    
    play = Play(game_id, play_id, players, num_frames, ball_x, ball_y, right_direction)
    
    predictions = play.miniMax()

    predictions_df = pd.concat([predictions_df, predictions], ignore_index=True)
    break
predictions_df.to_csv(base_dir / "predictions.csv", index=False)


merged_df = pd.merge(
    test_df,
    predictions_df,
    on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
    suffixes=('_true', '_pred')  # to distinguish x and y from each DataFrame
)


rmse = np.sqrt(0.5/len(merged_df)*(((merged_df['x_true'] - merged_df['x_pred']) ** 2).sum()
                 +((merged_df['y_true'] - merged_df['y_pred']) ** 2).sum()))
print(f"RMSE: {rmse}")
