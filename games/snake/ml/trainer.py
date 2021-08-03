"""
The template of the script for playing the game in the ml mode
"""
from games.snake.game.gamecore import GameStatus
from numpy.core.fromnumeric import argmax
import torch
import random
import numpy as np
import os
from collections import deque, namedtuple
from .model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
Point = namedtuple('Point', 'x, y')


class Agent:

    def __init__(self):
        self.BLOCK_SIZE = 10
        self.w = 300
        self.h = 300
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.model2 = Linear_QNet(11, 256, 3)
        # self.model2.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def is_collision(self, pt, body):
        # hits boundary
        print(pt)
        if pt.x > self.w - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - self.BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        for b in body:
            if(pt.x == b[0] and pt.y == b[1]):
                print("\n")
                print("collide self!!!!!!!!!!!!!!")
                print("\n")
                return True

        return False

    def get_state(self, game, direction):
        head = game["snake_head"]
        body = game["snake_body"]
        food = game["food"]
        point_l = Point(head[0] - 10, head[1])
        point_r = Point(head[0] + 10, head[1])
        point_u = Point(head[0], head[1] - 10)
        point_d = Point(head[0], head[1] + 10)

        dir_l = direction == "LEFT"
        dir_r = direction == "RIGHT"
        dir_u = direction == "UP"
        dir_d = direction == "DOWN"

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r, body)) or
            (dir_l and self.is_collision(point_l, body)) or
            (dir_u and self.is_collision(point_u, body)) or
            (dir_d and self.is_collision(point_d, body)),

            # Danger right
            (dir_u and self.is_collision(point_r, body)) or
            (dir_d and self.is_collision(point_l, body)) or
            (dir_l and self.is_collision(point_u, body)) or
            (dir_r and self.is_collision(point_d, body)),

            # Danger left
            (dir_d and self.is_collision(point_r, body)) or
            (dir_u and self.is_collision(point_l, body)) or
            (dir_r and self.is_collision(point_u, body)) or
            (dir_l and self.is_collision(point_d, body)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            food[0] < head[0],  # food left
            food[0] > head[0],  # food right
            food[1] < head[1],  # food up
            food[1] > head[1]  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model2(state0)

        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        self.agent = Agent()
        self.frame_iteration = 0
        self.last_x = 0
        self.last_y = 0
        self.direction = ""
        self.model2 = Linear_QNet(11, 256, 3)
        self.model2.load()
        pass

    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        if scene_info["status"] == "GAME_OVER":
            return "RESET"

        self.game = scene_info
        snake_head = scene_info["snake_head"]
        food = scene_info["food"]
        #print(self.last_x, self.last_y, snake_head[0], snake_head[1])
        # get direction
        if(snake_head[0] == self.last_x):
            if(snake_head[1] > self.last_y):
                self.direction = "DOWN"
            else:
                self.direction = "UP"
        elif(snake_head[1] == self.last_y):
            if(snake_head[0] > self.last_x):
                self.direction = "RIGHT"
            else:
                self.direction = "LEFT"

        state = self.agent.get_state(self.game, self.direction)
        print(self.direction)
        print(state)
        action = self.get_action(state)
        print(action)
        self.last_x = snake_head[0]
        self.last_y = snake_head[1]
        self.frame_iteration += 1
        if(self.frame_iteration == 1):
            return "NONE"
        clock_wise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        return new_dir

        # if snake_head[0] > food[0]:
        #     return "LEFT"
        # elif snake_head[0] < food[0]:
        #     return "RIGHT"
        # elif snake_head[1] > food[1]:
        #     return "UP"
        # elif snake_head[1] < food[1]:
        #     return "DOWN"

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model2(state0)

        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

    def reset(self):
        """
        Reset the status if needed
        """
        pass
