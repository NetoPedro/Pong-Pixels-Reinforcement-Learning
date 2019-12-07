import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import argparse

import torch

import wimblepong
from PIL import Image
from agent import Agent, Network
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play

episodes = 10000000
target_update = 250

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
player = Agent(player_id)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

writer = SummaryWriter()

win1 = 0
score = 0
iteraction = 0
player.random_start_iter = 9900
final_epsilon = 0.05
initial_epsilon = 1.0
exploration_eps = 250000
update_frequency = 4
player.epsilon = initial_epsilon
frames = 0
replays = 0

for i in range(0,episodes):
    points = 0
    score = 0
    if iteraction > 1000000:

        final_epsilon = 0.05
    
    while points < 21 :
        done = False
        cum_reward = 0
        (ob1,ob2) = env.reset()
        player.reset()
        while not done:

            action1 = player.get_action(ob1)
            action2 = opponent.get_action()
            (n_ob1, n_ob2), (rew1, rew2), done, info = env.step((action1,action2))

            player.store_memory(ob1, action1, n_ob1, rew1, done)

            if player.random_start_iter == 0:
                frames += 1
                if frames % update_frequency == 0:
                    iteraction += 1
                    replays += 1
                    player.update_network()

                if replays % target_update == 0:
                    player.update_target_network()
                cum_reward += rew1

                # Count the wins
                if rew1 == 10:
                    points += 1
                    score += 1
                if rew1 == -10:
                    points += 1
                    score -= 1

            ob1 = n_ob1
            ob2 = n_ob2

            player.epsilon = max(final_epsilon, initial_epsilon - iteraction / exploration_eps)
            if iteraction % 500000 == 0:
                torch.save(player.network.state_dict(), "model_6.mdl")
        writer.add_scalar("Reward/training",cum_reward,i*21 + points)
    if score >= 11:
        win1 += 1
    writer.add_scalar("Reward/score",score,i)
    writer.add_scalar("Reward/score_steos",score,iteraction)
    print("episode {} over. Broken WR: {:.3f}. Score: {}. Frames: {}. Steps: {}".format(i, win1 / (i + 1),score,frames,iteraction))

writer.close()
