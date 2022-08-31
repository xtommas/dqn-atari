import gym
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3 import DQN
from stable_baselines3.common import atari_wrappers
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

if __name__ == "__main__":
    
    frames = 4 # tamaño del framestack

    juego = "SpaceInvadersNoFrameskip-v4"
    env = make_atari_env(juego)
    env = VecFrameStack(env, n_stack = frames)

    root = tk.Tk()
    root.withdraw()
    nombremodelo = filedialog.askopenfilename(initialdir=os.getcwd())

    modelo = DQN.load(nombremodelo)
    modelo.set_env(env)

    num_eval_episodes = 100

    mean_reward, std_reward = evaluate_policy(modelo, modelo.get_env(), n_eval_episodes=num_eval_episodes, render = True)
    print("\nEpisodios: " + str(num_eval_episodes) + "\nRecompensa media: " + str(mean_reward) + "\nDesviación estándar de recompensa: " + str(np.round(std_reward, 2)) + "\n")


