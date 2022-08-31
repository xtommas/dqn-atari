import gym
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3 import DQN
from stable_baselines3.common import atari_wrappers
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from pathlib import Path
import os

if __name__ == "__main__":
    
    # Hiperparámetros

    buffer_size = 100000 # tamaño de la memoria de replay
    learning_rate = 0.0001 
    batch_size = 32 # tamaño del subconjunto tomado del buffer de replay
    learning_starts = 100000 # cuántos pasos en agente debería recolectar transiciones antes de empezar a aprender (coincide con la memoria de replay llena)
    target_update_interval = 1000 # frecuencia en pasos en la que se actualiza la target network
    train_freq = 4 # cantidad de pasos en las que el modelo se actualiza
    gradient_steps = 1 
    exploration_fraction = 0.1 # fracción en la que se va reduciendo la tasa de exploración
    exploration_final_eps = 0.01 # valor final de la probabilidad de elegir una acción aleatoria
    frames = 4 # tamaño del stack de frames

    timesteps = 10000000
    juego = "SpaceInvadersNoFrameskip-v4"

    # directorio del log de tensorboard
    path = Path(os.getcwd())
    tensorboard = str(path.parent) + "\dqn_atari_tensorboard"

    env = make_atari_env(juego)
    env = VecFrameStack(env, n_stack = frames)
    
    nombremodelo = juego + "DQN"
    
    modelo = DQN(policy  = "CnnPolicy", env = env, verbose=1, tensorboard_log = tensorboard, device="cuda", buffer_size = buffer_size, learning_rate = learning_rate, batch_size = batch_size, learning_starts = learning_starts, target_update_interval = target_update_interval, train_freq = train_freq, gradient_steps = gradient_steps, exploration_fraction = exploration_fraction, exploration_final_eps = exploration_final_eps, optimize_memory_usage = True)
    #modelo = DQN.load("./Modelos/" + nombremodelo)
    modelo.set_env(env)
    modelo.learn(total_timesteps=timesteps)
    modelo.save("./Modelos/" + nombremodelo)
    env.close()