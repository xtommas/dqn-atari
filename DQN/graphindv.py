import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Recompensa

data = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing.csv')
#batchsize16 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_BatchSize=16.csv')
#batchsize64 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_BatchSize=64.csv')
#buffersize50 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_BufferSize=50000.csv')
#buffersize200 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_BufferSize=200000.csv')
#exploration02 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_ExplorationFraction=0.2.csv')
#exploration005 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_ExplorationFraction=0.05.csv')
#framestack2 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_Framestack=2.csv')
#framestack6 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_Framestack=6.csv')
#learning00001 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_LearningRate=0.00001.csv')
#learning001 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\recompensa\\Boxing_LearningRate=0.001.csv')

# Episodios

#data = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing.csv')
#batchsize16 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_BatchSize=16.csv')
#batchsize64 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_BatchSize=64.csv')
#buffersize50 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_BufferSize=50000.csv')
#buffersize200 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_BufferSize=200000.csv')
# exploration02 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_ExplorationFraction=0.2.csv')
# exploration005 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_ExplorationFraction=0.05.csv')
# framestack2 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_Framestack=2.csv')
# framestack6 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_Framestack=6.csv')
#learning00001 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_LearningRate=0.00001.csv')
#learning001 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Boxing\\episodios\\Boxing_LearningRate=0.001.csv')

#f = plt.figure()
#f.set_figwidth(10)
#f.set_figheight(15)


plt.plot(data['Step'], data['Value'], label = 'Modelo Base')
#plt.plot(batchsize16['Step'], batchsize16['Value'], label = 'Modelo Batch Size = 16')
#plt.plot(batchsize64['Step'], batchsize64['Value'], label = 'Modelo Batch Size = 64')
# plt.plot(buffersize50['Step'], buffersize50['Value'], label = 'Modelo Buffer Size = 50.000')
# plt.plot(buffersize200['Step'], buffersize200['Value'], label = 'Modelo Buffer Size = 200.000')
# plt.plot(exploration02['Step'], exploration02['Value'], label = 'Modelo Exploration Fraction = 0.2')
# plt.plot(exploration005['Step'], exploration005['Value'], label = 'Modelo Exploration Fraction = 0.05')
# plt.plot(framestack2['Step'], framestack2['Value'], label = 'Modelo Framestack = 2')
# plt.plot(framestack6['Step'], framestack6['Value'], label = 'Modelo Framestack = 6')
#plt.plot(learning00001['Step'], learning00001['Value'], label = 'Modelo Learning Rate = 0.00001')
#plt.plot(learning001['Step'], learning001['Value'], label = 'Modelo Learning Rate = 0.001')

plt.xlabel("Paso de tiempo")
plt.ylabel("Recompensa")
#plt.ylabel("Frames / Pasos de tiempo")

plt.legend()

current_values = plt.gca().get_xticks()
# using format string '{:.0f}' here but you can choose others
plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])

#plt.xticks(np.arange(0, 10000000, 500000))



# Show the plot
plt.show()