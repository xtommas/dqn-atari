import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Initialize the lists for X and Y
data = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout.csv')
batchsize16 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_BatchSize=16.csv')
batchsize64 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_BatchSize=64.csv')
buffersize50 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_BufferSize=50000.csv')
buffersize200 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_BufferSize=200000.csv')
exploration02 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_ExplorationFraction=0.2.csv')
exploration005 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_ExplorationFraction=0.05.csv')
framestack2 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_Framestack=2.csv')
framestack6 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_Framestack=6.csv')
learning00001 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_LearningRate=0.00001.csv')
learning001 = pd.read_csv('C:\\Users\\tomas\\Documents\\Facultad\\Tesis\\RL\\DQN\\csv\\Breakout\\recompensa\\Breakout_LearningRate=0.001.csv')

f = plt.figure()
f.set_figwidth(25)
f.set_figheight(15)


line1 = plt.plot(data['Step'], data['Value'], label = 'Modelo Base')
line2 = plt.plot(batchsize16['Step'], batchsize16['Value'], label = 'Modelo Batch Size = 16')
line3 = plt.plot(batchsize64['Step'], batchsize64['Value'], label = 'Modelo Batch Size = 64')
line4 = plt.plot(buffersize50['Step'], buffersize50['Value'], label = 'Modelo Buffer Size = 50.000')
line5 = plt.plot(buffersize200['Step'], buffersize200['Value'], label = 'Modelo Buffer Size = 200.000')
line6 = plt.plot(exploration02['Step'], exploration02['Value'], label = 'Modelo Exploration Fraction = 0.2')
line7 = plt.plot(exploration005['Step'], exploration005['Value'], label = 'Modelo Exploration Fraction = 0.05')
line8 = plt.plot(framestack2['Step'], framestack2['Value'], label = 'Modelo Framestack = 2')
line9 = plt.plot(framestack6['Step'], framestack6['Value'], label = 'Modelo Framestack = 6')
line10 = plt.plot(learning00001['Step'], learning00001['Value'], label = 'Modelo Learning Rate = 0.00001')
line11 = plt.plot(learning001['Step'], learning001['Value'], label = 'Modelo Learning Rate = 0.001')

plt.xlabel("Paso de tiempo")
plt.ylabel("Recompensa")

plt.legend()

current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])

#plt.xticks(np.arange(0, 11000000, 2000000))

plt.text(10050000, data['Value'].values[-1], line1[0].get_label(),  color = line1[0].get_color(), fontweight = "bold")
plt.text(10050000, batchsize16['Value'].values[-1], line2[0].get_label(),  color = line2[0].get_color(), fontweight = "bold")
plt.text(10050000, batchsize64['Value'].values[-1], line3[0].get_label(),  color = line3[0].get_color(), fontweight = "bold")
plt.text(10050000, buffersize50['Value'].values[-1], line4[0].get_label(),  color = line4[0].get_color(), fontweight = "bold")
plt.text(10050000, buffersize200['Value'].values[-1], line5[0].get_label(),  color = line5[0].get_color(), fontweight = "bold")
plt.text(10050000, exploration02['Value'].values[-1], line6[0].get_label(),  color = line6[0].get_color(), fontweight = "bold")
plt.text(10050000, exploration005['Value'].values[-1], line7[0].get_label(),  color = line7[0].get_color(), fontweight = "bold")
plt.text(10050000, framestack2['Value'].values[-1], line8[0].get_label(),  color = line8[0].get_color(), fontweight = "bold")
plt.text(10050000, framestack6['Value'].values[-1], line9[0].get_label(),  color = line8[0].get_color(), fontweight = "bold")
plt.text(10050000, learning00001['Value'].values[-1], line10[0].get_label(),  color = line10[0].get_color(), fontweight = "bold")
plt.text(10050000, learning001['Value'].values[-1], line11[0].get_label(),  color = line11[0].get_color(), fontweight = "bold")

#plt.xlim([100000, 12500000])

plt.tight_layout(pad=6.0)
# Show the plot
plt.show()