import gym
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from os.path import exists, expanduser
import math

env = gym.make('Pendulum-v1')
filepath = expanduser("~/Documents/Random Coding Projects/MachineLearningExperiments/OpenAI-Gym-Pendulum/Saved Networks/P_gens-1_children-500_layers-1_layerHeight-4_networkTests-20_wMax-2_bMax-2_filter-False_1.txt")

#open file:
file = open(filepath, "r")
lines = file.readlines()
file.close()

#get constants:
constants = lines[0].split(",")
neurons_per_layer = int(constants[0])
num_layers = int(constants[1])
output_space = int(constants[2])
NN = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer+2))


#get Neural network:
for l in range(neurons_per_layer*num_layers+output_space):
    sets = lines[l+1].split(",")
    for i in range(neurons_per_layer+1):
        NN[l][i]=float(sets[i])
        #nums = sets[i].split(",")
        #NN[l][i][0] = float(nums[0])
        #NN[l][i][1] = float(nums[1])

#get old performances
if neurons_per_layer*num_layers+output_space + 1 < len(lines):
    print("\nPrevious performances:")
    num = 1
    for p in range(neurons_per_layer*num_layers+output_space+1, len(lines)):
        performance = lines[p].split(",")
        old_avg_reward = float(performance[0])
        old_tests_run = int(performance[1])
        print("- Saved Performance #", num, "- total tests =", old_tests_run, ", average reward =", old_avg_reward)
        num+=1
    print()

#other constants:
num_testing_iterations = 1000
num_display_iterations = 5

#environment constants:
max_episode_length = 200


# DEFINING FUNCTIONS:
#Get output of neuronal network:
def neuralNetworkOutput(NN, input):
    last_layer = input
    for l in range(num_layers):
        #print(last_layer)
        next_layer = np.zeros(neurons_per_layer)
        for i in range(neurons_per_layer):
            value = 0
            for j in range(len(last_layer)):
                value += NN[l*neurons_per_layer + i][j] * last_layer[j] #weights
            next_layer[i] = value + NN[l*neurons_per_layer+i][neurons_per_layer] #bias
        last_layer = next_layer
    output = 0
    #print(last_layer)
    for i in range(len(last_layer)):
        output += last_layer[i] * NN[num_layers*neurons_per_layer][i]
    output += NN[num_layers*neurons_per_layer][len(last_layer)]
    return output

print("Testing Model...")
print()
avg_reward = 0
for iteration in range(num_testing_iterations):
    obs = env.reset()
    total_reward = 0
    for step in range(max_episode_length):
        obsArray = np.array(obs)
        value = neuralNetworkOutput(NN, obsArray)
        action = 2*(2/(1+math.exp(-2*value)) - 1)
        #if action > 2:
        #    action = 2
        #elif action < -2:
        #    action = -2
        obs, reward, done, info = env.step([action])
        total_reward += reward
        if done:
            break
    avg_reward += total_reward

avg_reward /= num_testing_iterations
print("Model Results:")
print()
print("Average Reward =", avg_reward)
print()

#Displaying Final Model
print("Displaying Sample Runs...\n")
for iteration in range(num_display_iterations):
    obs = env.reset()
    obsArray = np.array(obs)
    for step in range(max_episode_length):
        action = neuralNetworkOutput(NN, obsArray)
        if action > 2:
            action = 2
        elif action < -2:
            action = -2
        obs, reward, done, info = env.step([action])
        obsArray = np.array(obs)
        env.render()
        if done:
            break

#Save performance:
print("Saving performance...")
file = open(filepath, "a")

lastline = []
lastline.append(str(avg_reward)+",")
lastline.append(str(num_testing_iterations)+"\n")
file.writelines(lastline)
file.close()

env.close()