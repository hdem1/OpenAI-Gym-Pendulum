import gym
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from os.path import exists, expanduser
import math

env = gym.make('Pendulum-v1')

#CONSTANTS:
num_generations = 20
num_children = 500
num_layers = 2
input_space = 3
neurons_per_layer = 4
output_space = 1
weight_max_magnitude = 5
bias_max_magnitude = 5
#min_exploration = 0.2
tests_per_network = 10
max_episode_length = 200
render_demos = True
final_testing_iterations = 1000
final_display_iterations = 5
alwaysSaving = True
askAboutSaving = True
num_reproducing = 5
reproduction_ratios = [0.3, 0.25, 0.2, 0.15, 0.1]
filteringStarts = True
reduceInput = True
if reduceInput:
    input_space = 2

exp_limit = 20

num_functions = 6
function_list = ["linear", "abs", "relu", "tanh"]
function_weights = [4,1,1,0]
tot = sum(function_weights)
for i in range(len(function_weights)):
    function_weights[i] /= tot
func_parameter_max_magnitude = 2


#best_NN = np.zeros((neurons_per_layer, input_space))
#for l in range(num_layers-1):
#    np.append(best_NN, np.zeros((neurons_per_layer, neurons_per_layer)))
#np.append(best_NN,np.zeros(neurons_per_layer))

# DEFINING FUNCTIONS:
#Get output of neuronal network:
def getNeuronOutput(neuron, last_layer):
    value = 0
    #Weights:
    for j in range(len(last_layer)):
        value += neuron[j] * last_layer[j] #weights
    #Bias:
    value += neuron[neurons_per_layer] #bias
    #Activation Function:
    function_type = int(neuron[neurons_per_layer+1])
    if function_list[function_type] == "linear": #Linear:
        return neuron[neurons_per_layer+2] * value
    elif function_list[function_type] == "abs": #Absolute Value + Linear:
        return abs(value) * neuron[neurons_per_layer+2]
    elif function_list[function_type] == "relu": #ReLu + Linear:
        return max(neuron[neurons_per_layer+2] * value, 0)
    elif function_list[function_type] == "tanh": #Tanh + Linear:
        return math.tanh(value) * neuron[neurons_per_layer+2]
    return

def neuralNetworkOutput(NN, input):
    #print("neural network output starting")
    last_layer = input
    for l in range(num_layers):
        #print(last_layer)
        next_layer = np.zeros(neurons_per_layer)
        for i in range(neurons_per_layer):
            #value = 0
            #for j in range(len(last_layer)):
            #    value += NN[l*neurons_per_layer + i][j] * last_layer[j] #weights
            #next_layer[i] = value + NN[l*neurons_per_layer+i][neurons_per_layer] #bias
            next_layer[i] = getNeuronOutput(NN[l*neurons_per_layer+i], last_layer)
        last_layer = next_layer
    #print("neural network output finished")
    return getNeuronOutput(NN[num_layers*neurons_per_layer], last_layer)
    #output = 0
    #print(last_layer)
    #for i in range(len(last_layer)):
    #    output += last_layer[i] * NN[num_layers*neurons_per_layer][i]
    #output += NN[num_layers*neurons_per_layer][len(last_layer)]
    #return output
    

def getIndex(weighted_array, value):
    if value > 1:
        print("ERROR")
    #print("starting index")
    index = 0
    while (sum(weighted_array[:(index+1)]) < value):
        index += 1
    #print("got index")
    return index

# Mutate the neural network to make a child:
def mutateNeuralNetwork(NN, randValue): #randValue is between 0 and 1
    #print("start mutating")
    newNN = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer+3))
    for l in range(num_layers*neurons_per_layer+1):
        for i in range(neurons_per_layer):
            oldValue1 = NN[l][i]
            newValue1 = oldValue1 + (1-2*rand.random())*randValue*weight_max_magnitude
            if newValue1 > weight_max_magnitude:
                newValue1 = weight_max_magnitude
            if newValue1 < -1*weight_max_magnitude:
                newValue1 = -1 * weight_max_magnitude
            newNN[l][i] = newValue1

        oldValue2 = NN[l][neurons_per_layer]
        newValue2 = oldValue2 + (1-2*rand.random())*randValue*bias_max_magnitude
        if newValue2 > bias_max_magnitude:
            newValue2 = bias_max_magnitude
        if newValue2 < -1*bias_max_magnitude:
            newValue2 = -1 * bias_max_magnitude
        newNN[l][neurons_per_layer] = newValue2

        
        #if (type(NN[l][neurons_per_layer] == int)):
        #    newFuncValue = num_functions * rand.random()
        #    newFuncParameter = (1-2*rand.random())*randValue*func_parameter_max_magnitude
        #else:
        funcValue = int(NN[l][neurons_per_layer+1])
        newFuncValue = funcValue
        if rand.random() < (1-function_weights[funcValue])*randValue: #1 - (num_functions-1)/num_functions*randValue:
            newFuncValue = rand.random()
            while getIndex(function_weights, newFuncValue) == funcValue:
                newFuncValue = rand.random()
            newFuncValue = getIndex(function_weights, newFuncValue)
        funcParameter = NN[l][neurons_per_layer+2]
        if (newFuncValue == funcValue):
            newFuncParameter = funcParameter + (1-2*rand.random())*randValue*func_parameter_max_magnitude
            newFuncParameter = min(newFuncParameter, func_parameter_max_magnitude)
            newFuncParameter = max(newFuncParameter, -1*func_parameter_max_magnitude) 
        else: 
            newFuncParameter = (1-2*rand.random())*randValue*func_parameter_max_magnitude
        newNN[l][neurons_per_layer+1] = newFuncValue
        newNN[l][neurons_per_layer+2] = newFuncParameter
    #print("done mutating")
    return newNN 

#Setting up neural network:
best_NNs = np.zeros((num_reproducing, neurons_per_layer*num_layers+output_space, neurons_per_layer+3))
for i in range(num_reproducing):
    best_NNs[i] = mutateNeuralNetwork(best_NNs[i], 1)

#print("Mutation test:")

#print(best_NNs[0])
#print(mutateNeuralNetwork(best_NNs[0],0.1))

#print("Neural Network Test:")
#print(best_NNs[0])
#s = 0
#for i in range(neurons_per_layer+1):
#    s+= best_NNs[0][i][neurons_per_layer]
#print(s)
#print(neuralNetworkOutput(best_NNs[0], np.zeros(input_space)))
#input("continue?")
print()
for gen in range(num_generations):
    bestRewards = np.full(num_reproducing,-200*17)
    avg_of_avg = 0
    best_NNs_in_generation = np.zeros((num_reproducing, neurons_per_layer*num_layers+output_space, neurons_per_layer+3))
    #maxRValue =  1 - (1-min_exploration)*(gen-start_evolving_gen)/(num_generations-start_evolving_gen)
    maxRValue =  (1 - gen/num_generations)*(1-gen/num_generations) #Squared for faster decrease
    print("Generation #", (gen+1), ":")
    print("max rValue =", maxRValue)
    print("Simulating generation - [", end = '', flush = True)
    lastPrint = 0
    for child in range(num_children):
        if child - lastPrint >= num_children / 10:
            print("*", end='', flush = True)
            lastPrint = child
        # Mutate nueral network
        index = num_reproducing
        while (sum(reproduction_ratios[:index])*num_children > child):
            index -= 1
        #rValue = 1 - (1-min_exploration)*(gen-start_evolving_gen)/(num_generations-start_evolving_gen)
        rValue = maxRValue * math.floor(child-sum(reproduction_ratios[:index])*num_children)/(num_children*reproduction_ratios[index])
        #print(rValue)
        newNN = mutateNeuralNetwork(best_NNs[index], rValue)
        #print(newNN)

        #Make variables:
        avg_reward = 0

        # Simulate:
        for test in range(tests_per_network):
            #Make iteration of the environment
            obs = env.reset()
            obsArray = np.array(obs)
            while filteringStarts and obsArray[1] > (-0.75+1.75*gen/num_generations) and abs(obsArray[2]) < (2+6*gen/num_generations):
                obs = env.reset()
                obsArray = np.array(obs)
            for step in range(max_episode_length):
                if reduceInput:
                    obsArray = [obsArray[2], math.atan(obsArray[1]/obsArray[0])]
                value = neuralNetworkOutput(newNN, obsArray)
                #print(value)
                if abs(value) < exp_limit and abs(value) > 1/exp_limit:
                    #print("start")
                    action = 2*(2/(1+math.exp(-2*value)) - 1)
                    #print("end")
                elif abs(value) >= exp_limit:
                    action = 2*value/abs(value)
                else:
                    action = 0
                #if action > 2:
                #    action = 2
                #elif action < -2:
                #    action = -2    
                #print(action)
                obs, reward, done, info = env.step([action])
                #print("action completed")
                obsArray = np.array(obs)
                #env.render()
                avg_reward += reward
                if done:
                    #print("done")
                    break
        avg_reward /= tests_per_network
        #Evaluate/save
        #print("Reward ", child, " = ", total_good_rewards)
        #print(avg_reward)
        avg_of_avg += avg_reward
        if avg_reward > bestRewards[num_reproducing-1]:
            #print(bestRewards)
            #print("----------------------------")
            #print(best_NNs_in_generation)
            index = num_reproducing - 1
            while (avg_reward > bestRewards[index - 1]):
                best_NNs_in_generation[index] = best_NNs_in_generation[index - 1]
                bestRewards[index] = bestRewards[index - 1]
                index -= 1
                if (index == 0):
                    break
            bestRewards[index] = avg_reward
            best_NNs_in_generation[index] = newNN
            #print(bestRewards)
    avg_of_avg /= num_children
    print("] - completed!")
    print("Best rewards = ", bestRewards)
    print("Generation average reward =", avg_of_avg)
    print()
    #print("Best reward index = ", bestIndex)
    best_NNs = best_NNs_in_generation
    #print(best_NNs_in_generation[0])
    #print("Best NN = ", best_NN)
    #Run demo:
    if render_demos:
        obs = env.reset()
        obsArray = np.array(obs)
        while filteringStarts and obsArray[1] > (-0.75+1.75*gen/num_generations) and abs(obsArray[2]) < (2+6*gen/num_generations):
            obs = env.reset()
            obsArray = np.array(obs)
        for step in range(max_episode_length):
            if reduceInput:
                obsArray = [obsArray[2], math.atan(obsArray[1]/obsArray[0])]
            value = neuralNetworkOutput(best_NNs[0], obsArray)
            if abs(value) < exp_limit and abs(value) > 1/exp_limit:
                #print("start")
                action = 2*(2/(1+math.exp(-2*value)) - 1)
                #print("end")
            elif abs(value) >= exp_limit:
                action = 2*value/abs(value)
            else:
                action = 0
            #if action > 2:
            #    action = 2
            #elif action < -2:
            #    action = -2
            obs, reward, done, info = env.step([action])
            obsArray = np.array(obs)
            env.render()
            if done:
                break


#Final Model Testing:
best_NN = best_NNs[0]
print()
print("Testing Final Model...")
print()
avg_reward = 0
for iteration in range(final_testing_iterations):
    obs = env.reset()
    for step in range(max_episode_length):
        if reduceInput:
            obsArray = [obsArray[2], math.atan(obsArray[1]/obsArray[0])]
        value = neuralNetworkOutput(best_NN, obsArray)
        if abs(value) < exp_limit and abs(value) > 1/exp_limit:
            #print("start")
            action = 2*(2/(1+math.exp(-2*value)) - 1)
            #print("end")
        elif abs(value) >= exp_limit:
            action = 2*value/abs(value)
        else:
            action = 0
        #if action > 2:
        #    action = 2
        #elif action < -2:
        #    action = -2
        obs, reward, done, info = env.step([action])
        obsArray = np.array(obs)
        avg_reward += reward
        if done:
            break

avg_reward /= final_testing_iterations
print("Final Model Results:")
print()
print("Average Reward =", avg_reward)
print()

#Displaying Final Model
print("Displaying sample runs of final model....")
for iteration in range(final_display_iterations):
    obs = env.reset()
    for step in range(max_episode_length):
        if reduceInput:
            obsArray = [obsArray[2], math.atan(obsArray[1]/obsArray[0])]
        obsArray = np.array(obs)
        value = neuralNetworkOutput(best_NN, obsArray)
        if abs(value) < exp_limit and abs(value) > 1/exp_limit:
            #print("start")
            action = 2*(2/(1+math.exp(-2*value)) - 1)
            #print("end")
        elif abs(value) >= exp_limit:
            action = 2*value/abs(value)
        else:
            action = 0
        #if action > 2:
        #    action = 2
        #elif action < -2:
        #    action = -2
        obs, reward, done, info = env.step([action])
        env.render()
        if done:
            break

#Saving Final Model
if askAboutSaving:
    response = input("Do you want to save the moodel: [y/n]")
    if response == "y":
        alwaysSaving = True
    else:
        alwaysSaving = False

if alwaysSaving:
    #Creating file:
    print()
    folder = expanduser("~/Documents/Random Coding Projects/MachineLearningExperiments/OpenAI-Gym-Pendulum/Saved Networks/")
    filename = "P"
    filename = filename + "_gens-" + str(num_generations)
    filename = filename + "_children-"+str(num_children) 
    filename = filename + "_layers-" + str(num_layers)
    filename = filename +"_layerHeight-"+str(neurons_per_layer)
    filename = filename + "_networkTests-"+str(tests_per_network)
    filename = filename + "_wMax-"+str(weight_max_magnitude)
    filename = filename + "_bMax-"+str(bias_max_magnitude)
    filename = filename + "_funcs-"+str(num_functions)
    filename = filename + "_paramMax-"+str(func_parameter_max_magnitude)
    filename = filename + "_filter-"+str(filteringStarts)
    if exists(folder + filename + ".txt"):
        value = 1
        while (exists(folder +filename + "_"+str(value))):
            value+=1
        filename = filename + "_" + str(value)
    filename = filename +".txt"
    file = open(folder+filename, "w")
    print("Creating file....")
    print("Filename =", filename)

    #Writing data:
    #First line = network constants:
    print("Saving constants...")
    firstline = []
    firstline.append(str(neurons_per_layer)+",")
    firstline.append(str(num_layers)+",")
    firstline.append(str(output_space)+"\n")
    file.writelines(firstline)
    #Following lines = neural network
    print("Saving final neural network...")
    for l in range(neurons_per_layer*num_layers+output_space):
        line = []
        for i in range(neurons_per_layer):
            line.append(str(best_NN[l][i])+";")
            #line.append(str(best_NN[l][i][0])+",")
            #line.append(str(best_NN[l][i][1])+";")
        line.append(str(best_NN[l][neurons_per_layer])+";")
        line.append(str(best_NN[l][neurons_per_layer+1])+",")
        line.append(str(best_NN[l][neurons_per_layer+2])+"\n")
        file.writelines(line)
    #All following lines = training performances
    print("Saving performance statistics...")
    lastline = []
    lastline.append(str(avg_reward)+",")
    lastline.append(str(final_testing_iterations)+"\n")
    file.writelines(lastline)

    file.close()
env.close()