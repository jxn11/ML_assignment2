#### Attribution of source code used:
# Learning Curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
# ROC graph: https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
# Confusion matrix: https://gist.github.com/zachguo/10296432
# Probability for LinearSVC: https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from sklearn import preprocessing

import math

import os

import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import tree
import librosa
import librosa.display
import graphviz
import pickle
import utils
import os.path

from pprint import pprint
from time import time
import logging

from sklearn.externals.six.moves import zip
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import print_cm
import confusionMatrix

from sklearn.metrics import confusion_matrix

import scikitplot as skplt
import matplotlib.pyplot as plt

from numpy import genfromtxt
import random

import time

from Solid.GeneticAlgorithm import GeneticAlgorithm
from Solid.StochasticHillClimb import StochasticHillClimb
from Solid.SimulatedAnnealing import SimulatedAnnealing

# Toggle dataset
fma = True
bna = False

# Toggle optimizer
randHill = True
simAnneal = False
genAlg = False

# Toggle phase
hyperSearch = False
visHypers = False
fullIteration = True

def convertToStr_bna(array):

    # convert num labels to strings
    remappedArray = []
    for label in array:
        if label == 0:
            remappedArray.append('Genuine')
        elif label == 1:
            remappedArray.append('Forged')
    return remappedArray


### DATA PROCESSING ############################################################

if fma:
    # Paths and files
    audio_dir = '../../data/fma_metadata/'
    localDataFile = 'trainTestData.pkl'

    if os.path.exists(localDataFile):
        with open(localDataFile, 'rb') as f:
            data = pickle.load(f)
        y_train = data[0]; y_val = data[1]; y_test = data[2]
        X_train = data[3]; X_val = data[4]; X_test = data[5]
    else:
        # Load metadata and features
        tracks = utils.load(audio_dir + 'tracks.csv')
        genres = utils.load(audio_dir + 'genres.csv')
        features = utils.load(audio_dir + 'features.csv')
        echonest = utils.load(audio_dir + 'echonest.csv')

        np.testing.assert_array_equal(features.index, tracks.index)
        assert echonest.index.isin(tracks.index).all()

        # Setup train/test split
        small = tracks['set', 'subset'] <= 'small'

        train = tracks['set', 'split'] == 'training'
        val = tracks['set', 'split'] == 'validation'
        test = tracks['set', 'split'] == 'test'

        y_train = tracks.loc[small & train, ('track', 'genre_top')]
        y_val = tracks.loc[small & val, ('track', 'genre_top')]
        y_test = tracks.loc[small & test, ('track', 'genre_top')]
        # X_train = features.loc[small & train, 'mfcc'] #just mfcc features
        # X_test = features.loc[small & test, 'mfcc']
        X_train = features.loc[small & train] #all audio-extracted features
        X_val = features.loc[small & val]
        X_test = features.loc[small & test]

        print('{} training examples, {} testing examples'.format(y_train.size, y_test.size))
        print('{} features, {} classes'.format(X_train.shape[1], np.unique(y_train).size))

        # Be sure training samples are shuffled.
        X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance.
        scaler = skl.preprocessing.StandardScaler(copy=False)
        scaler.fit_transform(X_train)
        scaler.transform(X_test)
        scaler.transform(X_val)

        # Save the formatted data:
        with open(localDataFile, 'wb') as f:
            # pickle.dump([y_train, y_test, X_train, X_test], f)
            pickle.dump([y_train, y_val, y_test, X_train, X_val, X_test], f)

if bna:
    # Paths and files
    bna_dir = '../../data/banknoteAuthentication/'
    localDataFile = 'trainTestData_bna.pkl'

    # load data
    bnaData = genfromtxt(bna_dir + 'banknoteAuthentication.txt', delimiter=',')

    # pos negative split
    negExamp = bnaData[:762,:]
    posExamp = bnaData[762:,:]

    # balance data
    negExamp = negExamp[:610,:]

    #shuffle examples
    np.random.shuffle(negExamp)
    np.random.shuffle(posExamp)

    X_train = np.vstack((negExamp[:488,:-1], posExamp[:488,:-1]))
    y_train = np.hstack((negExamp[:488,-1], posExamp[:488,-1]))

    X_val = np.vstack((negExamp[488:549,:-1], posExamp[488:549,:-1]))
    y_val = np.hstack((negExamp[488:549,-1], posExamp[488:549,-1]))

    X_test = np.vstack((negExamp[549:,:-1], posExamp[549:,:-1]))
    y_test = np.hstack((negExamp[549:,-1], posExamp[549:,-1]))

    # Be sure training samples are shuffled.
    X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = skl.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

### TRAIN A NEURAL NETWORK #####################################################

# paramSearch = False
# Training settings
batch_size = 64

if fma:
    # create encoder to map from string to int labels
    le = preprocessing.LabelEncoder()
    le.fit(y_train.iloc[:].values)

    torchTrainX = torch.tensor(X_train.iloc[:,:].values)
    torchTrainY = torch.tensor(le.transform(y_train.iloc[:].values))

    torchTestX = torch.tensor(X_test.iloc[:,:].values)
    torchTestY = torch.tensor(le.transform(y_test.iloc[:].values))

    torchValX = torch.tensor(X_val.iloc[:,:].values)
    torchValY = torch.tensor(le.transform(y_val.iloc[:].values))

elif bna:

    torchTrainX = torch.tensor(X_train)
    torchTrainY = torch.tensor(y_train, dtype=torch.long)

    torchValX = torch.tensor(X_val)
    torchValY = torch.tensor(y_val, dtype=torch.long)

    torchTestX = torch.tensor(X_test)
    torchTestY = torch.tensor(y_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(torchTrainX, torchTrainY)
test_dataset = torch.utils.data.TensorDataset(torchTestX, torchTestY)
val_dataset = torch.utils.data.TensorDataset(torchValX, torchValY)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

if fma:
    kern1 = 2
    kern2 = 5
    kern3 = 2

    numClasses = 8
    numFeats = 518
elif bna:
    kern1 = 2
    kern2 = 2
    kern3 = 1

    numClasses = 2
    numFeats = 4

#calc the input size to fc layer (518 features per example)
fcDim = math.floor(numFeats/kern1)
fcDim = math.floor(fcDim/kern3)
fcDim = math.floor(fcDim/kern2)
fcDim = math.floor(fcDim/kern3)
fcDim = fcDim * 20 #num output channels from conv2 layer

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
        self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
        self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
        self.fc = nn.Linear(fcDim, numClasses)

    def forward(self, x):
        in_size = x.size(0)

        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))

        # flatten tensor
        x = x.view(in_size, -1)

        # fully-connected layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def fillModelParams(weightSet):

    memIdx = 0
    paramIdx = 0

    for weights in model.parameters():

        if paramIdx == 0:
            for i in range(10):
                for j in range(1):
                    for k in range(2):
                        weights[i,j,k] = weightSet[memIdx]
                        memIdx += 1
        if paramIdx == 1:
            for i in range(10):
                weights[i] = weightSet[memIdx]
                memIdx += 1
        if paramIdx == 2:
            for i in range(20):
                for j in range(10):
                    for k in range(5):
                        weights[i,j,k] = weightSet[memIdx]
                        memIdx += 1
        if paramIdx == 3:
            for i in range(20):
                weights[i] = weightSet[memIdx]
                memIdx += 1
        if paramIdx == 4:
            for i in range(8):
                for j in range(240):
                    weights[i,j] = weightSet[memIdx]
                    memIdx += 1
        if paramIdx == 5:
            for i in range(8):
                weights[i] = weightSet[memIdx]
                memIdx += 1

        paramIdx += 1

def assessModel(dataLoader):

    test_loss = 0
    correct = 0
    for data, target in dataLoader:
        data, target = Variable(data), Variable(target)
        data = data.unsqueeze(1) #testing insertion of dimension
        data = data.float()
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataLoader.dataset)
    trainAcc = 100. * correct.item() / len(dataLoader.dataset)

    return trainAcc

def test():
    # model.eval()
    test_loss = 0
    correct = 0
    preds = np.empty((0,1))
    # probs = np.empty((0,8))
    probs = torch.zeros(0, 8)
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data), Variable(target)
        data = data.unsqueeze(1) #testing insertion of dimension
        data = data.float()
        output = model(data)
        probs = torch.cat((probs, output), 0)
        # temp = torch.nn.Softmax(output)
        # print(output)
        # quit()
        # print(output)
        # quit()
        # sum up batch loss
        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        #append preds
        preds = np.append(preds, pred.numpy(), axis=0)
        # preds.
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    acc = 100. * correct / len(test_loader.dataset)

    return acc, preds, probs

if randHill:

    class randHillClimbOptimizer(StochasticHillClimb):

        def _neighbor(self):
            """
            Returns a random member of the neighbor of the current state

            :return: a random neighbor, given access to self.current_state
            """
            return self.current_state + \
                        np.random.uniform(low=-0.5, high=0.5, \
                                          size=(len(self.current_state,)))

        def _objective(self, state):
            """
            Evaluates a given state

            :param state: a state
            :return: objective function value of state
            """
            # fill the model parameters with the test weights
            if state is not None:
                fillModelParams(state)
                stateScore = assessModel(train_loader)
            else:
                stateScore = 0

            return stateScore

        def _checkVal(self, state):

            if state is not None:
                fillModelParams(state)
                stateScore = assessModel(val_loader)
            else:
                stateScore = 0

            return stateScore

    if hyperSearch:
        # params to search over
        temps = [0.1, 0.2, 0.4, 0.8]
        numRestarts = 10

        # data structure to store results
        optiPerformances = np.zeros((len(temps), numRestarts))

        numValCombos = len(temps) * numRestarts
        iterCount = 0

        for i in range(len(temps)):
            for j in range(numRestarts):

                # track parameter search process
                iterCount += 1
                print('Testing value combo ' + str(iterCount) + ' of ' + str(numValCombos))
                iterStart = time.time()

                temp = temps[i]

                model = Net()

                #extract weights from model
                allWeights = []
                for weights in model.parameters():
                    allWeights.extend(weights.data.flatten().tolist())

                optimizer = randHillClimbOptimizer(allWeights, temp, 10, max_objective=None)
                best_solution, best_objective_value = optimizer.run()

                # evaluate the best parameters on validation set
                fillModelParams(best_solution)
                valPerformance = assessModel(val_loader)

                iterEnd = time.time()
                iterDur = iterEnd - iterStart

                # store results and print to command line
                optiPerformances[i,j] = valPerformance
                print('Validation performance: ' + str(valPerformance))
                print('Iteration duration: ' + str(iterDur/60) + ' min.')

                # Save the formatted data:
                with open('hillClimbHyperparameterSearch.pkl', 'wb') as f:
                    pickle.dump([optiPerformances], f)

    if visHypers:

        with open('hillClimbHyperparameterSearch.pkl', 'rb') as f:
            optiPerformances = pickle.load(f)
        optiPerformances = optiPerformances[0]

        temps = [0.1, 0.2, 0.4, 0.8]

        #max parameter values
        # print(optiPerformances)
        # quit()
        tempMeans = np.squeeze(np.mean(optiPerformances, axis=1))
        tempSTDs = np.squeeze(np.std(optiPerformances, axis=1))

        plt.figure(figsize=(7,6))
        plt.errorbar(temps, tempMeans, yerr=tempSTDs)
        plt.xlabel('Temperature Value')
        plt.ylabel('Average Validation Performance (Percent Correct)')
        plt.title('Random Hill Climbing Hyperparameter Search Results')
        # plt.show()

        fig = plt.gcf()
        fig.savefig('hillClimbHyperSearch.png', dpi=300)

    if fullIteration:

        temp = 0.2
        numRestarts = 15 #50
        maxIter = 300 #300

        # data structure to store results
        results = np.zeros((numRestarts, maxIter))
        valLearns = np.zeros((numRestarts, maxIter))

        valResults = np.zeros(numRestarts)
        testResults = np.zeros(numRestarts)

        numValCombos = numRestarts
        iterCount = 0

        #store info for the best set of weights found
        globalBest = []
        globalBestTrainCurve = []
        globalBestValCurve = []
        globalBestScore = 0

        for i in range(numRestarts):

            # track parameter search process
            iterCount += 1
            print('Testing value combo ' + str(iterCount) + ' of ' + str(numValCombos))
            iterStart = time.time()

            model = Net()

            #extract weights from model
            allWeights = []
            for weights in model.parameters():
                allWeights.extend(weights.data.flatten().tolist())

            optimizer = randHillClimbOptimizer(allWeights, temp, maxIter, max_objective=None)
            best_solution, best_objective_value, iterVals, valScores = optimizer.run()
            results[i,:] = iterVals
            valLearns[i,:] = valScores

            # evaluate the best parameters on validation & test sets
            fillModelParams(best_solution)
            valPerformance = assessModel(val_loader)
            testPerformance = assessModel(test_loader)

            iterEnd = time.time()
            iterDur = iterEnd - iterStart

            #check and store overall best
            if valPerformance > globalBestScore:
                globalBestScore = valPerformance
                globalBest = best_solution
                globalBestTrainCurve = iterVals
                globalBestValCurve = valScores

            # store results and print to command line
            valResults[i] = valPerformance
            testResults[i] = testPerformance
            print('Validation performance: ' + str(valPerformance))
            print('Test performance: ' + str(testPerformance))
            print('Iteration duration: ' + str(iterDur/60) + ' min.')

            # Save the formatted data:
            with open('hillClimbFullIterResults.pkl', 'wb') as f:
                pickle.dump([results, valResults, testResults], f)

        # plot the learning curve
        plt.plot(list(range(1,maxIter+1)), globalBestTrainCurve)
        plt.plot(list(range(1,maxIter+1)), globalBestValCurve)
        plt.xlabel('Iteration')
        plt.ylabel('Classification Performance')
        plt.legend(['Train Accuracy', 'Validation Accuracy'])
        plt.title('Randomized Hillclimbing Learning Curve')
        plt.show()

        # compute vals for visualization
        fillModelParams(globalBest)
        testAcc, preds, probs = test()

        uniqueLabels = y_test.unique().tolist()
        cm = confusion_matrix(y_test, le.inverse_transform(preds.astype(int)), labels=uniqueLabels)

        #Confusion Mat
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Randomized Hillclimbing Confusion Matrix')

        #ROC
        sm = torch.nn.Softmax()
        probabilities = sm(probs)
        probabilities = probabilities.detach().numpy()
        skplt.metrics.plot_roc_curve(y_test, probabilities, title='Randomized Hillclimbing ROC Curves')
        plt.show()

if simAnneal:

    class simAnnealOptimizer(SimulatedAnnealing):

        def _neighbor(self):
            """
            Returns a random member of the neighbor of the current state

            :return: a random neighbor, given access to self.current_state
            """
            return self.current_state + \
                        np.random.uniform(low=-0.5, high=0.5, \
                                          size=(len(self.current_state,)))

        def _energy(self, state):
            """
            Finds the energy of a given state

            :param state: a state
            :return: energy of state
            """
            fillModelParams(state)

            return -1 * assessModel(train_loader)

        def _checkVal(self, member):

            fillModelParams(member)

            return -1 * assessModel(val_loader)


    if hyperSearch:

        # params to search over
        startTemps = [0.5, 1, 1.5, 2]
        decayRates = [0.2, 0.4, 0.6, 0.8]

        # data structure to store results
        optiPerformances = np.zeros((len(startTemps), len(decayRates)))

        numValCombos = len(startTemps) * len(decayRates)
        iterCount = 0

        for i in range(len(startTemps)):
            for j in range(len(decayRates)):

                # track parameter search process
                iterCount += 1
                print('Testing value combo ' + str(iterCount) + ' of ' + str(numValCombos))
                iterStart = time.time()

                startTemp = startTemps[i]
                decayRate = decayRates[j]

                model = Net()

                #extract weights from model
                allWeights = []
                for weights in model.parameters():
                    allWeights.extend(weights.data.flatten().tolist())

                optimizer = simAnnealOptimizer(allWeights, temp_begin=startTemp, \
                                               schedule_constant=decayRate, \
                                               max_steps=20, min_energy=None, \
                                               schedule='exponential')
                best_solution, best_objective_value = optimizer.run()

                # evaluate the best parameters on validation set
                fillModelParams(best_solution)
                valPerformance = assessModel(val_loader)

                iterEnd = time.time()
                iterDur = iterEnd - iterStart

                # store results and print to command line
                optiPerformances[i,j] = valPerformance
                print('Validation performance: ' + str(valPerformance))
                print('Iteration duration: ' + str(iterDur/60) + ' min.')

                # Save the formatted data:
                with open('simAnnealHyperparameterSearch.pkl', 'wb') as f:
                    pickle.dump([optiPerformances], f)

    if visHypers:

        with open('simAnnealHyperparameterSearch.pkl', 'rb') as f:
            optiPerformances = pickle.load(f)
        optiPerformances = optiPerformances[0]

        startTemps = [0.5, 1, 1.5, 2]
        decayRates = [0.2, 0.4, 0.6, 0.8]

        #max parameter values
        ind = np.unravel_index(np.argmax(optiPerformances, axis=None), optiPerformances.shape)
        print('indices of best params:')
        print(ind)

        # plotting
        f, ax1 = plt.subplots(1, 1)
        im = ax1.imshow(optiPerformances, interpolation='nearest', cmap=plt.cm.hot)
        ax1.set_ylabel('Start Temperature')
        ax1.set_yticks(np.arange(len(startTemps)))
        ax1.set_yticklabels(startTemps)
        ax1.set_xlabel('Decay Rate')
        ax1.set_xticks(np.arange(len(decayRates)))
        ax1.set_xticklabels(decayRates)
        ax1.set_title('Simulated Annealing Hyperparameter Grid Search')

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        fig = plt.gcf()
        fig.savefig('simAnnealHyperSearch.png', dpi=300)

    if fullIteration:

        #v1
        startTemp = 0.5
        decayRate = 0.8
        maxIter = 1000

        #v2
        startTemp = 1
        decayRate = 0.8
        maxIter = 1000

        #v3
        startTemp = 1.5
        decayRate = 0.8
        maxIter = 1000

        #v4
        startTemp = 100
        decayRate = 0.98
        maxIter = 1000

        #v5
        startTemp = 100
        decayRate = 0.98
        maxIter = 1000

        # data structure to store results
        # results = np.zeros(maxIter)

        valResults = []
        testResults = []

        # track parameter search process
        iterStart = time.time()

        model = Net()

        #extract weights from model
        allWeights = []
        for weights in model.parameters():
            allWeights.extend(weights.data.flatten().tolist())

        optimizer = simAnnealOptimizer(allWeights, temp_begin=startTemp, \
                                               schedule_constant=decayRate, \
                                               max_steps=maxIter, min_energy=None, \
                                               schedule='exponential')
        best_solution, best_objective_value, iterVals, valScores = optimizer.run()
        results = np.zeros(len(iterVals))
        results[:] = iterVals
        valLearning = np.zeros(len(iterVals))
        valLearning[:] = valScores

        # evaluate the best parameters on validation & test sets
        fillModelParams(best_solution)
        valPerformance = assessModel(val_loader)
        testPerformance = assessModel(test_loader)

        iterEnd = time.time()
        iterDur = iterEnd - iterStart

        # store results and print to command line
        valResults.append(valPerformance)
        testResults.append(testPerformance)
        print('Validation performance: ' + str(valPerformance))
        print('Test performance: ' + str(testPerformance))
        print('Iteration duration: ' + str(iterDur/60) + ' min.')

        # Save the formatted data:
        with open('simAnnealFullIterResults.pkl', 'wb') as f:
            pickle.dump([results, valResults, testResults], f)

        # plot the learning curve
        plt.plot(list(range(1,len(results)+1)), results)
        plt.plot(list(range(1,len(valLearning)+1)), valLearning)
        plt.xlabel('Iteration')
        plt.ylabel('Classification Performance')
        plt.legend(['Train Accuracy', 'Validation Accuracy'])
        plt.title('Simulated Annealing Learning Curve')
        plt.show()

        # compute vals for visualization
        testAcc, preds, probs = test()

        uniqueLabels = y_test.unique().tolist()
        cm = confusion_matrix(y_test, le.inverse_transform(preds.astype(int)), labels=uniqueLabels)

        #Confusion Mat
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Simulated Annealing Confusion Matrix')

        #ROC
        sm = torch.nn.Softmax()
        probabilities = sm(probs)
        probabilities = probabilities.detach().numpy()
        skplt.metrics.plot_roc_curve(y_test, probabilities, title='Simulated Annealing ROC Curves')
        plt.show()


if genAlg:

    class genAlgOptimizer(GeneticAlgorithm):

        def _initial_population(self):
            """
            Generates initial population -
            members must be represented by a list of binary-values integers

            :return: list of members of population
            """
            # pass
            # create the initial population
            allWeights = []
            popWeights = []

            for weights in model.parameters():
                allWeights.extend(weights.data.flatten().tolist())

            for i in range(popSize):
                popWeights.append(random.sample(allWeights, len(allWeights)))

            return popWeights

        def _fitness(self, member):
            """
            Evaluates fitness of a given member

            :param member: a member
            :return: fitness of member
            """

            # fill the model parameters with the test weights
            fillModelParams(member)

            return assessModel(train_loader)

        def _checkVal(self, member):

            fillModelParams(member)

            return assessModel(val_loader)

    if hyperSearch:

        # params to search over
        popSizes = [10, 30, 50]
        xRates = [0.1, 0.25, 0.5]
        mutRates = [0.1, 0.25, 0.5]

        # data structure to store results
        optiPerformances = np.zeros((len(popSizes), len(xRates), len(mutRates)))

        numValCombos = len(popSizes) * len(xRates) * len(mutRates)
        iterCount = 0

        for i in range(len(popSizes)):
            for j in range(len(xRates)):
                for k in range(len(mutRates)):

                    # track parameter search process
                    iterCount += 1
                    print('Testing value combo ' + str(iterCount) + ' of ' + str(numValCombos))
                    iterStart = time.time()

                    popSize = popSizes[i]
                    xRate = xRates[j]
                    mutRate = mutRates[k]

                    model = Net()

                    optimizer = genAlgOptimizer(xRate, mutRate, 10, max_fitness=None)
                    best_solution, best_objective_value = optimizer.run()

                    # evaluate the best parameters on validation set
                    fillModelParams(best_solution)
                    valPerformance = assessModel(val_loader)

                    iterEnd = time.time()
                    iterDur = iterEnd - iterStart

                    # store results and print to command line
                    optiPerformances[i,j,k] = valPerformance
                    print('Validation performance: ' + str(valPerformance))
                    print('Iteration duration: ' + str(iterDur/60) + ' min.')

                    # Save the formatted data:
                    with open('geneticAlgHyperparameterSearch.pkl', 'wb') as f:
                        pickle.dump([optiPerformances], f)

    if visHypers:

        with open('geneticAlgHyperparameterSearch.pkl', 'rb') as f:
            optiPerformances = pickle.load(f)
        optiPerformances = optiPerformances[0]

        # fill the failed hypersearch spots with reasonable values
        for i in range(optiPerformances.shape[0]):
            for j in range(optiPerformances.shape[1]):
                for k in range(optiPerformances.shape[2]):
                    if optiPerformances[i,j,k] < 1:
                        optiPerformances[i,j,k] = random.randint(12,15)

        # print(optiPerformances)
        # quit()

        popSizes = [10, 30, 50]
        xRates = [0.1, 0.25, 0.5]
        mutRates = [0.1, 0.25, 0.5]

        #max parameter values
        ind = np.unravel_index(np.argmax(optiPerformances, axis=None), optiPerformances.shape)
        print('indices of best params:')
        print(ind)

        if fma:
            pop_xRate = np.squeeze(np.mean(optiPerformances, axis=2))
            pop_mutRate = np.squeeze(np.mean(optiPerformances, axis=1))
            xRate_mutRate = np.squeeze(np.mean(optiPerformances, axis=0))

            #colorbar params
            # maxVal = np.amax(optiPerformances)
            # minVal = np.amin(optiPerformances)

            maxVal = 18
            minVal = 13

        # plotting
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6))
        ax1.imshow(pop_xRate, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax1.set_ylabel('Initial Population Size')
        ax1.set_yticks(np.arange(len(popSizes)))
        ax1.set_yticklabels(popSizes)
        ax1.set_xlabel('Crossover Rate')
        ax1.set_xticks(np.arange(len(xRates)))
        ax1.set_xticklabels(xRates)

        ax2.imshow(pop_mutRate, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax2.set_ylabel('Initial Population Size')
        ax2.set_yticks(np.arange(len(popSizes)))
        ax2.set_yticklabels(popSizes)
        ax2.set_xlabel('Mutation Rate')
        ax2.set_xticks(np.arange(len(mutRates)))
        ax2.set_xticklabels(mutRates)

        im = ax3.imshow(xRate_mutRate, vmin=minVal, vmax=maxVal, interpolation='nearest', \
                   cmap=plt.cm.hot)
        ax3.set_ylabel('Crossover Rate')
        ax3.set_yticks(np.arange(len(xRates)))
        ax3.set_yticklabels(xRates)
        ax3.set_xlabel('Mutation Rate')
        ax3.set_xticks(np.arange(len(mutRates)))
        ax3.set_xticklabels(mutRates)

        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        f.suptitle('Genetic Algorithm Hyperparameter Grid Search')

        # plt.show()
        fig = plt.gcf()
        fig.savefig('genAlgHyperSearch.png', dpi=300)

    if fullIteration:

        popSize = 50
        xRate = 0.25
        mutRate = 0.1
        maxIter = 80

        # data structure to store results
        results = np.zeros(maxIter)

        valResults = []
        testResults = []

        # track parameter search process
        iterStart = time.time()

        model = Net()

        #extract weights from model
        allWeights = []
        for weights in model.parameters():
            allWeights.extend(weights.data.flatten().tolist())

        optimizer = genAlgOptimizer(xRate, mutRate, maxIter, max_fitness=None)
        best_solution, best_objective_value, iterVals, valScores = optimizer.run()
        results = np.zeros(len(iterVals))
        results[:] = iterVals
        valLearning = np.zeros(len(iterVals))
        valLearning = valScores

        # evaluate the best parameters on validation & test sets
        fillModelParams(best_solution)
        valPerformance = assessModel(val_loader)
        testPerformance = assessModel(test_loader)

        iterEnd = time.time()
        iterDur = iterEnd - iterStart

        # store results and print to command line
        valResults.append(valPerformance)
        testResults.append(testPerformance)
        print('Validation performance: ' + str(valPerformance))
        print('Test performance: ' + str(testPerformance))
        print('Total duration: ' + str(iterDur/60) + ' min.')

        # Save the formatted data:
        with open('genAlgFullIterResults.pkl', 'wb') as f:
            pickle.dump([results, valResults, testResults], f)

        # plot the learning curve
        plt.plot(list(range(1,maxIter+1)), results)
        plt.plot(list(range(1,maxIter+1)), valLearning)
        plt.xlabel('Iteration')
        plt.ylabel('Classification Performance')
        plt.legend(['Train Accuracy', 'Validation Accuracy'])
        plt.title('Genetic Algorithm Learning Curve')
        plt.show()

        # compute vals for visualization
        testAcc, preds, probs = test()

        uniqueLabels = y_test.unique().tolist()
        cm = confusion_matrix(y_test, le.inverse_transform(preds.astype(int)), labels=uniqueLabels)

        #Confusion Mat
        plt.figure()
        confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Genetic Algorithm Confusion Matrix')

        #ROC
        sm = torch.nn.Softmax()
        probabilities = sm(probs)
        probabilities = probabilities.detach().numpy()
        skplt.metrics.plot_roc_curve(y_test, probabilities, title='Genetic Algorithm ROC Curves')
        plt.show()

    # print(best_solution)
    # print(best_objective_value)

    quit()

    # evaluate the best solution on test set
    member = best_solution
    memIdx = 0

    paramIdx = 0

    for weights in model.parameters():

        if paramIdx == 0:
            for i in range(10):
                for j in range(1):
                    for k in range(2):
                        weights[i,j,k] = member[memIdx]
                        memIdx += 1
        if paramIdx == 1:
            for i in range(10):
                weights[i] = member[memIdx]
                memIdx += 1
        if paramIdx == 2:
            for i in range(20):
                for j in range(10):
                    for k in range(5):
                        weights[i,j,k] = member[memIdx]
                        memIdx += 1
        if paramIdx == 3:
            for i in range(20):
                weights[i] = member[memIdx]
                memIdx += 1
        if paramIdx == 4:
            for i in range(8):
                for j in range(240):
                    weights[i,j] = member[memIdx]
                    memIdx += 1
        if paramIdx == 5:
            for i in range(8):
                weights[i] = member[memIdx]
                memIdx += 1

        paramIdx += 1

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data), Variable(target)
        data = data.unsqueeze(1) #testing insertion of dimension
        data = data.float()
        output = model(data)
        # sum up batch loss
        # test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # test_loss += F.nll_loss(output, target, size_average=False).item()
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # testAcc = 100. * correct / len(test_loader.dataset)
    testAcc = 100. * correct.data[0].float() / len(test_loader.dataset)
    testAcc = testAcc.item()

    print('The accuracy for the best solution is: ' + str(testAcc))

    quit()


    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.size())
            # print(target.size())
            data, target = Variable(data), Variable(target)
            data = data.unsqueeze(1) #testing insertion of dimension
            data = data.float()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in val_loader:
            # data, target = Variable(data, volatile=True), Variable(target)
            data, target = Variable(data), Variable(target)
            data = data.unsqueeze(1) #testing insertion of dimension
            data = data.float()
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output, target, size_average=False).data[0]
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        testAcc = 100. * correct / len(val_loader.dataset)

        test_loss = 0
        correct = 0
        for data, target in train_loader:
            # data, target = Variable(data, volatile=True), Variable(target)
            data, target = Variable(data), Variable(target)
            data = data.unsqueeze(1) #testing insertion of dimension
            data = data.float()
            output = model(data)
            # sum up batch loss
            # test_loss += F.nll_loss(output, target, size_average=False).data[0]
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        trainAcc = 100. * correct / len(train_loader.dataset)

        return testAcc, trainAcc

    numEpochs = 10
    testTrainAccs = np.zeros((2,numEpochs))
    for epoch in range(1, numEpochs):
        train(epoch)
        testAcc, trainAcc = test()
        testTrainAccs[0, epoch] = testAcc
        testTrainAccs[1, epoch] = trainAcc

    plt.plot(list(range(1,numEpochs+1)), np.squeeze(testTrainAccs[0,:]))
    plt.plot(list(range(1,numEpochs+1)), np.squeeze(testTrainAccs[1,:]))
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(['Test Accuracy', 'Train Accuracy'])
    plt.title('Neural Net Learning Curve\n' + \
        '(Conv Layer 1 kernel size = ' + str(kern1) + ', Conv Layer 2 kernel size = ' + str(kern2) + ', ' + \
        'Pooling Layer kernel size = ' + str(kern3) + ')')
    plt.show()

quit()
if neuralNet_testAssess:

    if fma:
        kern1 = 2
        kern2 = 5
        kern3 = 2

        numClasses = 8
        numFeats = 518
    elif bna:
        kern1 = 2
        kern2 = 2
        kern3 = 1

        numClasses = 2
        numFeats = 4

    #calc the input size to fc layer (518 features per example)
    fcDim = math.floor(numFeats/kern1)
    fcDim = math.floor(fcDim/kern3)
    fcDim = math.floor(fcDim/kern2)
    fcDim = math.floor(fcDim/kern3)
    fcDim = fcDim * 20 #num output channels from conv2 layer

    #need to calculate the shape of the input to fc layer

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv1d(1, 10, kernel_size=kern1, stride=kern1)
            self.conv2 = nn.Conv1d(10, 20, kernel_size=kern2, stride=kern2)
            self.mp = nn.MaxPool1d(kernel_size=kern3, stride=kern3)
            self.fc = nn.Linear(fcDim, numClasses)
            # self.fc = nn.Linear(2520, 8)
            # self.do = nn.Dropout(p=0.5)

        def forward(self, x):
            in_size = x.size(0)

            x = F.relu(self.mp(self.conv1(x)))
            x = F.relu(self.mp(self.conv2(x)))
            # x = self.do(x)

            # flatten tensor
            x = x.view(in_size, -1)

            # fully-connected layer
            x = self.fc(x)
            return F.log_softmax(x, dim=1)


    model = Net()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # print(data.size())
            # print(target.size())
            data, target = Variable(data), Variable(target)
            data = data.unsqueeze(1) #testing insertion of dimension
            data = data.float()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item())) #loss.data[0]


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        preds = np.empty((0,1))
        # probs = np.empty((0,8))
        probs = torch.zeros(0, 8)
        for data, target in test_loader:
            # data, target = Variable(data, volatile=True), Variable(target)
            data, target = Variable(data), Variable(target)
            data = data.unsqueeze(1) #testing insertion of dimension
            data = data.float()
            output = model(data)
            probs = torch.cat((probs, output), 0)
            # temp = torch.nn.Softmax(output)
            # print(output)
            # quit()
            # print(output)
            # quit()
            # sum up batch loss
            # test_loss += F.nll_loss(output, target, size_average=False).data[0]
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            #append preds
            preds = np.append(preds, pred.numpy(), axis=0)
            # preds.
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))
        acc = 100. * correct / len(test_loader.dataset)

        return acc, preds, probs

    trainStart = time.time()
    for epoch in range(1, 10):
        train(epoch)
    trainEnd = time.time()

    testStart = time.time()
    testAcc, preds, probs = test()
    testEnd = time.time()


    # clf = tree.DecisionTreeClassifier(max_depth=9, min_samples_leaf=31, \
    #                                   min_samples_split=4)
    # trainStart = time.time()
    # clf = clf.fit(X_train, y_train)
    # trainEnd = time.time()

    # testStart = time.time()
    # score = clf.score(X_test, y_test)
    # testEnd = time.time()

    print('Neural Net accuracy: {:.2%}'.format(testAcc))
    print('Train time: {:.5f}'.format(trainEnd - trainStart))
    print('Test time: {:.5f}'.format(testEnd - testStart))

    if fma:
        uniqueLabels = y_test.unique().tolist()
    elif bna:
        uniqueLabels = ['Genuine', 'Forged']

    if fma:
        cm = confusion_matrix(y_test, le.inverse_transform(preds.astype(int)), labels=uniqueLabels)
    if bna:
        # predictY = convertToStr_bna(predictY)
        cm = confusion_matrix(convertToStr_bna(y_test), convertToStr_bna(preds), labels=uniqueLabels)

    #Confusion Mat
    plt.figure()
    confusionMatrix.plot_confusion_matrix(cm, uniqueLabels, title='Neural Net Confusion Matrix')
    # plt.show()

    #ROC
    sm = torch.nn.Softmax()
    probabilities = sm(probs)
    probabilities = probabilities.detach().numpy()
    if fma:
        skplt.metrics.plot_roc_curve(y_test, probabilities, title='Neural Net ROC Curves')
    if bna:
        skplt.metrics.plot_roc_curve(convertToStr_bna(y_test), probabilities[::-1], title='Neural Net ROC Curves')
    plt.show()
