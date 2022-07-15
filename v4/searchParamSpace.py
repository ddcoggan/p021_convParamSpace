import os
import torch
from torch import nn
import sys
import numpy as np
import functools
import random
import datetime
from copy import deepcopy
import pickle
import glob
import math

sys.path.append('/mnt/HDD12TB/masterScripts/DNN/BrainScore/model-tools-master')
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment

sys.path.append('/mnt/HDD12TB/masterScripts/DNN/BrainScore/brain-score-master')
from brainscore import score_model

sys.path.append('/mnt/HDD12TB/masterScripts/DNN')
from train import train

overwrite = True
paramLimit = (1e7,)
nLayers = 16
nIters = 32
nEpochs = 1
learningRate = .01
optimizerName = 'SGD'
batchSize = 64
workers = 8
trainset = 'imagenet1000'
nCategories = int(trainset[8:])


modelFiles=glob.glob('/home/dave/.result_caching/**/*cognet*', recursive = True)
for file in modelFiles:
    os.remove(file)

config = {'stride': {'mean': 1, 'slope': 0},
          'kernelSize': {'mean': 8, 'slope': 0},
          'nFeatures': {'mean': 128, 'slope': 0}}
middleLayer = int(nLayers / 2)
for feature in config:
    fmean = config[feature]['mean']
    fslope = config[feature]['slope']
    config[feature]['values'] = [int(fmean + (l-middleLayer) * fslope) for l in range(nLayers)]
print(config)

class CogNet(nn.Module):

    def __init__(self, nLayers, nFeatures, kernelSize, stride, nCats):
        super(CogNet, self).__init__()

        layers = []
        for l in range(nLayers):
            if l == 0:
                inFeatures = 3
            else:
                inFeatures = nFeatures[l-1]
            padding = int(np.floor(kernelSize[l] / 2))
            layer = nn.Conv2d(inFeatures, nFeatures[l], kernel_size=(kernelSize[l]), stride=(stride[l]), padding=padding)
            layers.append(layer)
            layers.append(nn.ReLU(inplace=True))
            if l % 4 == 3:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)

        # get output size by passing through random image
        input = torch.rand(1, 3, 224, 224)
        output = self.features(input)
        outSize = int(np.prod(output.shape))

        self.classifier = nn.Sequential(
            nn.Linear(outSize, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, nCats))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def nParams(model):
    params = 0
    for p, parameter in enumerate(model.parameters()):
        if len(parameter.shape) == 4:  # if layer is convolutional
            nOutFeatures = parameter.shape[0]
            kernelSize = parameter.shape[2]
            layerParams = nOutFeatures * kernelSize ** 2
            params += np.prod(parameter.shape)
    return params

# set up new results file or continue from last iteration
resPath = 'results.pkl'
if os.path.isfile(resPath) and not overwrite:
    results = pickle.load(open(resPath, 'rb'))
    lastConfig = deepcopy(results['allConfigs'][-1])
else:
    results = {'allScores': [],
               'allConfigs': [],
               'improvingScores': [],
               'improvingConfigs': []}
    lastConfig = deepcopy(config)
lastIter = len(results['allScores'])

# begin parameter search
for iter in range(lastIter, nIters):


    ### DEFINE ARCHITECTURE ###

    newConfig = deepcopy(lastConfig)
    if iter > 0:

        parameter = np.random.choice(['feature','kernel'])# make change to length of feature maps
        if parameter == 'feature':
            slopeUpDown = np.random.choice([1,-1]) # randomly increase or decrease slope
            newConfig['nFeatures']['slope'] += (2 * slopeUpDown)

        elif parameter == 'kernel':
            slopeUpDown = np.random.choice([1,-1]) # randomly increase or decrease slope
            newConfig['kernelSize']['slope'] += (.25 * slopeUpDown)

        # calculate new set of params
        middleLayer = int(nLayers / 2)
        for feature in newConfig:
            fmean = newConfig[feature]['mean']
            fslope = newConfig[feature]['slope']
            newConfig[feature]['values'] = [int(fmean + (l - middleLayer) * fslope) for l in range(nLayers)]
        print('New architecture created:\n', newConfig)

    if newConfig not in results['allConfigs']:

        ### CONSTRUCT MODEL ###

        #scores = np.empty(3) # used for running multiple iterations of each model
        #for rep in range(3):
        model = CogNet(nLayers,
                       newConfig['nFeatures']['values'],
                       newConfig['kernelSize']['values'],
                       newConfig['stride']['values'],
                       nCategories)
        nParameters = nParams(model)


        '''
        # increase parameters up to specified limit
        paramLimit = nParameters
        while nParameters < paramLimit:
            paramType = random.choice(list(newConfig.keys()))
            paramIdx = random.choice(np.arange(len(newConfig[paramType])))
            newConfig[paramType][paramIdx] += 1
            model = CogNet(nLayers,
                           newConfig['nFeatures'],
                           newConfig['kernelSize'],
                           newConfig['stride'])
            nParameters = nParams(model)
    
        while nParameters > paramLimit:
            paramType = random.choice(list(newConfig.keys()))
            paramIdx = random.choice(np.arange(len(newConfig[paramType])))
            newConfig[paramType][paramIdx] -= 1
            model = CogNet(nLayers,
                           newConfig['nFeatures'],
                           newConfig['kernelSize'],
                           newConfig['stride'])
            nParameters = nParams(model)
        '''


        ### TRAIN NETWORK ###

        outDir = os.path.join(f'models/model_{iter:03}')

        # get restart from file if necessary
        weightFiles = sorted(glob.glob(os.path.join(outDir, 'params/*.pt')))
        if 0 < len(weightFiles) and overwrite == False:
            restartFrom = weightFiles[-1]
        else:
            restartFrom = None

        # print out these values during training
        printOut = {'iteration': iter,
                    'nParams': nParameters}

        # call script
        if len(weightFiles) < nEpochs + 1 or overwrite:
            model = train(modelName=f'model_{iter:03}', model=model, datasetPath=trainset, learningRate=learningRate, optimizerName=optimizerName, batchSize=batchSize,
                          nEpochs=nEpochs, restartFrom=restartFrom, workers=workers, outDir=outDir, printOut=printOut, returnModel=True, skipZeroth=True)


        ### MEASURE BRAIN SCORE ###

        # put model on single GPU to not confuse brainscore
        model = model.module

        preprocessing = functools.partial(load_preprocess_images, image_size=224)
        identifier = f'cognet{iter:02}' #f'cognet{iter:02}_{rep}'
        activations_model = PytorchWrapper(identifier=identifier, model=model, preprocessing=preprocessing)
        modelCommit = ModelCommitment(identifier=identifier, activations_model=activations_model,layers=['features.33'])
        result = score_model(model_identifier=identifier, model=modelCommit, benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
        score = result[0].item()
        #scores[rep] = result[0].item()
        #score = np.mean(scores)


        ### RECORD RESULTS ###

        results['allConfigs'].append(newConfig)
        results['allScores'].append(score)

        print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Iteration: {iter+1}/{nIters} | nFeatures: {newConfig["nFeatures"]} | kernelSizes: {newConfig["kernelSize"]} | brainScore (conv5/IT): {score} |')

        if iter == 0:
            results['improvingConfigs'].append(deepcopy(newConfig))
            results['improvingScores'].append(score)
            lastConfig = deepcopy(newConfig)
        elif score > results['improvingScores'][-1]:
            print('score improved')
            results['improvingConfigs'].append(deepcopy(newConfig))
            results['improvingScores'].append(score)
            lastConfig = deepcopy(newConfig)
        else:
            print('score not improved')

        pickle.dump(results, open(f'results.pkl', 'wb'))
    else:
        print('this config already attempted, trying a different one...')
