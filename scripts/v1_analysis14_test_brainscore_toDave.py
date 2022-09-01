""" Summary
To test brainscore
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
import random
import time
import numpy
import scipy.io
import matplotlib.pyplot as plt
import tempfile
import shutil
import functools

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import collections
import itertools
import scipy.linalg as linalg

from scipy.linalg import get_blas_funcs
from torchsummary import summary
#from v1_utils import *
from brainscore import score_model
from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark

import sys
sys.path.insert(1, '/mnt/HDD12TB/masterScripts/DNN/BrainScore') # put the absolute path of BrainScore code
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from model_tools.brain_transformation import LayerMappedModel, TemporalIgnore, ModelCommitment, ProbabilitiesMapping
from model_tools.brain_transformation import LayerSelection
from model_tools.activations.pca import LayerPCA

def main(ROI):

    #### Parameters ####################################################################################################
    train_batch_size = 32
    val_batch_size = 32
    start_epoch = 0
    end_epoch = 70
    save_every_epoch = 10
    initial_learning_rate = 1e-3
    gpu_ids = [0,1]

    #### Create/Load model #############################################################################################
    model = models.alexnet(pretrained=True)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    elif len(gpu_ids) == 1:
        device = torch.device('cuda:%d'%(gpu_ids[0]))
        torch.cuda.set_device(device)
        model.cuda()
        model.to(device)

    #### define layers
    layers = ['features.1', 'features.4', 'features.7', 'features.9', 'features.11']

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier='cnn', model=model, preprocessing=preprocessing)
    hook = LayerPCA.hook(activations_model, n_components=300) # you will need pca for saving memory

    scores = numpy.zeros((len(layers), 2))
    for i, l in enumerate(layers): # you will need to change the paths below
        if os.path.exists('/home/tonglab/.result_caching/brainscore.score_model'):
            shutil.rmtree('/home/tonglab/.result_caching/brainscore.score_model')
        if os.path.exists('/home/tonglab/.result_caching/model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored'):
            shutil.rmtree('/home/tonglab/.result_caching/model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored')
        if os.path.exists('/home/tonglab/.result_caching/model_tools.activations.pca.LayerPCA._pcas'):
            shutil.rmtree('/home/tonglab/.result_caching/model_tools.activations.pca.LayerPCA._pcas')

        brainscore_model = LayerMappedModel('cnn', activations_model=activations_model, visual_degrees=8)  # layer -> region
        brainscore_model.commit(ROI, l)
        brainscore_model = TemporalIgnore(brainscore_model)  # ignore time_bins

        if ROI == 'V1':
            score = score_model(model_identifier=brainscore_model.identifier, model=brainscore_model, benchmark_identifier='movshon.FreemanZiemba2013public.V1-pls')
        elif ROI == 'V2':
            score = score_model(model_identifier=brainscore_model.identifier, model=brainscore_model, benchmark_identifier='movshon.FreemanZiemba2013public.V2-pls')
        elif ROI == 'V4':
            score = score_model(model_identifier=brainscore_model.identifier, model=brainscore_model, benchmark_identifier='dicarlo.MajajHong2015public.V4-pls')
        elif ROI == 'IT':
            score = score_model(model_identifier=brainscore_model.identifier, model=brainscore_model, benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')

        scores[i, :] = score.data
        print(i)

    hook.remove()
    scipy.io.savemat('python_v1_analysis14_test_brainscore_%s.mat' % (ROI),{'layers': layers, 'scores': scores})

if __name__ == '__main__':
    # main()

    main('IT')