import json
import os

import numpy as np
import torch
from torch import nn
from torch import optim

from learning_score_update_function import lsuf_network_module

# hyperparameters
inp_dim = 2  # input dimensions
out_dim = 1  # output dimensions
hid_dim = [100, 100, 100, 100]  # hidden unit dimensions
activation_function_array = [nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU, nn.LeakyReLU]  # activation function per layer
iterations = 1300  # train iterations or epochs
activation_function = 'LeakyReLU'  # for saving the model

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]
# train data path
path = '../work_dir1/train_data_2input.json'
with open(path, 'rb') as f:
    train_data = json.load(f)

# get data (inputs,labels) from json file and convert it to np arrays
inputs = []
labels = []
for i in train_data['data']:
    inputs.append(np.array([index for index in i[0]]))
    labels.append(np.array(i[1]))
inputs = np.array(inputs)
labels = np.array(labels)

# dataset
# data augmentation
fake_density = 0.0001
interval_fake = 0.94
size_fake_slice = 5000
fake_detection = np.mgrid[interval_fake:1:fake_density, 0.1:1:fake_density].reshape(2, -1).T
fake_track = np.mgrid[0.1:1:fake_density, interval_fake:1:fake_density].reshape(2, -1).T

# True positives : tp
tp = torch.tensor(inputs[labels == 1], dtype=torch.float32)
# False positives : fp
fp = torch.tensor(inputs[labels == 0], dtype=torch.float32)

# module definition
model = lsuf_network_module.sc_up_fc_nn(inp_dim, out_dim, hid_dim, activation_function_array)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# loss function
criterion = torch.nn.SmoothL1Loss()

# train and append loss value simultaneously
loss_values = []
model.train()
for e in range(iterations):
    # prepare the augmented data for adding
    slice1 = fake_detection[
             np.random.choice(fake_detection.shape[0], size_fake_slice, replace=False), :]
    slice2 = fake_track[
             np.random.choice(fake_track.shape[0], size_fake_slice, replace=False), :]
    fake_inp = np.concatenate((slice1, slice2), axis=0)
    fake_lab = np.ones(2 * size_fake_slice)

    # tp indexes : whole size
    tp_idx = torch.randint(low=0, high=len(tp), size=[len(tp)])
    # fp indexes : 75% of tp size
    fp_idx = tp_idx[0:int(len(tp_idx) * .75)]

    # nn input
    inp = torch.cat([fp[fp_idx, :], tp[tp_idx, :], torch.tensor(fake_inp, dtype=torch.float32)])
    # nn target
    target = np.concatenate(
        [np.zeros(len(fp[fp_idx, :]), np.float32), np.ones(len(tp[tp_idx, :]) + fake_lab.shape[0], np.float32)])
    target = torch.from_numpy(target).view(target.shape[0], -1)

    optimizer.zero_grad()
    output = model(inp)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(e)
    print(loss.item())

    # saving the model
    # need to change the e or save more at one run because the best AMOTA is usually for e between 1100 and 1300
    if e == 1050:  # precisley in our case loss == 0.093
        FILE = "{}.pth".format(
            "hl{}_{}_hu{}_e{}_{}i".format(
                len(hid_dim),
                activation_function,
                hid_dim[0],
                e,
                inp_dim))
        torch.save(model, FILE)
