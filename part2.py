#Code Created by: Kyle Ketterer
#Date: 04/07/2025

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


import torch                                        # root package
from torch.utils.data import Dataset, DataLoader    # dataset representation and loading
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import time #time the training of the models


#Preprocessing Data Section =====================================================
#import datasets
train_ks = pd.read_csv('train_kdd_small.csv')
test_ks = pd.read_csv('test_kdd_small.csv')

#create encoders
protocol_encoder = LabelEncoder()
service_encoder = LabelEncoder()
flag_encoder = LabelEncoder()

#fit and transform training data
train_ks['protocol_type'] = protocol_encoder.fit_transform(train_ks['protocol_type'])
train_ks['service'] = service_encoder.fit_transform(train_ks['service'])
train_ks['flag'] = flag_encoder.fit_transform(train_ks['flag'])

#transorm test data
test_ks['protocol_type'] = protocol_encoder.transform(test_ks['protocol_type'])
test_ks['service'] = service_encoder.transform(test_ks['service'])
test_ks['flag'] = flag_encoder.transform(test_ks['flag'])


def label_to_int(x):
    if x == 'normal':
        return 0
    else:
        return 1

#identify which rows are normal and which are attacks
train_ks['label'] = train_ks['label'].apply(label_to_int)
test_ks['label'] = test_ks['label'].apply(label_to_int)

#get all features
X_train = train_ks.drop(columns=['label'])
X_test = test_ks.drop(columns=['label'])

#get the label
y_train = train_ks['label']
y_test = test_ks['label']


#Creating and Testing Model Section =====================================================

