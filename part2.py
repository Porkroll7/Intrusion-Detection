#Code Created by: Kyle Ketterer
#Date: 04/07/2025

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import torch                                                # root package
from torch.utils.data import TensorDataset, DataLoader      # dataset representation and loading
import torch.nn as nn                                       # neural networks
import torch.nn.functional as F                             # layers, activations and more
import torch.optim as optim                                 # optimizers e.g. gradient descent, ADAM, etc.

import time #time the training of the models

import matplotlib.pyplot as plt



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

#turn label strings into integers, 0 = normal, 1 = attack
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


#scale features so that features evenly contribute to the model
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



#Creating and Testing Model Section =====================================================


#numPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#turn tensors into TDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#create DataLoaders
train_DataLoader = DataLoader(train_dataset, batch_size=64, shuffle=True) #shuffle so that the model doesn't learn the order of the data
test_DataLoader = DataLoader(test_dataset, batch_size=64)


class IntrusionNN(nn.Module):                   #https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
    def __init__(self, input_size):             #constructor
        super(IntrusionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)     #maps the input features to 8 values
        self.fc2 = nn.Linear(8, 4)              #maps the 8 values from prev layer to 4 values
        self.output = nn.Linear(4, 2)           #maps the 4 values from prev layer to the two output cases (normal, not normal)

    def forward(self, x):                       #defines steps the model should take when receiving data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
    
    
#Create the object we defined above
model = IntrusionNN(input_size=X_train.shape[1])    #input size is the number of features
optimizer = optim.SGD(model.parameters(), lr=0.01)  #SGD optimizer https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
lossFunction = nn.CrossEntropyLoss()                #loss function for classification https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

start = time.time()

epochs = 25     #number of times to train the model
model.train()   #set model to training mode
losses = []

#Training model========================================================
for epoch in range(epochs):
    total_loss = 0
    for features, labels in train_DataLoader:       #get a batch of data
        optimizer.zero_grad()                   #clear the gradients from the previous step
        outputs = model(features)               #model's predictions
        loss = lossFunction(outputs, labels)   
        loss.backward()                         #backpropagation
        optimizer.step()                        #update weights of model
        total_loss += loss.item()               #update loss
    losses.append(total_loss)                  #store loss for plotting
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

end = time.time()
print(f"Training time: {end - start:.5f} seconds")


#Evaluating model======================================================

model.eval()  #set model to evaluation mode

predictionList = [] 
labelList = []

for features, labels in test_DataLoader:            #batch over test data
    outputs = model(features)                   #get model predictions
    predictions = outputs.argmax(dim=1)         #get class index (highest score)
    predictionList.extend(predictions.tolist()) #store prediction
    labelList.extend(labels.tolist())           #store label

#classification report
print("=== Neural Network Classification Report ===")
print(classification_report(labelList, predictionList)) #print classification report


f = open("part2.txt", "w")
f.write("=== Loss Per Epoch ===\n")
for i, loss in enumerate(losses):
    f.write(f"Epoch {i+1}: Loss = {loss:.4f}\n")

f.write("\n")

# Write training time
f.write(f"Training time: {end - start:.5f} seconds\n\n")

# Classification Report
f.write("Neural Network Classification Report\n")
f.write(classification_report(labelList, predictionList))
f.write("\n")



import matplotlib.pyplot as plt
#plotting loss over epochs
plt.figure()
plt.plot(range(1, len(losses) + 1), losses, marker="D", color='red')
plt.title("Loss Over Epochs During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("plot_loss.png")
plt.show()
