import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

combinedData= [train_df, test_df]

# changing homeplanet data: 0: Europa, 1: Earth, 2: Mars, 3: nan
for data in combinedData:
  data.loc[data['HomePlanet'].isna(), 'HomePlanet']=data['HomePlanet'].mode()[0]
  data.loc[data['HomePlanet']=='Europa', 'HomePlanet']=0
  data.loc[data['HomePlanet']=='Earth', 'HomePlanet']=1
  data.loc[data['HomePlanet']=='Mars', 'HomePlanet']=2
  data['HomePlanet']=data['HomePlanet'].astype('float32')

# changing CryoSleep data: 0: false, 1: false, 2: nan
  data.loc[data['CryoSleep'].isna(), 'CryoSleep']=data['CryoSleep'].mode()[0]
  data.loc[data['CryoSleep']==0, 'CryoSleep']=0
  data.loc[data['CryoSleep']==1, 'CryoSleep']=1
  data['CryoSleep']=data['CryoSleep'].astype('float32')

# changing VIP data: 0: false, 1: false, 2: nan
  data.loc[data['VIP'].isna(), 'VIP']=data['VIP'].mode()[0]
  data.loc[data['VIP']==0, 'VIP']=0
  data.loc[data['VIP']==1, 'VIP']=1
  data['VIP']=data['VIP'].astype('float32')

# changing Age data: 0: 0-20, 1: 20-30, 3: 30-60, 4: >60
  data.loc[data['VIP']==0, 'VIP']=0
  data.loc[data['VIP']==1, 'VIP']=1
  data.loc[data['VIP'].isna(), 'VIP']=2
  data['VIP']=data['VIP'].astype('float32')

# zero missing data in RoomService, FoodCourt, ShoppingMall, Spa and VRDeck. 
  data.loc[data['RoomService'].isna(), 'RoomService']=0
  data.loc[data['FoodCourt'].isna(), 'FoodCourt']=0
  data.loc[data['ShoppingMall'].isna(), 'ShoppingMall']=0
  data.loc[data['Spa'].isna(), 'Spa']=0
  data.loc[data['VRDeck'].isna(), 'VRDeck']=0

# Creating an aggregate field called Service1 that addes: RoomService, FoodCourt, ShoppingMall
  data['Service1']= data['RoomService'] + data['Spa'] + data['VRDeck']
  data['Service1']= data['Service1'].astype('float32')

# Creating an aggregate field called Service12 that adds: Spa, VRDeck
  data['Service2']= data['FoodCourt'] + data['ShoppingMall']
  data['Service2']= data['Service2'].astype('float32')

# Filling missing Age data with the majority which is 17
  data.loc[data['Age'].isna(), 'Age']=28
# make Age an ordinal field
  data['AgeGroup']= pd.cut(data['Age'],10, labels=[0,1,2,3,4,5,6,7,8,9])
  data['AgeGroup']=data['AgeGroup'].astype('float32')

# Removing nan entries from Destination ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 'Unknown']
  data.loc[data['Destination'].isna(), 'Destination']=data['Destination'].mode()[0]
  data.loc[data['Destination']=='TRAPPIST-1e', 'Destination']=0
  data.loc[data['Destination']=='PSO J318.5-22', 'Destination']=1
  data.loc[data['Destination']=='55 Cancri e', 'Destination']=2
  data['Destination']= data['Destination'].astype('float32')

# Creating Group field from PassengerId
  data['GroupId']= data['PassengerId'].str.extract('([0-9][0-9][0-9][0-9])', expand=False)
  data['GroupId']= data['GroupId'].astype('float32')
# Creating Deck and side fields from Cabin
  data['Deck']= data['Cabin'].str.extract('([A-Za-z])+', expand=False)
  data['Side']= data['Cabin'].str[-1]
  
  data.loc[data['Deck'].isna(), 'Deck']=data['Deck'].mode()[0]
  data.loc[data['Deck']=='B', 'Deck']=0
  data.loc[data['Deck']=='F', 'Deck']=1
  data.loc[data['Deck']=='A', 'Deck']=2
  data.loc[data['Deck']=='G', 'Deck']=3
  data.loc[data['Deck']=='E', 'Deck']=4
  data.loc[data['Deck']=='D', 'Deck']=5
  data.loc[data['Deck']=='C', 'Deck']=6
  data.loc[data['Deck']=='T', 'Deck']=7
  data['Deck']= data['Deck'].astype('float32')

  data.loc[data['Side'].isna(), 'Side']=data['Side'].mode()[0]
  data.loc[data['Side']=='P', 'Side']=0
  data.loc[data['Side']=='S', 'Side']=1
  data['Side']= data['Side'].astype('float32')

# Deleting original columns from TRAIN
train_df= train_df.drop(['Spa'], axis=1)
train_df= train_df.drop(['VRDeck'], axis=1)
train_df= train_df.drop(['RoomService'], axis=1)
train_df= train_df.drop(['FoodCourt'], axis=1)
train_df= train_df.drop(['ShoppingMall'], axis=1)
train_df= train_df.drop(['Age'], axis=1)
train_df= train_df.drop(['Name'], axis=1)
train_df= train_df.drop(['PassengerId'], axis=1)
train_df= train_df.drop(['Cabin'], axis=1)

# Deleting original columns TEST
test_df= test_df.drop(['Spa'], axis=1)
test_df= test_df.drop(['VRDeck'], axis=1)
test_df= test_df.drop(['RoomService'], axis=1)
test_df= test_df.drop(['FoodCourt'], axis=1)
test_df= test_df.drop(['ShoppingMall'], axis=1)
test_df= test_df.drop(['Age'], axis=1)
test_df= test_df.drop(['Name'], axis=1)
#test_df= test_df.drop(['PassengerId'], axis=1)
test_df= test_df.drop(['Cabin'], axis=1)

# changing Trasportation data: 0: false, 1: false on training
train_df.loc[train_df['Transported']==0, 'Transported']=-1
train_df.loc[train_df['Transported']==1, 'Transported']= 1
train_df['Transported']=train_df['Transported'].astype('float32')

# X_train, X_test, Y_train, Y_test = train_test_split(train_df, train_df['Transported'], test_size=0.2, random_state=0)

##################################################################
## Using the authors code with minor changes
##################################################################

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from sklearn import svm

def getPositivePosterior(x, mu1, mu2, cov1, cov2, positive_prior):
    """Returns the positive posterior p(y=+1|x)."""
    conditional_positive = np.exp(-0.5 * (x - mu1).T.dot(np.linalg.inv(cov1)).dot(x - mu1)) / np.sqrt(np.linalg.det(cov1)*(2 * np.pi)**x.shape[0])
    conditional_negative = np.exp(-0.5 * (x - mu2).T.dot(np.linalg.inv(cov2)).dot(x - mu2)) / np.sqrt(np.linalg.det(cov2)*(2 * np.pi)**x.shape[0])
    marginal_dist = positive_prior * conditional_positive + (1 - positive_prior) * conditional_negative
    positivePosterior = conditional_positive * positive_prior / marginal_dist
    return positivePosterior

class LinearNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearNetwork, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

def getAccuracy(x_test, y_test, model):
    """Calculates the classification accuracy."""
    predicted = model(Variable(torch.from_numpy(x_test)))
    accuracy = np.sum(torch.sign(predicted).data.numpy() == np.matrix(y_test).T) * 1. / len(y_test)
    return accuracy

def pconfClassification(inputSize, num_epochs, lr, x_train_p, x_test, y_test, r):
    model = LinearNetwork(input_size=inputSize, output_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_train_p))
        confidence = Variable(torch.from_numpy(r))
        optimizer.zero_grad()
        negative_logistic = nn.LogSigmoid()
        logistic = -1. * negative_logistic(-1. * model(inputs))
        loss = torch.sum(-model(inputs)+logistic * 1. / confidence)  # note that \ell_L(g) - \ell_L(-g) = -g with logistic loss
        loss.backward()
        optimizer.step()
    params = list(model.parameters())
    accuracy = getAccuracy(x_test=x_test, y_test=y_test, model=model)
    return params, accuracy


##################################################################
## Our code to adapt to different datasets
##################################################################

def Group7_PconfClassification(num_epochs, learning_rate, confidence_cutoff, label_ColumnName, X_train, X_test, Y_train, Y_test):
  n_positive= len(X_train[X_train[label_ColumnName]==1])
  n_negative= len(X_train[X_train[label_ColumnName]==-1])
  mu1= X_train[X_train[label_ColumnName]==1].drop(label_ColumnName, axis=1).mean()
  mu2= X_train[X_train[label_ColumnName]==-1].drop(label_ColumnName, axis=1).mean()
  cov1= X_train[X_train[label_ColumnName]==1].drop(label_ColumnName, axis=1).cov()
  cov2= X_train[X_train[label_ColumnName]==-1].drop(label_ColumnName, axis=1).cov()
  x_train_p= X_train[X_train[label_ColumnName]==1].drop(label_ColumnName, axis=1)
  x_train_p= x_train_p
  x_train_p= x_train_p.to_numpy()

  # calculating the exact positive-confidence values: r
  positive_prior = n_positive/(n_positive + n_negative)
  r=[]
  x_train_n=[]
  for i in range(n_positive):
      x = x_train_p[i, :]
      x2 = getPositivePosterior(x, mu1.to_numpy(), mu2.to_numpy(), cov1.to_numpy(), cov2.to_numpy(), positive_prior)
      if x2 > confidence_cutoff:
        x_train_n.append(x_train_p[i])
        r.append(x2)

  x_train_n= np.asarray(x_train_n)
  r= np.asarray(r)
  r = np.matrix(r).T
  x_test= X_test.drop(label_ColumnName, axis=1)
  x_test= x_test.to_numpy()
  y_test= Y_test.astype('float32').to_numpy()
  param, accuracy= pconfClassification(pd.DataFrame(x_train_n).shape[1], num_epochs, learning_rate, x_train_n, x_test, y_test, r)
  return param, accuracy

##################################################################
## Running the experiment
##################################################################

train_df_mod= train_df.copy()
train_df_mod= train_df_mod.drop_duplicates()
train_df_mod= train_df_mod.drop(['GroupId'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(train_df_mod, train_df_mod['Transported'], test_size=0.2, random_state=0)


param, accuracy= Group7_PconfClassification(500, .001, 0.085, 'Transported', X_train, X_test, Y_train, Y_test)

accuracy
