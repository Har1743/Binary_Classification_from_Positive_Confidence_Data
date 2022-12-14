import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn import svm
from sklearn.model_selection import train_test_split

df = pd.read_csv('Phishing_Legitimate_full.csv')

float_cols = df.select_dtypes('float64').columns
for c in float_cols:
    df[c] = df[c].astype('float32')
    
int_cols = df.select_dtypes('int64').columns
for c in int_cols:
    df[c] = df[c].astype('int32')
    
# dropping as they are of no use
df = df.drop(['HttpsInHostname'], axis=1)

# let's do binning for url_lenght
df['UrlLength_band'] = pd.cut(df['UrlLength'], 9)
df[['UrlLength_band', 'CLASS_LABEL']].groupby(['UrlLength_band'], as_index=False).mean().sort_values(by='UrlLength_band', ascending=True)

df_1 = df.copy()

for dataset in df_1:    
    df_1.loc[ df['UrlLength'] <= 38.778, 'UrlLength'] = 0
    df_1.loc[(df['UrlLength'] > 38.778) & (df['UrlLength'] <= 65.556), 'UrlLength'] = 1
    df_1.loc[(df['UrlLength'] > 65.556) & (df['UrlLength'] <= 92.333), 'UrlLength'] = 2
    df_1.loc[(df['UrlLength'] > 92.333) & (df['UrlLength'] <= 119.111), 'UrlLength'] = 3
    df_1.loc[(df['UrlLength'] > 92.333) & (df['UrlLength'] <= 119.111), 'UrlLength'] = 4
    df_1.loc[(df['UrlLength'] > 119.111) & (df['UrlLength'] <= 145.889), 'UrlLength'] = 5
    df_1.loc[(df['UrlLength'] > 145.889) & (df['UrlLength'] <= 172.667), 'UrlLength'] = 6
    df_1.loc[(df['UrlLength'] > 172.667) & (df['UrlLength'] <= 199.444), 'UrlLength'] = 7
    df_1.loc[(df['UrlLength'] > 199.444) & (df['UrlLength'] <= 226.222), 'UrlLength'] = 8
    df_1.loc[ df['UrlLength'] > 226.222, 'UrlLength'] = 9
df_1.head(10)

not_helpful_columns = ['id', 'EmbeddedBrandName', 'DoubleSlashInPath', 'RightClickDisabled', 
                       'UrlLengthRT', 'ExtFavicon', 'UrlLength_band']
df_1 = df_1.drop(not_helpful_columns, axis=1)
df_1['CLASS_LABEL'].replace(0, -1, inplace=True)

for i in df_1:
    df_1[i] = df_1[i].astype('float32')
    
accepted_columns = ['NumDots', 'SubdomainLevel', 'PathLevel', 'NumDash', 'NumPercent', 'NumDashInHostname', 
                    'NumUnderscore', 'NumQueryComponents', 'DomainInSubdomains', 'CLASS_LABEL']
#                     'UrlLength', 'AbnormalFormAction', 
#                     'DomainInPaths', 'NoHttps', 'RandomString', 'NumNumericChars', 'HostnameLength', 'NumSensitiveWords', 
#                     'PctExtHyperlinks', 'InsecureForms', 'RelativeFormAction', 'PctNullSelfRedirectHyperlinks', 
#                     'FakeLinkInStatusBar', 'PopUpWindow', 'ExtFormAction', 'SubdomainLevelRT', 'PctExtResourceUrlsRT', 
#                     'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT', 'CLASS_LABEL']

df_1=df_1[accepted_columns]

X_train, X_test, Y_train, Y_test = train_test_split(df_1, df_1['CLASS_LABEL'], test_size=0.2, random_state=0)

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

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

n_estimators = [50, 100, 150, 200, 250, 300, 350]

XRF_train = X_train.drop("CLASS_LABEL", axis=1)
YRF_train = X_train["CLASS_LABEL"]

for val in n_estimators:
    score = cross_val_score(ensemble.RandomForestClassifier(n_estimators= val, random_state= 42), 
                            XRF_train, YRF_train, cv= kf, scoring="accuracy")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
    
best_accuracy_lookup = []

for i in range(0, 5):
    param, accuracy = Group7_PconfClassification(500, .001, 0.01, 'CLASS_LABEL', X_train, X_test, Y_train, Y_test)
    best_accuracy_lookup.append(accuracy)
    
for i in range(0, 5):
    print("Accuracy ", i+1, "is ", best_accuracy_lookup[i])
print("\nBest Accuracy for Pconf on Phishing Web is ", max(best_accuracy_lookup))

best_accuracy_lookup = []

for i in range(0, 10):
    param, accuracy = Group7_PconfClassification(500, .001, 0.01, 'CLASS_LABEL', X_train, X_test, Y_train, Y_test)
    best_accuracy_lookup.append(accuracy)
    
for i in range(0, 10):
    print("Accuracy ", i+1, "is ", best_accuracy_lookup[i])
print("\nBest Accuracy for Pconf on Phishing Web is ", max(best_accuracy_lookup))
