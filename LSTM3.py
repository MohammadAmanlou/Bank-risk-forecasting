#%%
from utils import *
from dataset import *
from preprocess import *
from data import *
from Models import *
from Evaluation import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

dataset = Dataset("Dataset_haghighi.xlsx" , ["CUSTOMER_TP"])
data = dataset.getDividedDatasets()[0]
prp = Preprocess(data , "LABLE_JARI" )
prp.classificationPreprocess(0.2 , "SUMAVGAMNT_JARI")
config = Config()
df = pd.concat([prp.X_train , prp.y_train] , axis = 1)
df.rename(columns = {0:prp.label} , inplace=True)
df.drop(columns = ['CUSTOMER_TP'] , inplace = True )
#%%
category_counts = df['COD_MELI'].value_counts()
categories_to_drop = category_counts[category_counts != 12].index
df = df[~df['COD_MELI'].isin(categories_to_drop)]
#%%
df_pivot = df.melt(id_vars=['YM', 'COD_MELI'], var_name='Feature', value_name='Value')

df_pivot.dropna(subset=["COD_MELI"], inplace=True)


#%%
X = np.array(df_pivot[df_pivot['Feature'] != prp.label].pivot_table(index=['YM', 'COD_MELI'], columns='Feature', values='Value').reset_index().drop(['YM', 'COD_MELI'], axis=1))
y = np.array(df_pivot[df_pivot['Feature'] == prp.label].pivot_table(index=['YM', 'COD_MELI'], values='Value').reset_index().drop(['YM', 'COD_MELI'], axis=1)).flatten()
#%%
X = X.reshape(12, len(df["COD_MELI"].unique()), X.shape[1])
y = y.reshape(12, len(df["COD_MELI"].unique()))

#%%
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

if X_tensor.shape[0] != y_tensor.shape[0]:
    raise ValueError('Number of samples in X and y are different')
#%%

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

input_size = X.shape[2]
hidden_size = 640
num_layers = 3
output_size = 10794

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()

with torch.no_grad():
    outputs = model(X_tensor)
    predicted_labels = (torch.sigmoid(outputs) > 0.5).float()

print(predicted_labels)
# %%
