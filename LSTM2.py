import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset , DataLoader
from utils import *
from dataset import *
from preprocess import *
from data import *
from Models import *
from Evaluation import *
# "Risk_Dataset_Haghighi.csv"

class CustomDataset(Dataset):
    def __init__(self, data_path):
        dataset = Dataset( data_path , ["CUSTOMER_TP"])
        data = dataset.getDividedDatasets()[0]
        data.data.sort_values(['YM'], inplace=True)
        prp = Preprocess(data , "LABLE_JARI" )
        prp.preprocess(0.2 , "SUMAVGAMNT_JARI")
        self.df = pd.concat([prp.X_train,prp.y_train] , axis = 0)
        features = self.df.iloc[:, :-1].values
        labels = self.df.iloc[:, -1].values

        # Construct a list of dictionaries where each dictionary represents a data sample
        self.data = []
        for i in range(len(self.df)):
            self.data.append({'features': features[i], 'label': labels[i]})
        self.label = prp.label
        self.features_names = prp.features

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample["features"], sample["label"]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size

        h0 = torch.zeros( batch_size, self.hidden_size).to(device) 
        c0 = torch.zeros( batch_size, self.hidden_size).to(device) 

        out, _  = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out


input_size = 27
hidden_size = 127
num_layers = 3
output_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = CustomDataset("Risk_Dataset_Haghighi.csv")
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

