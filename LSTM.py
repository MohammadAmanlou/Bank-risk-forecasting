import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load data from csv
data = pd.read_csv("Risk_Dataset_Haghighi.csv")
data = data.fillna(0)
# Feature selection
features = data.drop(["LABLE_JARI"], axis=1).values
target = data["LABLE_JARI"].values

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Convert data to PyTorch tensors
X = torch.tensor(features_scaled, dtype=torch.float32)
y = torch.tensor(target, dtype=torch.float32)

# Create sequences of data for LSTM model
def create_sequences(features, target, n_steps):
    X_seq, y_seq = [], []
    for i in range(len(features) - n_steps):
        X_seq.append(features[i:i + n_steps])
        y_seq.append(target[i + n_steps])
    return torch.stack(X_seq), torch.stack(y_seq)

n_steps = 1
X_seq, y_seq = create_sequences(X, y, n_steps)

# Split data into training and testing sets
train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# Build LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[-1])
        return output

input_size = X.shape[1]
hidden_size = 50
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
# Train model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate model
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print('Test Loss:', test_loss.item())

# Make predictions
with torch.no_grad():
    predictions = model(X_test)

# Inverse transform predictions
predictions = scaler.inverse_transform(predictions.numpy())

# Compare predictions with actual values
print('Predictions:', predictions)
print('Actual:', y_test.numpy())