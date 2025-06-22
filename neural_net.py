# neural_net.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_preprocessing import set_seed


class HousePriceNN(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_neural_net(X_train, X_test, y_train, y_test, preprocessor, hidden_layers=[64, 32], seed=42, epochs=200):
    set_seed(seed)

    # Transform features
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    # Normalize target
    from sklearn.preprocessing import StandardScaler
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.to_numpy().reshape(-1, 1))

    # Tensors
    X_train_tensor = torch.tensor(X_train_proc, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    model = HousePriceNN(input_dim=X_train_tensor.shape[1], hidden_layers=hidden_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).numpy()
        y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
        y_test_actual = y_test.to_numpy().reshape(-1, 1)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    results = {
        "Model": f"NeuralNet {hidden_layers}",
        "RMSE": rmse,
        "R²": r2,
        "MAE": mae,
        "MAPE (%)": mape
    }

    return results
