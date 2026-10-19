import copy
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OriginalLSTM(nn.Module):
    def __init__(self, hidden_size=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.lstm(x.unsqueeze(-1))
        return self.output(sequence[:, -1]).squeeze(-1)


class OriginalBiLSTM(nn.Module):
    """PyTorch translation of the prior Keras two-layer BiLSTM/LSTM model."""

    def __init__(self, units=128, dropout=0.2, bidirectional=False):
        super().__init__()
        self.first = nn.LSTM(1, units, batch_first=True, bidirectional=bidirectional)
        first_width = units * (2 if bidirectional else 1)
        self.second = nn.LSTM(first_width, units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Sequential(nn.Linear(units, 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        sequence, _ = self.first(x.unsqueeze(-1))
        sequence = self.dropout(sequence)
        sequence, _ = self.second(sequence)
        last = self.dropout(sequence[:, -1])
        return self.dense(last).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            weight_norm(
                nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(
                nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(self.network(x) + residual)


class OriginalTCN(nn.Module):
    def __init__(self, channels=(32, 32, 32), kernel_size=3, dropout=0.3):
        super().__init__()
        blocks = []
        for index, output_channels in enumerate(channels):
            input_channels = 1 if index == 0 else channels[index - 1]
            blocks.append(
                TemporalBlock(
                    input_channels,
                    output_channels,
                    kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.output = nn.Linear(channels[-1], 1)

    def forward(self, x):
        features = self.network(x.unsqueeze(1))
        return self.output(features[:, :, -1]).squeeze(-1)


class GenericBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta, size, is_forecast):
        width = self.forecast_size if is_forecast else self.backcast_size
        return theta[:, : min(width, theta.shape[1])]


class TrendBasis(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def forward(self, theta, size, is_forecast):
        theta = theta[:, : self.degree + 1]
        t = torch.arange(
            size if not is_forecast else 2 * size,
            dtype=torch.float32,
            device=theta.device,
        )
        if is_forecast:
            t = t[size:]
        t = t.unsqueeze(0).repeat(theta.shape[0], 1) / size
        basis = torch.stack([t**power for power in range(self.degree + 1)], dim=-1)
        return torch.sum(basis * theta.unsqueeze(1), dim=-1)


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics=2):
        super().__init__()
        self.harmonics = harmonics

    def forward(self, theta, size, is_forecast):
        theta = theta[:, : 2 * self.harmonics]
        t = torch.arange(
            size if not is_forecast else 2 * size,
            dtype=torch.float32,
            device=theta.device,
        )
        if is_forecast:
            t = t[size:]
        t = t.unsqueeze(0).repeat(theta.shape[0], 1) / 12.0
        cosine = [torch.cos(2 * np.pi * i * t) for i in range(1, self.harmonics + 1)]
        sine = [torch.sin(2 * np.pi * i * t) for i in range(1, self.harmonics + 1)]
        basis = torch.stack(cosine + sine, dim=-1)
        return torch.sum(basis * theta.unsqueeze(1), dim=-1)


class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis, layers=4, layer_size=64):
        super().__init__()
        self.input_size = input_size
        self.basis = basis
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size if index == 0 else layer_size, layer_size)
                for index in range(layers)
            ]
        )
        self.backcast = nn.Linear(layer_size, theta_size)
        self.forecast = nn.Linear(layer_size, theta_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        backcast = self.basis(self.backcast(x), self.input_size, False)
        forecast = self.basis(self.forecast(x), 1, True)
        return backcast, forecast


class OriginalNBeats(nn.Module):
    def __init__(self, lookback=12):
        super().__init__()
        stack_types = ["generic", "trend", "seasonality"]
        theta_dims = [12, 4, 4]
        bases = {
            "generic": GenericBasis(lookback, 1),
            "trend": TrendBasis(degree=3),
            "seasonality": SeasonalityBasis(harmonics=2),
        }
        self.stacks = nn.ModuleList()
        for stack_type, theta_dim in zip(stack_types, theta_dims):
            self.stacks.append(
                nn.ModuleList(
                    [
                        NBeatsBlock(lookback, theta_dim, bases[stack_type], 4, 64)
                        for _ in range(2)
                    ]
                )
            )

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.shape[0], 1, device=x.device)
        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
        return forecast.squeeze(-1)


class NHiTSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, pooling_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.pooling_size = pooling_size
        pooled_size = input_size // pooling_size if pooling_size > 1 else input_size
        self.pooling = (
            nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)
            if pooling_size > 1
            else None
        )
        layers = []
        for index in range(n_layers):
            layers.extend(
                [
                    nn.Linear(pooled_size if index == 0 else hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_size, pooled_size)
        self.forecast = nn.Linear(hidden_size, 1)

    def forward(self, x):
        pooled = self.pooling(x.unsqueeze(1)).squeeze(1) if self.pooling else x
        hidden = self.mlp(pooled)
        backcast = self.backcast(hidden)
        if self.pooling:
            backcast = torch.nn.functional.interpolate(
                backcast.unsqueeze(1), size=self.input_size, mode="linear", align_corners=True
            ).squeeze(1)
        return backcast, self.forecast(hidden)


class OriginalNHiTS(nn.Module):
    def __init__(self, lookback=12):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(lookback, 512, 2, pooling, 0.1)
                for pooling in [2, 2, 1]
            ]
        )

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.shape[0], 1, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast.squeeze(-1)


MODEL_SETTINGS = {
    "LSTM": {
        "batch_size": 128,
        "epochs_original": 100,
        "epochs_candidate_cap": 8,
        "patience": 3,
        "training_sample_cap": 4000,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "architecture": "2-layer LSTM, 128 hidden units, dropout 0.2",
    },
    "BiLSTM": {
        "batch_size": 64,
        "epochs_original": 300,
        "epochs_candidate_cap": 8,
        "patience": 3,
        "training_sample_cap": 4000,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "architecture": "two recurrent layers, variable-specific 128/150/200 units",
    },
    "TCN": {
        "batch_size": 256,
        "epochs_original": 80,
        "epochs_candidate_cap": 3,
        "patience": 2,
        "training_sample_cap": 4000,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "architecture": "3 causal residual blocks, channels 32/32/32, kernel 3",
    },
    "N-BEATS": {
        "batch_size": 128,
        "epochs_original": 100,
        "epochs_candidate_cap": 8,
        "patience": 3,
        "training_sample_cap": 4000,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "architecture": "generic/trend/seasonality stacks, 2 blocks each, width 64",
    },
    "N-HiTS": {
        "batch_size": 128,
        "epochs_original": 100,
        "epochs_candidate_cap": 8,
        "patience": 3,
        "training_sample_cap": 4000,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "architecture": "3 stacks, pooling 2/2/1, 2 layers, width 512",
    },
}


def make_model(model_name: str, lookback: int, variable: str) -> nn.Module:
    if model_name == "LSTM":
        return OriginalLSTM()
    if model_name == "BiLSTM":
        units = 200 if variable == "SPEI" else 150 if variable == "TWS" else 128
        bidirectional = variable in {"SPEI", "TWS"}
        return OriginalBiLSTM(units=units, bidirectional=bidirectional)
    if model_name == "TCN":
        return OriginalTCN()
    if model_name == "N-BEATS":
        return OriginalNBeats(lookback=lookback)
    if model_name == "N-HiTS":
        return OriginalNHiTS(lookback=lookback)
    raise KeyError(model_name)


@dataclass
class FitResult:
    model: nn.Module
    best_validation_loss: float
    epochs_run: int
    device: str


def fit_torch_model(
    model_name: str,
    variable: str,
    lookback: int,
    train: Dict,
    validation: Dict,
    seed: int,
) -> FitResult:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name, lookback, variable).to(device)
    settings = MODEL_SETTINGS[model_name]
    train_valid = train["mask"] > 0
    validation_valid = validation["mask"] > 0
    if train_valid.sum() == 0 or validation_valid.sum() == 0:
        raise RuntimeError("No valid train or validation labels")
    valid_indices = np.flatnonzero(train_valid)
    sample_cap = settings["training_sample_cap"]
    if len(valid_indices) > sample_cap:
        rng = np.random.default_rng(seed)
        valid_indices = np.sort(rng.choice(valid_indices, size=sample_cap, replace=False))
    dataset = TensorDataset(
        torch.from_numpy(train["X"][valid_indices]),
        torch.from_numpy(train["y"][valid_indices]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_x = torch.from_numpy(validation["X"][validation_valid]).to(device)
    val_y = torch.from_numpy(validation["y"][validation_valid]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(settings["epochs_candidate_cap"]):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(batch_x)
                loss = torch.mean((prediction - batch_y) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        with torch.no_grad():
            prediction = model(val_x)
            validation_loss = float(torch.mean((prediction - val_y) ** 2).item())
        epochs_run = epoch + 1
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return FitResult(model=model, best_validation_loss=best_loss, epochs_run=epochs_run, device=str(device))


def predict_torch(model: nn.Module, samples: Dict, batch_size=4096) -> np.ndarray:
    device = next(model.parameters()).device
    output = np.full(len(samples["y"]), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(output), batch_size):
            end = min(len(output), start + batch_size)
            x = torch.from_numpy(samples["X"][start:end]).to(device)
            output[start:end] = model(x).detach().cpu().numpy().astype(np.float32)
    return output


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    if observed.size < 2:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson_r": np.nan,
            "normalized_rmse": np.nan,
            "bias": np.nan,
            "n": int(observed.size),
        }
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    std = float(np.std(observed))
    pearson = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if std > 0 and np.std(predicted) > 0
        else np.nan
    )
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(observed, predicted)),
        "pearson_r": pearson,
        "normalized_rmse": rmse / std if std > 0 else np.nan,
        "bias": float(np.mean(predicted - observed)),
        "n": int(observed.size),
    }


def metric_rows(
    predictions: pd.DataFrame,
    common: Dict,
) -> Tuple[Dict, Dict]:
    valid = predictions["valid"].astype(bool)
    pixel = calculate_metrics(
        predictions.loc[valid, "observed"].to_numpy(),
        predictions.loc[valid, "predicted"].to_numpy(),
    )
    monthly_frame = (
        predictions.loc[valid]
        .groupby("date", as_index=False)[["observed", "predicted"]]
        .mean()
    )
    monthly = calculate_metrics(
        monthly_frame["observed"].to_numpy(),
        monthly_frame["predicted"].to_numpy(),
    )
    pixel.update(common)
    monthly.update(common)
    monthly["n_months"] = monthly.pop("n")
    return pixel, monthly
