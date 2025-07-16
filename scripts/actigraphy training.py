# import libraries
import pandas as pd
import numpy as np
import polars as pl
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from matplotlib.ticker import MaxNLocator

# import series and train datasets
sample = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv")
train = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/train.csv")

# define function to import actigraphy for a fiven id and generate plots
def analyze_actigraphy(id, only_one_week=False, small=False):
    actigraphy = pl.read_parquet(f'/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/id={id}/part-0.parquet')
    day = actigraphy.get_column('relative_date_PCIAT') + actigraphy.get_column('time_of_day') / 86400e9
    sample = train[train['id'] == id]
    age = sample['Basic_Demos-Age'].item()
    sex = ['boy', 'girl'][sample['Basic_Demos-Sex'].item()]
    actigraphy = (actigraphy.with_columns((day.diff() * 86400).alias('diff_seconds'), (np.sqrt(np.square(pl.col('X')) + np.square(pl.col('Y')) + np.square(pl.col('Z'))).alias('norm'))))
    if only_one_week:
        start = np.ceil(day.min())
        mask = (start <= day.to_numpy()) & (day.to_numpy() <= start + 7*3)
        mask &= ~ actigraphy.get_column('non-wear_flag').cast(bool).to_numpy()
    else:
        mask = np.full(len(day), True)  
    if small:
        timelines = [('enmo', 'forestgreen'), ('light', 'orange')]
    else:
        timelines = [('X', 'm'), ('Y', 'm'), ('Z', 'm'), ('enmo', 'forestgreen'), ('anglez', 'lightblue'), ('light', 'orange'), ('non-wear_flag', 'chocolate')]
    _, axs = plt.subplots(len(timelines), 1, sharex=True, figsize=(12, len(timelines) * 1.1 + 0.5))
    for ax, (feature, color) in zip(axs, timelines):
        ax.set_facecolor('white')
        ax.scatter(day.to_numpy()[mask], actigraphy.get_column(feature).to_numpy()[mask], color=color, label=feature, s=1)
        ax.legend(loc='upper left', facecolor='white')
        if feature == 'diff_seconds':
            ax.set_ylim(-0.5, 20.5)
    axs[-1].set_xlabel('day')
    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    axs[0].set_title(f'id={id}, {sex}, age={age}')
    plt.show()

# pull actigraphy data for participant 029a19c9
analyze_actigraphy('029a19c9', only_one_week=False)


def extract_features_from_parquet(parquet_path):
    df = pl.read_parquet(parquet_path)
    
    # Only keep wear-time data
    df = df.filter(df['non-wear_flag'] == 0)
    
    feats = {}
    signals = ['X', 'Y', 'Z', 'enmo', 'anglez', 'light']
    
    for col in signals:
        col_vals = df[col]
        feats[f'{col}_mean'] = col_vals.mean()
        feats[f'{col}_std'] = col_vals.std()
        feats[f'{col}_min'] = col_vals.min()
        feats[f'{col}_max'] = col_vals.max()
        feats[f'{col}_median'] = col_vals.median()
        feats[f'{col}_skewness'] = col_vals.skew()
        feats[f'{col}_kurtosis'] = col_vals.kurtosis()
        feats[f'{col}_percentile_25'] = col_vals.quantile(0.25)
        feats[f'{col}_percentile_75'] = col_vals.quantile(0.75)
        col_vals_shifted = col_vals[1:]  # Remove first element
        col_vals_original = col_vals[:-1]  # Remove last element
        zero_crossings = ((col_vals_original * col_vals_shifted) < 0).sum()
        feats[f'{col}_zero_crossings'] = zero_crossings
    
    feats['non_wear_ratio'] = df['non-wear_flag'].mean()
    
    # Extract participant ID from path
    feats['id'] = parquet_path.split("id=")[-1].split("/")[0]
    
    return feats

paths = glob.glob('/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet/id=*/part-0.parquet')


features = []
with ThreadPoolExecutor(max_workers=8) as executor:
    for feat in tqdm(executor.map(extract_features_from_parquet, paths), total=len(paths)):
        features.append(feat)

features = pd.DataFrame(features)


data = features.merge(train[['id', 'sii']], on='id')
data.drop(columns=['id'], inplace=True)  # Drop ID before training


class SiiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SiiMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    
    def forward(self, x):
        return self.net(x)


X = data.drop(columns=['sii']).values
y = data['sii'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

train_dataset = SiiDataset(X_train, y_train)
val_dataset = SiiDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


model = SiiMLP(X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 15

for epoch in range(n_epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            val_preds.append(preds.numpy())
            val_targets.append(yb.numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    rmse = mean_squared_error(val_targets, val_preds, squared=False)
    print(f"Epoch {epoch+1} | RMSE: {rmse:.4f}")