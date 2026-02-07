import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    df_full = pd.read_csv('Yield_curve(2010-2024).csv', index_col='Date', parse_dates=True)
    df_full = df_full.interpolate(method='linear').dropna(axis=1, how='any').sort_index()
    print("Successfully loaded 'Yield_curve(2010-2024).csv'.")
except FileNotFoundError:
    print("Error: 'Yield_curve(2010-2024).csv' not found.")
    exit()

maturities_np = np.array([float(m.split(' ')[0]) / 12 if 'Mo' in m else float(m.split(' ')[0]) for m in df_full.columns])
n_maturities = len(maturities_np)
N_STEPS = 10
SPLINE_SMOOTHING = 2e-3

def B_func_np_g2(t, T, param):
    if abs(param) < 1e-6: return T - t
    return (1 - np.exp(-param * (T - t))) / param

def get_phi_t_g2(t, a, b, sigma, eta, rho, spline):
    f_M_prime_t = spline.derivative(n=1)(t)
    f_M_t = spline(t)
    f_0_t = f_M_prime_t * t + f_M_t
    
    t = max(t, 1e-6)
    
    term1 = (sigma**2 / (2 * a**2)) * (1 - np.exp(-a * t))**2 if a > 1e-6 else 0.0
    term2 = (eta**2 / (2 * b**2)) * (1 - np.exp(-b * t))**2 if b > 1e-6 else 0.0
    term3 = 0.0
    if a > 1e-6 and b > 1e-6:
        term3 = (rho * sigma * eta / (a * b)) * (1 - np.exp(-a * t)) * (1 - np.exp(-b * t))
        
    return f_0_t + term1 + term2 + term3

def get_ln_A_tT_g2(t, T, a, b, sigma, eta, rho, spline):
    P0_T_y = spline(T)
    P0_t_y = spline(t)
    
    P0_T = np.exp(-P0_T_y * T) if np.isfinite(P0_T_y) else np.nan
    P0_t = np.exp(-P0_t_y * t) if np.isfinite(P0_t_y) else np.nan
    
    if np.isnan(P0_T) or np.isnan(P0_t) or P0_t < 1e-9: return np.nan
    
    with np.errstate(divide='ignore'):
        log_term = np.log(P0_T / P0_t)
    if np.isinf(log_term): return np.nan
    
    try:
        B_a = B_func_np_g2(t, T, a)
        B_b = B_func_np_g2(t, T, b)
        v1 = (sigma**2 / (2 * a)) * (B_a**2) if a > 1e-6 else 0.0
        v2 = (eta**2 / (2 * b)) * (B_b**2) if b > 1e-6 else 0.0
        v3 = 0.0
        if abs(a + b) > 1e-6:
            v3 = (rho * sigma * eta / (a + b)) * B_a * B_b
        V_t_T = v1 + v2 + 2 * v3

        B_a_0T = B_func_np_g2(0, T, a)
        B_b_0T = B_func_np_g2(0, T, b)
        v1_0T = (sigma**2 / (2 * a)) * (B_a_0T**2) if a > 1e-6 else 0.0
        v2_0T = (eta**2 / (2 * b)) * (B_b_0T**2) if b > 1e-6 else 0.0
        v3_0T = 0.0
        if abs(a + b) > 1e-6:
            v3_0T = (rho * sigma * eta / (a + b)) * B_a_0T * B_b_0T
        V_0_T = v1_0T + v2_0T + 2 * v3_0T

        B_a_0t = B_func_np_g2(0, t, a)
        B_b_0t = B_func_np_g2(0, t, b)
        v1_0t = (sigma**2 / (2 * a)) * (B_a_0t**2) if a > 1e-6 else 0.0
        v2_0t = (eta**2 / (2 * b)) * (B_b_0t**2) if b > 1e-6 else 0.0
        v3_0t = 0.0
        if abs(a + b) > 1e-6:
            v3_0t = (rho * sigma * eta / (a + b)) * B_a_0t * B_b_0t
        V_0_t = v1_0t + v2_0t + 2 * v3_0t
        
    except (ValueError, OverflowError):
        return np.nan
        
    variance_adjustment = 0.5 * (V_t_T - V_0_T + V_0_t)
    return log_term + variance_adjustment

def g2_plus_plus_call_option_price(K, t, S, T, r_t, params, maturities_np, yields_np):
    a, b, sigma, eta, rho = params
    spline = UnivariateSpline(maturities_np, yields_np, s=SPLINE_SMOOTHING, k=3, ext=1)
    def P_np(tau):
        y = spline(tau)
        return np.exp(-y * tau)
    P_t_S = P_np(S - t)
    P_t_T = P_np(T - t)
    if np.isnan(P_t_S) or np.isnan(P_t_T): return np.nan
    
    tau_opt, tau_bond_opt = S - t, T - S
    try:
        if abs(a) < 1e-6 or abs(b) < 1e-6 or abs(a+b) < 1e-6: return np.nan
        v1 = (sigma**2 / (2*a**3)) * (1 - np.exp(-a*tau_bond_opt))**2 * (1 - np.exp(-2*a*tau_opt))
        v2 = (eta**2 / (2*b**3)) * (1 - np.exp(-b*tau_bond_opt))**2 * (1 - np.exp(-2*b*tau_opt))
        v3 = (2*rho*sigma*eta / (a*b*(a+b))) * (1-np.exp(-a*tau_bond_opt)) * (1-np.exp(-b*tau_bond_opt)) * (1-np.exp(-(a+b)*tau_opt))
        sigma_p_sq = v1 + v2 + v3
    except (ZeroDivisionError, OverflowError): return np.nan

    sigma_p = np.sqrt(np.maximum(0, sigma_p_sq))
    if sigma_p < 1e-9: return max(0.0, P_t_T - K * P_t_S)
    
    with np.errstate(all='ignore'):
        d1 = (np.log(P_t_T / (K * P_t_S)) / sigma_p) + 0.5 * sigma_p
        d2 = d1 - sigma_p
    if not np.isfinite(d1) or not np.isfinite(d2): return np.nan
    
    return P_t_T * norm.cdf(d1) - K * P_t_S * norm.cdf(d2)

def generate_dataset_from_yields(df_yields, samples_per_day, device):
    all_features, all_labels = [], []
    for date in tqdm(df_yields.index, desc="Generating data from yield curves (G2++, s=1e-3)"):
        yields_np = df_yields.loc[date].values / 100.0
        if np.isnan(yields_np).any(): continue
        
        spline = UnivariateSpline(maturities_np, yields_np, s=SPLINE_SMOOTHING, k=3, ext=1)
        r0_current = spline(1e-6)
        
        for _ in range(samples_per_day):
            a, b = np.random.uniform(0.05, 1.0), np.random.uniform(0.01, 0.5)
            sigma, eta = np.random.uniform(0.01, 0.1), np.random.uniform(0.01, 0.1)
            rho = np.random.uniform(-0.9, 0.9)
            params = (a, b, sigma, eta, rho)
            
            S = np.random.uniform(0.5, 5.0)
            T = S + np.random.uniform(1.0, 10.0)
            
            try:
                forward_price = np.exp(-spline(T) * T) / np.exp(-spline(S) * S)
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            
            if not np.isfinite(forward_price) or forward_price < 1e-9:
                continue

            K = forward_price * np.random.uniform(0.9, 1.1)
            option_price = g2_plus_plus_call_option_price(K, 0, S, T, r0_current, params, maturities_np, yields_np)
            
            if np.isfinite(option_price) and 1e-7 < option_price < 1.0:
                tau = T - S
                moneyness = K / forward_price
                
                feature_vector = [S, T, K, tau, moneyness, a, b, sigma, eta, rho] + list(yields_np)
                
                all_features.append(torch.tensor(feature_vector, dtype=torch.float32))
                all_labels.append(torch.tensor([option_price], dtype=torch.float32))

    if not all_features:
        print("Warning: No data generated.")
        return None
    
    return TensorDataset(torch.stack(all_features), torch.stack(all_labels))


class DeepBSDE_Y0_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=6, dropout_rate=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return F.softplus(self.network(x))

class DeepBSDE_Z_Model_G2(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=4, dropout_rate=0.2):
        super().__init__()
        z_input_dim = input_dim + 3
        layers = [nn.Linear(z_input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)
    def forward(self, t, x, y, y_params):
        z_input = torch.cat((t, x, y, y_params), dim=1)
        return self.network(z_input)

SAMPLES_PER_DAY = 20
full_dataset = generate_dataset_from_yields(df_full, SAMPLES_PER_DAY, 'cpu')
if full_dataset is None:
    exit()
    
print(f"\nTotal dataset size: {len(full_dataset)}")

dataset_size = len(full_dataset)
train_val_size = int(0.85 * dataset_size)
test_size = dataset_size - train_val_size
train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=torch.Generator().manual_seed(42))

val_size = int(0.18 * train_val_size)
train_size = train_val_size - val_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)

pure_input_dim = 10 + n_maturities

torch.manual_seed(SEED)
model_Y0 = DeepBSDE_Y0_Model(input_dim=pure_input_dim).to(device)
torch.manual_seed(SEED)
model_Z = DeepBSDE_Z_Model_G2(input_dim=pure_input_dim).to(device)

all_params = list(model_Y0.parameters()) + list(model_Z.parameters())
optimizer = torch.optim.Adam(all_params, lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7)
loss_fn = nn.MSELoss()

num_epochs = 1000
best_val_loss = float('inf')
best_model_weights = None
patience = 100
patience_counter = 0

# 인덱스 정의
IDX_S, IDX_T, IDX_K = 0, 1, 2
IDX_A, IDX_B, IDX_SIGMA, IDX_ETA, IDX_RHO = 5, 6, 7, 8, 9
IDX_YIELDS_START = 10


for epoch in tqdm(range(num_epochs), desc="Training DeepBSDE(G2++)"):
    
    model_Y0.train()
    model_Z.train()
    
    for features_batch_cpu, labels_batch_cpu in train_loader:
        features_batch = features_batch_cpu.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        batch_size = features_batch.shape[0]
        
        S_batch = features_batch[:, IDX_S].unsqueeze(1)
        K_batch = features_batch[:, IDX_K].unsqueeze(1)
        a_batch = features_batch[:, IDX_A].unsqueeze(1)
        b_batch = features_batch[:, IDX_B].unsqueeze(1)
        sigma_batch = features_batch[:, IDX_SIGMA].unsqueeze(1)
        eta_batch = features_batch[:, IDX_ETA].unsqueeze(1)
        rho_batch = features_batch[:, IDX_RHO].unsqueeze(1)
        
        yields_batch_cpu = features_batch[:, IDX_YIELDS_START:].cpu().numpy()
        
        dt_batch = S_batch / N_STEPS
        sqrt_dt_batch = torch.sqrt(dt_batch)
        
        N1_batch = torch.normal(mean=0.0, std=1.0, size=(batch_size, N_STEPS), device=device)
        N2_uncorr_batch = torch.normal(mean=0.0, std=1.0, size=(batch_size, N_STEPS), device=device)
        rho_sqrt_term = torch.sqrt(F.relu(1.0 - rho_batch**2))
        N2_batch = rho_batch * N1_batch + rho_sqrt_term * N2_uncorr_batch
        dW1_batch = N1_batch * sqrt_dt_batch
        dW2_batch = N2_batch * sqrt_dt_batch
        
        pure_features_batch = features_batch
        
        Y_i = model_Y0(pure_features_batch)
        
        # on-the-fly로 r0, ln_A, B_a, B_b, phi 계산
        S_cpu = S_batch.cpu().numpy().flatten()
        T_cpu = features_batch[:, 1].cpu().numpy().flatten()
        a_cpu = a_batch.cpu().numpy().flatten()
        b_cpu = b_batch.cpu().numpy().flatten()
        sigma_cpu = sigma_batch.cpu().numpy().flatten()
        eta_cpu = eta_batch.cpu().numpy().flatten()
        rho_cpu = rho_batch.cpu().numpy().flatten()

        r0_list = []
        ln_A_ST_list = []
        B_a_ST_list = []
        B_b_ST_list = []
        phi_paths_list = [] 
        
        for i in range(batch_size):
            spline = UnivariateSpline(maturities_np, yields_batch_cpu[i], s=SPLINE_SMOOTHING, k=3, ext=1)
            r0_list.append(float(spline(1e-6)))
            
            ln_A_ST_list.append(float(get_ln_A_tT_g2(S_cpu[i], T_cpu[i], a_cpu[i], b_cpu[i], sigma_cpu[i], eta_cpu[i], rho_cpu[i], spline)))
            B_a_ST_list.append(float(B_func_np_g2(S_cpu[i], T_cpu[i], a_cpu[i])))
            B_b_ST_list.append(float(B_func_np_g2(S_cpu[i], T_cpu[i], b_cpu[i])))
            
            dt_i = S_cpu[i] / N_STEPS
            phi_path = []
            for k in range(N_STEPS):
                t_k = k * dt_i
                phi_path.append(float(get_phi_t_g2(t_k, a_cpu[i], b_cpu[i], sigma_cpu[i], eta_cpu[i], rho_cpu[i], spline)))
            phi_paths_list.append(phi_path)
            
        r_i = torch.tensor(r0_list, device=device, dtype=torch.float32).unsqueeze(1)
        ln_A_ST_batch = torch.tensor(ln_A_ST_list, device=device, dtype=torch.float32).unsqueeze(1)
        B_a_ST_batch = torch.tensor(B_a_ST_list, device=device, dtype=torch.float32).unsqueeze(1)
        B_b_ST_batch = torch.tensor(B_b_ST_list, device=device, dtype=torch.float32).unsqueeze(1)
        phi_paths_batch = torch.tensor(np.array(phi_paths_list), device=device, dtype=torch.float32)
        
        x_i = torch.zeros_like(Y_i)
        y_i = torch.zeros_like(Y_i)

        for j in range(N_STEPS):
            t_j = j * dt_batch
            
            Z_i = model_Z(t_j, x_i, y_i, pure_features_batch)
            Z1_i, Z2_i = Z_i[:, 0].unsqueeze(1), Z_i[:, 1].unsqueeze(1)
            
            dW1_j, dW2_j = dW1_batch[:, j].unsqueeze(1), dW2_batch[:, j].unsqueeze(1)
            N1_j, N2_j = N1_batch[:, j].unsqueeze(1), N2_batch[:, j].unsqueeze(1)

            phi_j = phi_paths_batch[:, j].unsqueeze(1)
            r_i = x_i + y_i + phi_j 

            Y_i = Y_i + (r_i * Y_i) * dt_batch + Z1_i * dW1_j + Z2_i * dW2_j
            
            exp_a_dt, exp_b_dt = torch.exp(-a_batch * dt_batch), torch.exp(-b_batch * dt_batch)
            exp_2a_dt, exp_2b_dt = torch.exp(-2 * a_batch * dt_batch), torch.exp(-2 * b_batch * dt_batch)

            var_x = torch.where(a_batch.abs() < 1e-6, sigma_batch**2 * dt_batch, sigma_batch**2 * (1.0 - exp_2a_dt) / (2.0 * a_batch))
            std_dev_x = torch.sqrt(F.relu(var_x))
            x_i = x_i * exp_a_dt + std_dev_x * N1_j
            
            var_y = torch.where(b_batch.abs() < 1e-6, eta_batch**2 * dt_batch, eta_batch**2 * (1.0 - exp_2b_dt) / (2.0 * b_batch))
            std_dev_y = torch.sqrt(F.relu(var_y))
            y_i = y_i * exp_b_dt + std_dev_y * N2_j
            
        Y_S_pred = Y_i
        x_S, y_S = x_i, y_i

        P_S_T = torch.exp(ln_A_ST_batch) * torch.exp(-B_a_ST_batch * x_S - B_b_ST_batch * y_S)
        Y_S_target = F.relu(P_S_T - K_batch)
        
        valid_mask = torch.isfinite(Y_S_pred) & torch.isfinite(Y_S_target)
        
        num_valid = valid_mask.sum()
        
        if num_valid > 0:
            loss_total = loss_fn(Y_S_pred[valid_mask], Y_S_target[valid_mask])
            
            if not torch.isnan(loss_total):
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model_Y0.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model_Z.parameters(), 1.0)
                optimizer.step()
        
    model_Y0.eval()
    current_val_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features_batch_cpu, labels_batch_cpu in val_loader:
            features_batch = features_batch_cpu.to(device, non_blocking=True)
            labels_batch = labels_batch_cpu.to(device, non_blocking=True)
            
            pure_features_batch = features_batch[:, :pure_input_dim]
            preds_val = model_Y0(pure_features_batch)
            
            loss_val = loss_fn(preds_val, labels_batch)
            
            if not torch.isnan(loss_val):
                current_val_loss += loss_val.item() * features_batch.size(0)
                total_samples += features_batch.size(0)
    
    if total_samples > 0:
        current_val_loss /= total_samples
    else:
        current_val_loss = 0.0
    
    scheduler.step()

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_weights = copy.deepcopy(model_Y0.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

print(f"Training finished. Best validation MSE: {best_val_loss:.8f}")

model_Y0.load_state_dict(best_model_weights)
model_Y0.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for features_batch_cpu, labels_batch_cpu in test_loader:
        features_batch = features_batch_cpu.to(device, non_blocking=True)
        labels_batch = labels_batch_cpu.to(device, non_blocking=True)

        pure_features_batch = features_batch[:, :pure_input_dim]
        preds = model_Y0(pure_features_batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels_batch.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

valid_indices = np.isfinite(all_preds.flatten()) & np.isfinite(all_labels.flatten())
all_preds = all_preds[valid_indices]
all_labels = all_labels[valid_indices]

r2 = r2_score(all_labels, all_preds)
final_mse = mean_squared_error(all_labels, all_preds)

print("\n" + "="*50)
print(f"FINAL DeepBSDE RESULTS (G2++)")
print("="*50)
print(f"Test Set MSE: {final_mse:.8f}")
print(f"Test Set R2 Score: {r2:.4f}")
print("="*50)