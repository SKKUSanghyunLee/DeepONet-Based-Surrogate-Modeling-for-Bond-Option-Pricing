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
SPLINE_SMOOTHING = 2e-3

def B_func_np_g2(t, T, param):
    if abs(param) < 1e-6: return T - t
    return (1 - np.exp(-param * (T - t))) / param

def get_ln_A_tT_g2(t, T, a, b, sigma, eta, rho, spline):
    S = t
    tau_opt, tau_bond_opt = S - 0.0, T - S
    
    try:
        if abs(a) < 1e-6 or abs(b) < 1e-6 or abs(a+b) < 1e-6:
             return np.nan 
        
        v1 = (sigma**2 / (2*a**3)) * (1 - np.exp(-a*tau_bond_opt))**2 * (1 - np.exp(-2*a*tau_opt))
        v2 = (eta**2 / (2*b**3)) * (1 - np.exp(-b*tau_bond_opt))**2 * (1 - np.exp(-2*b*tau_opt))
        v3 = (2*rho*sigma*eta / (a*b*(a+b))) * (1-np.exp(-a*tau_bond_opt)) * (1-np.exp(-b*tau_bond_opt)) * (1-np.exp(-(a+b)*tau_opt))
        
        V_adj = 0.5 * (v1 + v2 + v3)
        
    except (ZeroDivisionError, OverflowError): 
        return np.nan

    P0_T_y = spline(T)
    P0_t_y = spline(t)
    P0_T = np.exp(-P0_T_y * T) if np.isfinite(P0_T_y) else np.nan
    P0_t = np.exp(-P0_t_y * t) if np.isfinite(P0_t_y) else np.nan
    
    if np.isnan(P0_T) or np.isnan(P0_t) or P0_t < 1e-9: return np.nan
    
    with np.errstate(divide='ignore'):
        log_term = np.log(P0_T / P0_t)
        
    if np.isinf(log_term): return np.nan
    
    return log_term + V_adj

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
        if abs(a) < 1e-6 or abs(b) < 1e-6 or abs(a+b) < 1e-6:
            return np.nan
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
    for date in tqdm(df_yields.index, desc="Generating PINN data from yield curves (G2++)"):
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
                if not np.isfinite(forward_price) or forward_price < 1e-9: continue
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            
            K = forward_price * np.random.uniform(0.9, 1.1)

            option_price = g2_plus_plus_call_option_price(K, 0, S, T, r0_current, params, maturities_np, yields_np)

            if np.isfinite(option_price) and 1e-7 < option_price < 1.0:
                tau = T - S
                moneyness = K / forward_price
                feature_vector = [S, T, K, tau, moneyness, a, b, sigma, eta, rho] + list(yields_np)

                all_features.append(torch.tensor(feature_vector, dtype=torch.float32, device='cpu'))
                all_labels.append(torch.tensor([option_price], dtype=torch.float32, device='cpu'))

    if not all_features: return None
    return TensorDataset(torch.stack(all_features), torch.stack(all_labels))

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_layers=6, dropout_rate=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return F.softplus(self.network(x))

SAMPLES_PER_DAY = 20
full_dataset = generate_dataset_from_yields(df_full, SAMPLES_PER_DAY, device)
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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)

base_input_dim = train_dataset[0][0].shape[0]
model_input_dim = base_input_dim + 3
num_yields = len(maturities_np)
param_feature_count = base_input_dim - num_yields

torch.manual_seed(SEED)
model = PINN(input_dim=model_input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-7)
loss_fn = nn.MSELoss()

num_epochs = 1000
best_val_loss = float('inf')
best_model_weights = None
patience = 100
patience_counter = 0
lambda_terminal = 10.0


for epoch in tqdm(range(num_epochs), desc="Training PINN(G2++)"):
    model.train()
    for features_batch_cpu, labels_batch_cpu in train_loader:
        
        features_batch = features_batch_cpu.to(device, non_blocking=True)
        
        optimizer.zero_grad()

        batch_size = features_batch.size(0)
        S_option = features_batch[:, 0].unsqueeze(1)
        T_option = features_batch[:, 1].unsqueeze(1)
        K_option = features_batch[:, 2].unsqueeze(1)

        t_pde = torch.rand(batch_size, 1, device=device) * S_option
        x_pde = torch.randn(batch_size, 1, device=device) * 0.05
        y_pde = torch.randn(batch_size, 1, device=device) * 0.05
        t_pde.requires_grad = True
        x_pde.requires_grad = True
        y_pde.requires_grad = True

        a, b, sigma, eta, rho = (features_batch[:, i].unsqueeze(1) for i in range(5, 10))
        yields_pde = features_batch[:, param_feature_count:]

        t_pde_cpu = t_pde.detach().cpu().squeeze().numpy()
        yields_pde_cpu = yields_pde.detach().cpu().numpy()
        params_cpu = features_batch[:, 5:10].detach().cpu().numpy()

        phi_list = []
        for i in range(batch_size):
            a_i, b_i, sig_i, eta_i, rho_i = params_cpu[i]
            spline = UnivariateSpline(maturities_np, yields_pde_cpu[i], s=SPLINE_SMOOTHING, k=3, ext=1)
            t_val = t_pde_cpu[i] if np.isscalar(t_pde_cpu) else t_pde_cpu[i].item()
            t_val = max(t_val, 1e-6)

            try:
                f_M_prime_t = spline.derivative(n=1)(t_val)
                f_M_t = spline(t_val)
                f_0_t = f_M_prime_t * t_val + f_M_t
                
                if abs(a_i) < 1e-6 or abs(b_i) < 1e-6:
                     phi_list.append(f_0_t)
                     continue

                term1 = (sig_i**2 / (2*a_i**2)) * (1 - np.exp(-a_i*t_val))**2
                term2 = (eta_i**2 / (2*b_i**2)) * (1 - np.exp(-b_i*t_val))**2
                term3 = (rho_i*sig_i*eta_i / (a_i*b_i)) * (1-np.exp(-a_i*t_val)) * (1-np.exp(-b_i*t_val))
                
                phi_val = f_0_t + term1 + term2 + term3
            except (ValueError, ZeroDivisionError, OverflowError):
                phi_val = 0.0
                
            phi_list.append(phi_val)

        phi_t = torch.tensor(phi_list, device=device, dtype=torch.float32).unsqueeze(1)

        model_input_pde = torch.cat([t_pde, x_pde, y_pde, features_batch], dim=1)
        C = model(model_input_pde)

        C_grads = torch.autograd.grad(C, [t_pde, x_pde, y_pde], grad_outputs=torch.ones_like(C), create_graph=True)
        C_t, C_x, C_y = C_grads[0], C_grads[1], C_grads[2]

        C_xx = torch.autograd.grad(C_x, x_pde, grad_outputs=torch.ones_like(C_x), create_graph=True)[0]
        C_yy = torch.autograd.grad(C_y, y_pde, grad_outputs=torch.ones_like(C_y), create_graph=True)[0]
        C_xy = torch.autograd.grad(C_x, y_pde, grad_outputs=torch.ones_like(C_x), create_graph=True)[0]

        r_t = x_pde + y_pde + phi_t

        residual = (C_t - a*x_pde*C_x - b*y_pde*C_y +
                    0.5*sigma**2*C_xx + 0.5*eta**2*C_yy +
                    rho*sigma*eta*C_xy - r_t*C)

        loss_pde = loss_fn(residual, torch.zeros_like(residual))

        t_terminal = S_option
        x_terminal = torch.randn(batch_size, 1, device=device) * 0.05
        y_terminal = torch.randn(batch_size, 1, device=device) * 0.05

        S_cpu = S_option.cpu().numpy()
        T_cpu = T_option.cpu().numpy()
        K_cpu = K_option.cpu().numpy()
        x_terminal_cpu = x_terminal.cpu().numpy()
        y_terminal_cpu = y_terminal.cpu().numpy()

        payoff_list = []
        for i in range(batch_size):
            spline = UnivariateSpline(maturities_np, yields_pde_cpu[i], s=SPLINE_SMOOTHING, k=3, ext=1)
            
            S_i, T_i, K_i = S_cpu[i][0], T_cpu[i][0], K_cpu[i][0]
            a_i, b_i, sig_i, eta_i, rho_i = params_cpu[i]
            x_S_i, y_S_i = x_terminal_cpu[i][0], y_terminal_cpu[i][0]

            try:
                log_A_S_T = get_ln_A_tT_g2(S_i, T_i, a_i, b_i, sig_i, eta_i, rho_i, spline)
                
                if np.isnan(log_A_S_T):
                    payoff_list.append(0.0)
                    continue

                B_a_S_T = B_func_np_g2(S_i, T_i, a_i)
                B_b_S_T = B_func_np_g2(S_i, T_i, b_i)

                P_S_T = np.exp(log_A_S_T - B_a_S_T * x_S_i - B_b_S_T * y_S_i)
                
                payoff = max(P_S_T - K_i, 0.0)
                
            except (ValueError, ZeroDivisionError, OverflowError):
                payoff = 0.0
                
            payoff_list.append(payoff)

        true_payoff = torch.tensor(payoff_list, device=device, dtype=torch.float32).unsqueeze(1)
        
        model_input_terminal = torch.cat([t_terminal, x_terminal, y_terminal, features_batch], dim=1)
        C_terminal_pred = model(model_input_terminal)
        
        loss_terminal = loss_fn(C_terminal_pred, true_payoff)
        
        loss_total = loss_pde + lambda_terminal * loss_terminal

        if not torch.isnan(loss_total):
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    current_val_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features_batch_cpu, labels_batch_cpu in val_loader:
            features_batch = features_batch_cpu.to(device, non_blocking=True)
            labels_batch = labels_batch_cpu.to(device, non_blocking=True)
            
            t_data = torch.zeros(features_batch.size(0), 1, device=device)
            x_data = torch.zeros(features_batch.size(0), 1, device=device)
            y_data = torch.zeros(features_batch.size(0), 1, device=device)
            model_input_data = torch.cat([t_data, x_data, y_data, features_batch], dim=1)
            
            preds_val = model(model_input_data)
            loss_val = loss_fn(preds_val, labels_batch)

            if not torch.isnan(loss_val):
                current_val_loss += loss_val.item() * features_batch.size(0)
                total_samples += features_batch.size(0)

    if total_samples > 0: current_val_loss /= total_samples
    else: current_val_loss = 0.0
    
    scheduler.step()

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        break

print(f"Training finished. Best validation MSE: {best_val_loss:.8f}")

model.load_state_dict(best_model_weights)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for features_batch_cpu, labels_batch_cpu in test_loader:
        features_batch = features_batch_cpu.to(device, non_blocking=True)
        labels_batch = labels_batch_cpu.to(device, non_blocking=True)

        t_data = torch.zeros(features_batch.size(0), 1, device=device)
        x_data = torch.zeros(features_batch.size(0), 1, device=device)
        y_data = torch.zeros(features_batch.size(0), 1, device=device)
        model_input_data = torch.cat([t_data, x_data, y_data, features_batch], dim=1)
        
        preds = model(model_input_data)
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
print(f"FINAL PINN RESULTS (G2++)")
print("="*50)
print(f"Test Set MSE: {final_mse:.8f}")
print(f"Test Set R2 Score: {r2:.4f}")
print("="*50)