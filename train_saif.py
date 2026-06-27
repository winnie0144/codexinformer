import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from models_saif import InformerSAIF
from utils_saif import SAIFScaler, get_saif_time_features


class SAIFDataset(Dataset):
    def __init__(self, data, marks, stamps, seq_len, label_len, pred_len):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.marks = torch.as_tensor(marks, dtype=torch.float32)
        self.stamps = stamps
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        dec_input = torch.zeros_like(seq_y)
        dec_input[: self.label_len] = seq_y[: self.label_len]

        mark_x = self.marks[s_begin:s_end]
        mark_y = self.marks[r_begin:r_end]
        target_stamps = pd.to_datetime(self.stamps[s_end:r_end]).strftime("%Y-%m-%d %H:%M:%S").tolist()
        target_y = seq_y[-self.pred_len :, 0]

        return seq_x, mark_x, dec_input, mark_y, target_y, target_stamps

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Informer to forecast electricity prices.")
    parser.add_argument("--data-path", default="cleaned_data.csv", help="Input CSV with SETTLEMENTDATE, RRP, TOTALDEMAND.")
    parser.add_argument("--output-dir", default=None, help="Folder for prediction CSV, plots, and checkpoint.")
    parser.add_argument("--seq-len", type=int, default=96, help="Encoder history length.")
    parser.add_argument("--label-len", type=int, default=48, help="Known decoder context length.")
    # 原本：parser.add_argument("--pred-len", type=int, default=24, help="Forecast horizon.")
    # 修改後：預設跑 30 小時，把最後 6 小時當作邊界擋箭牌
    parser.add_argument("--pred-len", type=int, default=30, help="Forecast horizon (internally 30, we slice 24).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--skip-rows", type=int, default=23, help="Rows to skip before building lag features.")
    parser.add_argument("--eval-mode", choices=["all", "random", "daily"], default="all")
    parser.add_argument("--max-eval-windows", type=int, default=None, help="Optional cap for faster evaluation.")
    parser.add_argument("--plot-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_price_data(data_path, skip_rows):
    df = pd.read_csv(data_path)
    required = {"SETTLEMENTDATE", "RRP", "TOTALDEMAND"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df[["SETTLEMENTDATE", "RRP", "TOTALDEMAND"]].copy()
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    if skip_rows > 0:
        df = df.iloc[skip_rows:].reset_index(drop=True)
    return df


def improved_nll_loss(mu, log_var, y, alpha=0.1):
    # 【關鍵手術 1】嚴格限制 log_var 的下限！
    # 原本是 min=-5.0 (代表標準差允許縮到 0.08，太小了)
    # 將下限鎖定在 min=-1.5 (代表標準差 \sigma 最小只能到 ~0.47)
    # 這等於在縮放空間地下室焊死了一塊鋼板，禁止 Loss 鑽地道作弊
    bounded_log_var = torch.clamp(log_var, min=-1.5, max=5.0)
    
    precision = torch.exp(-bounded_log_var)
    nll = 0.5 * precision * (y - mu) ** 2 + 0.5 * bounded_log_var
    
    # 真實空間還原計算
    y_dollar = torch.sinh(y)
    mu_dollar = torch.sinh(mu)
    
    weight = torch.clamp(torch.abs(y_dollar), min=1.0, max=500.0)
    mse_dollar = (y_dollar - mu_dollar) ** 2
    
    return torch.mean(nll + alpha * (weight * mse_dollar))


def make_eval_indices(mode, total_len, stamps, max_eval_windows):
    if total_len <= 0:
        return []

    if mode == "all":
        indices = list(range(total_len))
    elif mode == "random":
        sample_size = min(10, total_len)
        indices = sorted(random.sample(range(total_len), sample_size))
    else:
        sample_time_1 = pd.to_datetime(stamps[0])
        sample_time_2 = pd.to_datetime(stamps[1])
        time_delta = sample_time_2 - sample_time_1
        steps_per_day = max(1, int(pd.Timedelta(days=1) / time_delta))
        max_start_idx = max(0, total_len - (9 * steps_per_day) - 1)
        start_idx = random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        indices = [idx for idx in (start_idx + d * steps_per_day for d in range(10)) if idx < total_len]

    if max_eval_windows is not None:
        indices = indices[:max_eval_windows]
    return indices


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for bx_enc, bm_enc, bx_dec, bm_dec, by, _ in loader:
        bx_enc = bx_enc.to(device)
        bm_enc = bm_enc.to(device)
        bx_dec = bx_dec.to(device)
        bm_dec = bm_dec.to(device)
        by = by.to(device)

        optimizer.zero_grad()
        mu, log_var = model(bx_enc, bm_enc, bx_dec, bm_dec)
        
        # --- 【新增】訓練切片防禦：只拿前 24 小時算 Loss ---
        # 讓模型把尾端的邊界混亂自己吸收到第 25 ~ 30 小時去
        mu_slice = mu[:, :24]
        log_var_slice = log_var[:, :24]
        by_slice = by[:, :24]
        
        loss = improved_nll_loss(mu_slice, log_var_slice, by_slice)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate_and_save(model, test_ds, scaler, indices, save_dir, plot_samples, device):
    global_errors = []
    global_sq_errors = []
    total_inside_ci_points = 0
    total_points_evaluated = 0
    all_results = []

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            bx_enc, bm_enc, bx_dec, bm_dec, by, stamps = test_ds[idx]
            mu, log_var = model(
                bx_enc.unsqueeze(0).to(device),
                bm_enc.unsqueeze(0).to(device),
                bx_dec.unsqueeze(0).to(device),
                bm_dec.unsqueeze(0).to(device),
            )

            # --- 【新增】評估切片防禦：只截取前 24 小時進行商業結算 ---
            mu = mu[:, :24]
            log_var = log_var[:, :24]
            by = by[:24]
            stamps = stamps[:24]

            mu_scaled = mu.cpu().numpy()[0]
            log_var_scaled = log_var.cpu().numpy()[0]
            std_scaled = np.exp(0.5 * log_var_scaled)
            
            # --- 【新增】溫度校正放大係數 Beta ---
            # 依據盤後統計，常態區間誤差約 52 元，但模型區間只給 30 元，因此調整放大 2.2 倍
            beta = 2.2 
            
            # 將原本的 1.96 乘以 beta
            lower_scaled = mu_scaled - (1.96 * beta) * std_scaled
            upper_scaled = mu_scaled + (1.96 * beta) * std_scaled

            pred_mu = scaler.inverse_rrp(mu_scaled)
            lower_bound = scaler.inverse_rrp(lower_scaled)
            upper_bound = scaler.inverse_rrp(upper_scaled)
            true_val = scaler.inverse_rrp(by.numpy())

            diff = true_val - pred_mu
            ci_range = upper_bound - lower_bound
            inside = (true_val >= lower_bound) & (true_val <= upper_bound)

            global_errors.extend(np.abs(diff))
            global_sq_errors.extend(diff ** 2)
            total_inside_ci_points += int(np.sum(inside))
            total_points_evaluated += len(true_val)

           # 在迴圈內，預先算好相對步數與安全百分比誤差
            horizon_steps = np.arange(1, len(true_val) + 1)

            # 避免除以零的防彈型 APE 寫法
            safe_denominator = np.where(np.abs(true_val) < 1e-4, 1e-4, np.abs(true_val))
            ape = np.abs(diff) / safe_denominator

            all_results.append(pd.DataFrame({
                "Test_Sample_Sequence": i + 1,
                "Dataset_Index": idx,
                "Horizon_Step": horizon_steps,           # [新增] 1到24，方便分步統計衰退率
                "Timestamp": stamps,
                
                # --- 核心預測表現 ---
                "Actual_Price": true_val,
                "Predicted_Mean": pred_mu,
                "Difference_Error": diff,
                "Absolute_Pct_Error": ape,               # [新增] 商業體感誤差
                
                # --- 風險邊界(CI)表現 ---
                "CI_Lower": lower_bound,
                "CI_Upper": upper_bound,
                "CI_Width": ci_range,
                "Inside_CI": inside.astype(int),         # [新增] 1=命中區間，0=破位暴雷
            }))

            if i < plot_samples:
                save_plot(save_dir, i + 1, idx, stamps, true_val, pred_mu, lower_bound, upper_bound, diff, ci_range)

            if (i + 1) % 50 == 0 or (i + 1) == len(indices):
                print(f"Evaluated {i + 1}/{len(indices)} windows")

    report = pd.concat(all_results, ignore_index=True)
    report_path = os.path.join(save_dir, "all_test_predictions_consolidated.csv")
    report.to_csv(report_path, index=False)

    mae = float(np.mean(global_errors))
    rmse = float(np.sqrt(np.mean(global_sq_errors)))
    picp = float(total_inside_ci_points / total_points_evaluated)
    return report_path, mae, rmse, picp


def save_plot(save_dir, sample_no, dataset_idx, stamps, true_val, pred_mu, lower_bound, upper_bound, diff, ci_range):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot output.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2.5, 1]})
    ax1.plot(stamps, true_val, label="Actual Price", color="#1f77b4", marker="o", markersize=4)
    ax1.plot(stamps, pred_mu, label="Predicted Mean", color="#ff7f0e", linestyle="--")
    ax1.fill_between(stamps, lower_bound, upper_bound, color="#ff7f0e", alpha=0.2, label="95% CI")
    ax1.set_title(f"Sample {sample_no} (Dataset Index: {dataset_idx})")
    ax1.set_ylabel("RRP Price")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    colors = ["#2ca02c" if d > 0 else "#d62728" for d in diff]
    ax2.bar(range(len(diff)), diff, color=colors, alpha=0.6, label="Actual - Predicted")
    ax2.plot(range(len(diff)), ci_range, color="#9467bd", marker="x", label="CI Width")
    ax2.set_xlabel("Prediction Horizon (Hours)")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"sample_{sample_no:02d}_plot.png"))
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.abspath(args.data_path)
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"saif_informer_eval_{args.eval_mode}_{args.epochs}")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = load_price_data(data_path, args.skip_rows)
    time_marks = get_saif_time_features(df)
    scaler = SAIFScaler()
    data_scaled = scaler.fit_transform(df)

    train_size = int(args.train_ratio * len(data_scaled))
    stamps = df["SETTLEMENTDATE"].values
    train_ds = SAIFDataset(data_scaled[:train_size], time_marks[:train_size], stamps[:train_size], args.seq_len, args.label_len, args.pred_len)
    test_ds = SAIFDataset(data_scaled[train_size:], time_marks[train_size:], stamps[train_size:], args.seq_len, args.label_len, args.pred_len)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise ValueError("Not enough data for the selected seq_len, label_len, pred_len, and train_ratio.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = InformerSAIF(
        enc_in=6,   # <--- 【修改】從 4 改成 6 (對應資料的 6 個特徵)
        dec_in=6,   # <--- 【修改】從 4 改成 6
        out_len=args.pred_len,
        d_model=args.d_model,
        nhead=args.nhead,
        d_ff=args.d_ff,
        dropout=args.dropout,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(f"Device: {device}")
    print(f"Rows: {len(df)} | Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | Loss: {avg_loss:.6f} | {time.time() - start_time:.1f}s")

    checkpoint_path = os.path.join(output_dir, "informer_saif_checkpoint.pth")
    torch.save(model.state_dict(), checkpoint_path)

    eval_indices = make_eval_indices(args.eval_mode, len(test_ds), stamps[train_size:], args.max_eval_windows)
    if not eval_indices:
        raise ValueError("No evaluation windows were selected.")
    report_path, mae, rmse, picp = evaluate_and_save(model, test_ds, scaler, eval_indices, output_dir, args.plot_samples, device)

    print("=" * 60)
    print("SAIF Informer electricity price forecast complete")
    print(f"Evaluated windows : {len(eval_indices)}")
    print(f"Global MAE        : {mae:.4f}")
    print(f"Global RMSE       : {rmse:.4f}")
    print(f"Global PICP       : {picp:.2%}")
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"Predictions CSV   : {report_path}")
    print(f"Output folder     : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
