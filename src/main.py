import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import h5py
import numpy as np
import pandas as pd
import os
import argparse
import random

from dataset import StreamDataset
from model import DeepSetsRegressor
from plot import plot_loss_curve, plot_test_loss_curve, plot_real_versus_predicted

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for DeepSets Regressor")
    parser.add_argument('--h5_file', type=str, default="../data/baseline_dataset.h5", help="Path to the dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--input_dim', type=int, default=3, help="Dimension of the input dataset")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--run_idx', type=int, default=0, help="The idx of the running; also decided the random seed for reproducibility")
    parser.add_argument('--num_workers', type=int, default=7, help="number of workers used in dataloader, should be lower than the number of cpus per task in sbatch script")
    parser.add_argument('--hidden_dim', type=int, default=128, help="number of hidden dimension in our MLP model, 128 as default model, 256 is the large model")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoints/checkpoint_latest.pth if it exists")
    return parser.parse_args()

def set_random_seed(seed):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def run_evaluation(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    total = 0

    all_preds = []
    all_targets = []
    all_indices = []

    with torch.no_grad():
        for batch_data, batch_labels, batch_ids in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device).view(-1, 1)

            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_labels.size(0)
            total += batch_labels.size(0)

            all_preds.append(outputs.view(-1).cpu().numpy())
            all_targets.append(batch_labels.view(-1).cpu().numpy())
            all_indices.append(batch_ids.cpu().numpy())

    avg_loss = total_loss / total

    return (np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_indices),
            avg_loss)

if __name__ == "__main__":
    args = parse_args()
    base_seed = 42
    current_seed = base_seed + args.run_idx
    set_random_seed(current_seed)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # --- Config ---
    H5_FILE = args.h5_file
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    INPUT_DIM = args.input_dim

    print(f"Running Environment: Device={DEVICE}")
    print(f"Hyperparams: Seed={current_seed}, Input_Dim={INPUT_DIM}, Epochs={EPOCHS}")

    # --- Data Loading ---
    train_dataset = StreamDataset(H5_FILE, INPUT_DIM, mode='train')
    val_dataset = StreamDataset(H5_FILE, INPUT_DIM, mode='val')
    test_dataset = StreamDataset(H5_FILE, INPUT_DIM, mode='test')

    val_dataset.set_stats(train_dataset.mean, train_dataset.std)
    test_dataset.set_stats(train_dataset.mean, train_dataset.std)

    g = torch.Generator()
    g.manual_seed(current_seed)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    
    # --- Model Setup ---
    model = DeepSetsRegressor(input_dim=INPUT_DIM, hidden_dim=args.hidden_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    print("=== Architecture Summary ===")
    print(summary(model, input_size=(BATCH_SIZE, 4000, INPUT_DIM)))
    
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = f"checkpoints/best_model.pth"
    latest_ckpt_path = "checkpoints/checkpoint_latest.pth"
    start_epoch = 0

    # --- Resume ---
    if args.resume and os.path.isfile(latest_ckpt_path):
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_epoch = ckpt['best_epoch']
        train_loss_history = ckpt['train_loss_history']
        val_loss_history = ckpt['val_loss_history']
        test_loss_history = ckpt['test_loss_history']
        print(f"Resumed from epoch {ckpt['epoch'] + 1}, best_val_loss={best_val_loss:.4f}")
    elif args.resume:
        print(f"--resume set but no checkpoint found at {latest_ckpt_path}, starting from scratch.")

    # --- Training Loop ---
    for epoch in range(start_epoch, EPOCHS):
        # 1. Training Phase
        model.train()
        total_train_loss = 0
        train_total = 0

        for batch_data, batch_labels, _ in train_loader:
            batch_data = batch_data.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_labels.size(0)
            train_total += batch_labels.size(0)

        avg_train_loss = total_train_loss / train_total

        # 2. Validation & Test Logging Phase
        _, _, _, avg_val_loss = run_evaluation(model, val_loader, DEVICE, criterion)
        _, _, _, avg_test_loss = run_evaluation(model, test_loader, DEVICE, criterion)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            indicator = " [Best Model Saved]"
        else:
            indicator = ""

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        test_loss_history.append(avg_test_loss)

        # Save latest checkpoint for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'test_loss_history': test_loss_history,
        }, latest_ckpt_path)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}{indicator}")

    
    print("\n" + "="*40)
    print("TRAINING FINISHED. EVALUATING TEST SET...")
    print("="*40)

    print(f"Evaluation: Final Trained Model (Epoch {EPOCHS})")
    l_preds, l_targets, l_ids, l_loss = run_evaluation(model, test_loader, DEVICE, criterion)
    plot_real_versus_predicted(l_targets, l_preds, l_ids, EPOCHS, f"./results/pred_vs_true_last_{EPOCHS}.png")
    pd.DataFrame({'ID': l_ids, 'True': l_targets, 'Pred': l_preds}).to_csv(f"results/test_last_epoch{args.epochs}.csv", index=False)

    print(f"Evaluation: Best Validated Model (from Epoch {best_epoch})")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    b_preds, b_targets, b_ids, b_loss = run_evaluation(model, test_loader, DEVICE, criterion)
    plot_real_versus_predicted(b_targets, b_preds, b_ids, best_epoch, f"./results/pred_vs_true_best_{best_epoch}.png")
    pd.DataFrame({'ID': b_ids, 'True': b_targets, 'Pred': b_preds}).to_csv(f"results/test_best_epoch{best_epoch}.csv", index=False)

    print(f"Renamed the Best Validated Model")
    best_model_path_new = f"checkpoints/best_model_epoch_{best_epoch}.pt"
    os.rename(best_model_path, best_model_path_new)
    
    plot_loss_curve(train_loss_history, val_loss_history)
    plot_test_loss_curve(test_loss_history)
    
    loss_summary = pd.DataFrame({
        'Epoch': range(1, EPOCHS + 1),
        'Train_Loss': train_loss_history,
        'Val_Loss': val_loss_history,
        'Test_Loss': test_loss_history
    })
    loss_summary.to_csv(f"results/loss_history.csv", index=False)

    print(f"\n[Final Metrics on Test Set]")
    print(f"Last Epoch -> Loss: {l_loss:.4f}")
    print(f"Best Epoch -> Loss: {b_loss:.4f}")
    print(f"Results saved to 'results' and 'checkpoints' folders.")