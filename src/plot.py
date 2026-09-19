import matplotlib.pyplot as plt
import numpy as np

def plot_real_versus_predicted(targets, preds, all_indices, epoch_num, savefig_path = "./results/pred_vs_true.png"):
    targets = np.array(targets)
    preds = np.array(preds)
    ids = np.array(all_indices)


    plt.figure(figsize=(10, 10))

    plt.scatter(targets, preds, color='blue', alpha=0.3, s=30, label='Data Points')

    errors = np.abs(targets - preds)


    top_k = 10
    if len(errors) < top_k:
        top_k = len(errors)

    worst_indices = np.argsort(errors)[-top_k:]

    bad_targets = targets[worst_indices]
    bad_preds = preds[worst_indices]
    bad_ids = ids[worst_indices]

    plt.scatter(bad_targets, bad_preds, color='red', s=100, edgecolors='black', zorder=5, label=f'Worst {top_k} Predictions')

    for t, p, sid in zip(bad_targets, bad_preds, bad_ids):
        plt.annotate(f'ID: {sid}',
                     xy=(t, p),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     color='darkred',
                     fontweight='bold')

    data_min = min(np.min(targets), np.min(preds))
    data_max = max(np.max(targets), np.max(preds))
    buffer = (data_max - data_min) * 0.05
    plot_min = data_min - buffer
    plot_max = data_max + buffer

    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Perfect Prediction (y=x)')

    plt.title(f'Predictions vs. True Values (Epoch {epoch_num})', fontsize=16)
    plt.xlabel('True Values (Targets)', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(savefig_path, dpi=300)
    print("bad id list:", bad_ids)

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', color='orange', linewidth=2, linestyle='--')
    
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    last_epoch = len(train_losses)

    plt.annotate(f'{final_train:.4f}', 
                 xy=(last_epoch, final_train), 
                 xytext=(10, 0), textcoords='offset points', 
                 color='blue', fontweight='bold')
    plt.annotate(f'{final_val:.4f}', 
                 xy=(last_epoch, final_val), 
                 xytext=(10, 0), textcoords='offset points', 
                 color='orange', fontweight='bold')
    
    plt.title(f'Loss Over Epochs (Final Val: {final_val:.4f})', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss (Log Scale)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which="both")
    plt.yscale('log')
    
    plt.xlim(right=last_epoch * 1.15)
    
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)

def plot_test_loss_curve(test_losses):
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(test_losses) + 1)
    plt.plot(epochs, test_losses, label='Test Loss', color='green', linewidth=2)
    
    final_test = test_losses[-1]
    last_epoch = len(test_losses)

    plt.annotate(f'Final: {final_test:.4f}', 
                 xy=(last_epoch, final_test), 
                 xytext=(10, 0), textcoords='offset points', 
                 color='green', fontweight='bold')
    
    plt.title('Test Loss Performance', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss (Log Scale)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, which="both")
    plt.yscale('log')
    
    plt.xlim(right=last_epoch * 1.15)
    
    plt.tight_layout()
    plt.savefig('loss_curve_test.png', dpi=300)