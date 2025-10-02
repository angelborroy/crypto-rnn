import json
import numpy as np

# Read training history
loss_hist = np.loadtxt('meta/vigenere/loss_hist.txt', delimiter=',')
acc_hist = np.loadtxt('meta/vigenere/acc_hist.txt', delimiter=',')

training_stats = {
    'total_steps': int(loss_hist[-1, 0]),
    'final_loss': float(loss_hist[-1, 1]),
    'final_accuracy': float(acc_hist[-1, 1]),
    'total_training_time_seconds': float(np.sum(loss_hist[:, 2])),
    'avg_time_per_step': float(np.mean(loss_hist[:, 2]))
}

print(json.dumps(training_stats, indent=2))

with open('training_summary_baseline.json', 'w') as f:
    json.dump(training_stats, f, indent=2)
