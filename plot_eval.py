import numpy as np
import matplotlib.pyplot as plt

def smooth_predictions(pred, window=5):
    smoothed = []
    for i in range(len(pred)):
        start = max(0, i - window)
        end = min(len(pred), i + window)
        smoothed.append(np.bincount(pred[start:end]).argmax())
    return np.array(smoothed)

def get_event_transitions(sequence):
    frames = []
    values = []

    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            frames.append(i)
            values.append(sequence[i])

    return frames, values

pred = np.load("sample_pred.npy")
gt = np.load("sample_gt.npy")
min_len = min(len(pred), len(gt))
pred = pred[:min_len]
gt = gt[:min_len]

frames = np.arange(min_len)
plt.figure(figsize=(12, 3))
plt.scatter(frames, gt, label="Ground Truth", s=10)
plt.scatter(frames, pred, label="Predicted", s=10)
plt.xlabel("Frame")
plt.ylabel("Event Class")
plt.title("Raw Prediction vs Ground Truth")
plt.legend()
plt.tight_layout()
plt.savefig("raw_eval_plot.png")
plt.show()

pred_frames, pred_vals = get_event_transitions(pred)
gt_frames, gt_vals = get_event_transitions(gt)

plt.figure(figsize=(12, 3))
plt.scatter(gt_frames, gt_vals, label="Ground Truth", s=60)
plt.scatter(pred_frames, pred_vals, label="Predicted", s=60)
plt.xlabel("Frame")
plt.ylabel("Event Class")
plt.title("Event Transitions Only (Refined View)")
plt.legend()
plt.tight_layout()
plt.savefig("clean_eval_plot.png")
plt.show()

for w in [1, 2, 3]:
    pred_s = smooth_predictions(pred, window=w)
    pred_frames_s, pred_vals_s = get_event_transitions(pred_s)

    plt.figure(figsize=(12, 3))
    plt.scatter(gt_frames, gt_vals, label="Ground Truth", s=60)
    plt.scatter(pred_frames_s, pred_vals_s, label=f"Smoothed (w={w})", s=60)

    plt.xlabel("Frame")
    plt.ylabel("Event Class")
    plt.title(f"Smoothed Transitions (window={w})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"smoothed_w{w}.png")
    plt.show()