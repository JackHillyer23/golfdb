import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(predicted_events, gt_events, num_frames, title="Prediction vs Ground Truth"):
    """
    pred_events: list of predicted frame indices
    gt_events: list of ground truth frame indices
    num_frames: total number of frames
    """

    timeline = np.arange(num_frames)

    pred_line = np.zeros(num_frames)
    gt_line = np.zeros(num_frames)

    pred_line[predicted_events] = 1
    gt_line[gt_events] = 1

    plt.figure(figsize=(12, 3))
    plt.plot(timeline, gt_line, label="Ground Truth", linewidth=2)
    plt.plot(timeline, pred_line, label="Predicted", linestyle='dashed')
    plt.yticks([0, 1])
    plt.xlabel("Frame")
    plt.ylabel("Event")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.show()