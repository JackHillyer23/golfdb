import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from model import EventDetector
from util import correct_preds
import matplotlib.pyplot as plt

event_names = ['Address', 'Toe-up', 'Mid-backswing', 'Top',
               'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish']

def get_per_event(model, split, seq_length, device, n_cpu=0):
    dataset = GolfDB(
        data_file='data/val_split_{}.pkl'.format(split),
        vid_dir='data/videos_160/',
        seq_length=seq_length,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        train=False
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                             num_workers=n_cpu, drop_last=False)
    all_correct = []
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        images = images.to(device)
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch)
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        gt = labels.squeeze().numpy()
        _, _, _, _, c = correct_preds(probs, gt)
        all_correct.append(c)
    return np.array(all_correct)

if __name__ == '__main__':
    device = torch.device('cpu')
    seq_length = 64

    # Load authors V2 model
    from model import EventDetector as V2Detector
    v2_model = V2Detector(pretrain=True, width_mult=1., lstm_layers=1,
                          lstm_hidden=256, bidirectional=True, dropout=False)
    sd = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    v2_model.load_state_dict(sd['model_state_dict'])
    v2_model.to(device)
    v2_model.eval()
    print('Evaluating authors V2...')
    v2_correct = get_per_event(v2_model, 1, seq_length, device)

    # Load EfficientNet model
    from model_efficientnet import EventDetector as EffDetector
    eff_model = EffDetector(pretrain=True, lstm_layers=1,
                            lstm_hidden=256, bidirectional=True, dropout=False)
    sd = torch.load('models/swingnet_efficientnet_4500.pth.tar', map_location=device)
    eff_model.load_state_dict(sd['model_state_dict'])
    eff_model.to(device)
    eff_model.eval()
    print('Evaluating EfficientNet...')
    eff_correct = get_per_event(eff_model, 1, seq_length, device)

    # Build heatmap data
    heatmap_data = np.array([
        v2_correct.mean(axis=0),
        eff_correct.mean(axis=0)
    ])
    model_labels = ['MobileNetV2\n(Authors, 0.715)', 'EfficientNet-B0\n(Mine, 0.719)']

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(8))
    ax.set_xticklabels(event_names, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(2))
    ax.set_yticklabels(model_labels, fontsize=11)

    for i in range(2):
        for j in range(8):
            val = heatmap_data[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='black' if 0.3 < val < 0.8 else 'white')

    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.set_title('Per-Event Accuracy: Authors Baseline vs My Best Model', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graph_heatmap_comparison.png', dpi=150)
    print('Saved graph_heatmap_comparison.png')
    plt.show()