import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
from util import correct_preds

event_names = ['Address', 'Toe-up', 'Mid-backswing','Top', 'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish']

def run_eval(model, seq_length, device, n_cpu=0):
    dataset = GolfDB(
        data_file='data/val_split_1.pkl',
        vid_dir='data/videos_160/',
        seq_length=seq_length,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        train=False
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=n_cpu, 
        drop_last=False)


    video_results = []  # per video, per event
    pce_scores_list = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        images = images.to(device)

        # process the video in 64 frame batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]: # handles last batch if shorter than 64 frames
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

        # calculates which events were correctly predicted within the tolerance window
        _, _, _, _, c = correct_preds(probs, gt)
        video_results.append(c)
        pce_scores_list.append(np.mean(c))
        print(f'  Video {i}: PCE={np.mean(c):.3f}')
    return np.array(video_results), np.mean(pce_scores_list)

def load_v2_scratch(device):
    from model_v2_scratch import EventDetector
    model = EventDetector(pretrain=True, 
                          width_mult=1., 
                          lstm_layers=1,
                          lstm_hidden=256, 
                          bidirectional=True, 
                          dropout=False)
    sd = torch.load('models/swingnet_v2_scratch_2000.pth.tar', map_location=device)
    model.load_state_dict(sd['model_state_dict'])
    return model

def load_v3(device):
    from model_v3 import EventDetector
    model = EventDetector(pretrain=True, 
                          width_mult=1., 
                          lstm_layers=1,
                          lstm_hidden=256, 
                          bidirectional=True, 
                          dropout=False)
    sd = torch.load('models/swingnet_v3_2000.pth.tar', map_location=device)
    model.load_state_dict(sd['model_state_dict'])
    return model

def load_efficientnet(device):
    from model_efficientnet import EventDetector
    model = EventDetector(pretrain=True, 
                          lstm_layers=1,
                          lstm_hidden=256, 
                          bidirectional=True, 
                          dropout=False)
    sd = torch.load('models/swingnet_efficientnet_2000.pth.tar', map_location=device)
    model.load_state_dict(sd['model_state_dict'])
    return model


if __name__ == '__main__':
    device = torch.device('cpu')
    seq_length = 64

    # the models to be evaluated, authors baseline is fixed
    models_config = [
        ('MobileNetV2\n(Authors)', 0.715, None),   # authors result
        ('MobileNetV2\n(Scratch)', None, load_v2_scratch),
        ('MobileNetV3-Large', None, load_v3),
        ('EfficientNet-B0', None, load_efficientnet),
    ]

    model_names = []
    pce_scores = []
    all_event_correct = {}

    for name, fixed_pce, loader in models_config:
        print(f'\nEvaluating {name}...')
        if loader is None:
            pce_scores.append(fixed_pce)
            model_names.append(name)
            continue
        model = loader(device)
        model.to(device)
        model.eval()
        event_results, pce = run_eval(model, seq_length, device)
        pce_scores.append(pce)
        model_names.append(name)
        all_event_correct[name] = event_results
        print(f'{name} PCE: {pce:.3f}')

    # PCE bar chart
    print('\nGenerating PCE bar chart...')
    fig, ax = plt.subplots(figsize=(10,6))
    # Contrasting colours for bar chart
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(model_names, pce_scores, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    for bar, score in zip(bars, pce_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('PCE (Probability of Correct Event)', fontsize=13)
    ax.set_title('Golf Swing Event Detection: PCE Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.85)
    #Authors baseline height for comparison
    ax.axhline(y=0.715, color='#2196F3', linestyle='--', alpha=0.5, label='Authors baseline (0.715)')
    
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_pce_comparison.png', dpi=150)
    print('Saved graph_pce_comparison.png')

    # Per-Event accuracy heatmap
    print('\nGenerating per-event heatmap...')
    # only includes models that were evaluated so no author baseline
    heatmap_models = [n for n in model_names if n in all_event_correct]
    accuracy_grid = np.array([all_event_correct[n].mean(axis=0) for n in heatmap_models])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(accuracy_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels(event_names, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(heatmap_models)))
    ax.set_yticklabels(heatmap_models, fontsize=11)
    
    #go through each cell and add the calculated accuracy value
    for i in range(len(heatmap_models)):
        for j in range(8):
            val = accuracy_grid[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='black' if 0.3 <val< 0.8 else 'white')
    
    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.set_title('Per-Event Detection Accuracy by Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graph_event_heatmap.png', dpi=150)
    print('Saved graph_event_heatmap.png')