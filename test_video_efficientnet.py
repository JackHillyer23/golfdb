import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model_efficientnet import EventDetector
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish'
}

class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])
            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images))
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video', default='JackDrive_slower.mp4')
    parser.add_argument('-s', '--seq-length', type=int, default=64)
    args = parser.parse_args()

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    device = torch.device('cpu')
    print('Using device:', device)

    model = EventDetector(pretrain=True,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/swingnet_efficientnet_4500.pth.tar', map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print('Loaded EfficientNet weights')

    print('Testing...')
    for sample in dl:
        images = sample['images']
        batch = 0
        while batch * args.seq_length < images.shape[1]:
            if (batch + 1) * args.seq_length > images.shape[1]:
                image_batch = images[:, batch * args.seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * args.seq_length:(batch + 1) * args.seq_length, :, :, :]
            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Confidence: {}'.format([np.round(c, 3) for c in confidence]))

    # show detected frames
    cap = cv2.VideoCapture(args.path)
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
        cv2.imshow(event_names[i], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # confidence bar chart for this video
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4CAF50' if c >= 0.5 else '#FF9800' if c >= 0.3 else '#F44336' 
              for c in confidence]
    bars = ax.bar(list(event_names.values()), confidence, color=colors, 
                  edgecolor='black', linewidth=0.5)
    for bar, c in zip(bars, confidence):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{c:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('EfficientNet-B0 Event Detection Confidence\n{}'.format(args.path), 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('confidence_{}.png'.format(args.path.replace('.mp4', '').replace('.mov', '')), dpi=150)
    plt.show()