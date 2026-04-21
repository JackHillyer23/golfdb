import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from model import EventDetector
from util import correct_preds

def eval_split(model, split, seq_length, device, n_cpu=0):
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
    correct = []
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
        correct.append(np.mean(c))
    return np.mean(correct)

if __name__ == '__main__':
    device = torch.device('cpu')
    seq_length = 64

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()

    pces = []
    for split in [1, 2, 3, 4]:
        pce = eval_split(model, split, seq_length, device)
        print('Split {} PCE: {:.3f}'.format(split, pce))
        pces.append(pce)

    print('\nAverage PCE across all splits: {:.3f}'.format(np.mean(pces)))