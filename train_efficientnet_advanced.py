from dataloader import GolfDB, Normalize, ToTensor
from model_efficientnet import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv

if __name__ == '__main__':
    split = 1
    iterations = 4500
    it_save = 100
    n_cpu = 0
    seq_length = 64
    bs = 4
    k = 3

    device = torch.device('cpu')
    print('Using device:', device)

    model = EventDetector(pretrain=True,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    checkpoint = torch.load('models/swingnet_efficientnet_3500.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Resumed from iteration 2000')

    model.train()
    model.to(device)

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
                     train=True)

    data_loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                             num_workers=n_cpu, drop_last=True)

    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    losses = AverageMeter()

    log_file = open('efficientnet_continued_loss.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['iteration', 'loss'])

    print('Continuing EfficientNet training...')
    i = 3500
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            logits = model(images)
            labels = labels.view(bs * seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            writer.writerow([i, losses.val])
            log_file.flush()
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()},
                           'models/swingnet_efficientnet_{}.pth.tar'.format(i))
                print('Saved checkpoint at iteration {}'.format(i))
            if i == iterations:
                break

    log_file.close()
    print('Done.')