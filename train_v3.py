from dataloader import GolfDB, Normalize, ToTensor
from model_v3 import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import csv

if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = 2000       
    it_save = 100          # save every 100 iterations
    n_cpu = 0              # 0 is safer on Windows for CPU training
    seq_length = 64
    bs = 4                 # much smaller batch size for CPU
    k = 5                 # frozen layers

    device = torch.device('cpu')
    print('Using device:', device)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.to(device)

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])]),
                     train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # same class weights as original
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    print('Training MobileNetV3 on CPU...')
    log_file = open('mobilenetv3_loss.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['iteration', 'loss'])
    i = 0
    #checkpoint = torch.load('models/swingnet_v3_2000.pth.tar', map_location=device)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #print('Resumed from checkpoint')
    #i = 2000  # start from 2000

    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            logits = model(images)
            labels = labels.view(bs * seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            log_writer.writerow([i, loss.item()])
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()},
                           'models/swingnet_v3_{}.pth.tar'.format(i))
                print('Saved model at iteration {}'.format(i))
            if i == iterations:
                break

    print('Training complete.')
    log_file.close()