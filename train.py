import time

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from utils import clip_gradient, save_checkpoint, print_examples

from get_loader import get_loader
from models import Encoder, Decoder


def train(device, encoder, decoder, train_loader, criterion,
          encoder_optimizer, decoder_optimizer, grad_clip):
    encoder.train()
    decoder.train()
    
    for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        imgs = imgs.to(device)
        captions = captions.to(device)

        # Forward prop.
        features = encoder(imgs)
        outputs = decoder(features, captions[:-1])
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        
        # Backward prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)
        
        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
    return loss
    

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    load_model = False
    save_model = True
    fine_tune_encoder = False
    max_length = 30
    embed_size = 256
    hidden_size = 256
    num_layers = 1
    
    # Hyperparameters
    num_workers = 2
    batch_size = 64
    num_epochs = 100
    start_epoch = 0
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    grad_clip = 5.
    
    train_loader, dataset = get_loader(
        root_folder="data/Flickr 8k/Images/",
        annotation_file="data/captions.txt",
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )
    vocab_size = len(dataset.vocab)

    # initialize model, loss etc
    encoder = Encoder(embed_size).to(device)
    decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, patience=3)
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, patience=3)
    encoder.fine_tune(fine_tune_encoder)

    if load_model:
        checkpoint = torch.load('Saved model/checkpoint.pt', map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler'])
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler'])
        if fine_tune_encoder:
            encoder.fine_tune(fine_tune_encoder)
        
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])

    for epoch in range(0, num_epochs):
        start = time.time()
        
        loss = train(device, encoder, decoder, train_loader, criterion,
                     encoder_optimizer, decoder_optimizer, grad_clip)
        encoder_scheduler.step(loss)
        decoder_scheduler.step(loss)
        
        epoch_time = time.time() - start
        
        # Print report every epoch        
        print('Epoch - {} | time: {}:{}'.format(epoch + 1, int(epoch_time) // 60, int(epoch_time) % 60))
        
        # Test images every n epochs
        n = 10
        if (epoch + 1) % n == 0:
            print('\n=================================================================')
            print('Results on test images. Epoch {}'.format(epoch + 1))
            print_examples(encoder, decoder, device, dataset, max_length)
            print('=================================================================\n')
        
        if save_model:
            save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer,
                            encoder_scheduler, decoder_scheduler)    
