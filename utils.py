import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(encoder, decoder, device, dataset, max_length):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_samples = {}
    test_samples['test_img1'] = {'path': 'test_examples/dog.jpg',
                                 'correct': 'Dog on a beach by the ocean'}
    test_samples['test_img2'] = {'path': 'test_examples/child.jpg',
                                 'correct': 'Child holding red frisbee outdoors'}
    test_samples['test_img3'] = {'path': 'test_examples/bus.png',
                                 'correct': 'Bus driving by parked cars'}
    test_samples['test_img4'] = {'path': 'test_examples/boat.png',
                                 'correct': 'A small boat in the ocean'}
    test_samples['test_img5'] = {'path': 'test_examples/horse.png',
                                 'correct': 'A cowboy riding a horse in the desert'}
    test_samples['test_img6'] = {'path': 'test_examples/girl.jpg',
                                 'correct': 'An angry girl holding a pistol'}
    test_samples['test_img7'] = {'path': 'test_examples/students.jpg',
                                 'correct': 'Students staying in the classroom'}
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    for i, (key, value) in enumerate(test_samples.items()):
        img = transform(Image.open(value['path']).convert("RGB")).unsqueeze(0)
        img = img.to(device)
        vocabulary = dataset.vocab
        result_caption = []
        
        with torch.no_grad():
            features = encoder(img).unsqueeze(0).unsqueeze(0)
            states = None
            
            for _ in range(max_length):
                hiddens, states = decoder.lstm(features, states)
                output = decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                features = decoder.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break
        
        caption = ' '.join(vocabulary.itos[idx] for idx in result_caption)
        print('Example {} CORRECT: {}'.format(i, value['correct']))
        print('Example {} OUTPUT: {}'.format(i, caption))
        print()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    encoder_scheduler, decoder_scheduler):
    state = {'epoch': epoch,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict(),
             'encoder_scheduler': encoder_scheduler.state_dict(),
             'decoder_scheduler': decoder_scheduler.state_dict()}
    filename = 'Saved model/checkpoint.pt'
    torch.save(state, filename)
