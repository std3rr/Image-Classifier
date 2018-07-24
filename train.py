import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import argparse
from collections import OrderedDict
from time import time, sleep
import sys, os

from PIL import Image


def main():
    
    in_args = get_args()
    netw = Network(
        dataloaders=create_dataloaders(in_args.data_dir), 
        arch=in_args.arch, 
        gpu=in_args.gpu, 
        checkpoint=in_args.checkpoint,
        save_dir=in_args.save_dir,
        epochs=in_args.epochs,
        learning_rate=in_args.learning_rate,
        hidden_units=in_args.hidden_units
    )
    netw.train(epochs=in_args.epochs)

def get_args():
    """
    Get command line arguments and return a dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',help='Directory containing the images')
    parser.add_argument('--arch',type=str, default='list',help='CNN model architecture to use for image classification')
    parser.add_argument('--hidden_units',type=int, default=512, help='Number of hidden units to add before classification')
    parser.add_argument('--learning_rate',type=int, default=0.001, help='Predefined learning rate for gradient decsent')
    parser.add_argument('--epochs',type=int, default=4, help='Number of epochs to train')
    parser.add_argument('--save_dir',type=str, default='checkpoints',help='Custom directory so save checkpoints')
    parser.add_argument('--checkpoint',help='checkpoint model file to load')
    parser.add_argument('--gpu',type=bool, const=True, nargs='?',default=False, help='Enable gpu/cuda mode')
    
    parsed_args = parser.parse_args()
    if parsed_args.arch == 'list':
        archlist = [key for key,val in models.__dict__.items() if str(val).startswith("<function ")]
        print("Available base architechtures:")
        print("------------------------------")
        for name in archlist:
            print(f'  {name}')
        exit()
        
    return parsed_args




def create_dataloaders(data_dir):
    """
    Define datasets and transforms for training, testing and validation
    Parameters:
        data directory path
    Returns:
        image datasets and dataloader for training, testing and validation
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(33),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (.485,.456,.406),
                (.229,.224,.225)
            )        
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (.485,.456,.406),
                (.229,.224,.225)        
            )
        ]),    
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (.485,.456,.406),
                (.229,.224,.225)
            )         
        ])
    }

    images = OrderedDict()
    images.datasets = OrderedDict([
        ('training', datasets.ImageFolder(train_dir, transform=data_transforms['training'])),
        ('test', datasets.ImageFolder(train_dir, transform=data_transforms['test'])),
        ('validation', datasets.ImageFolder(train_dir, transform=data_transforms['validation']))  
    ])
    images.dataloaders = OrderedDict([
        ('training', torch.utils.data.DataLoader(images.datasets['training'], shuffle=True, batch_size=64, pin_memory=True)),
        ('test', torch.utils.data.DataLoader(images.datasets['test'], batch_size=32, pin_memory=True)),
        ('validation', torch.utils.data.DataLoader(images.datasets['validation'], batch_size=32, pin_memory=True)),    
    ])
    return images


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    size = 256
    ccrop = 224
    
    margin = ( size - ccrop ) / 2

    image = image.resize((size,size))
    image = image.crop( (margin, margin, size-margin, size-margin) )

    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ( image - mean ) / std
    image = image.transpose((2,0,1))

    return image


class Network():
    def __init__(self, dataloaders=None, arch='resnet101', hidden_units=512, learning_rate=0.001, save_dir='', epochs=4, gpu=False, checkpoint=None):

        self.device = 'cpu'
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.save_dir = save_dir
        
        self.epochs = 0 # this is the epochs processed counter
        load_pretrained = True
        self.class_to_idx = {}
        
        if dataloaders != None:
            self.images = dataloaders
            self.class_to_idx = self.images.datasets['training'].class_to_idx
        elif checkpoint != None:
            load_pretrained = False # asume no training

        self.load(checkpoint=checkpoint) if checkpoint != None else None
        self.model = getattr(models, arch)(pretrained=load_pretrained)
                
        # remove the last layer and check if its a 'classifier' or 'fc' (feature map)
        # This dependes on architecture
        
        prev_last_layer = self.model._modules.popitem()
        self.classifier_key = prev_last_layer[0]
        
        # disable gradient descent backprop for pretrained layers
        # as we have poped the last layer we can update all of the model left
        
        for p in self.model.parameters():
            p.requires_grad = False
                
        our_in_features = 0
        our_out_features = len(self.class_to_idx)
        
        if self.classifier_key == 'fc':
            our_in_features = prev_last_layer[1].in_features
            
        elif self.classifier_key == 'classifier':
            # if we want to keep original classifier arch just 
            # update out_features of last linear func and reattach..
            # prev_last_layer[1][-1].out_features = 102
            # self.model._modules.update([prev_last_layer])
            # for this project e will use our own though..

            our_in_features = prev_last_layer[1][0].in_features
            
        else:
            print(f"Error: does not recognize last layer of architechture 'self.arch'!", prev_last_layer)
        
        # Attach and replace last layer with our own classifier
        self.model._modules.update([(self.classifier_key, nn.Sequential(OrderedDict([
            ('fc1', nn.Linear( our_in_features, hidden_units )),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear( hidden_units, len(self.class_to_idx) )),
            ('output', nn.LogSoftmax(dim=1))    
        ])))])
        
 
        if gpu:
            if not torch.cuda.is_available():
                print(f"Error: sorry, no gpu seem available, will use {self.device}!")
            else:
                self.device = 'cuda'
                #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # define error loss function and optimizer
        self.criterion = nn.NLLLoss()
        # Only register optimizer to our "new" classifier layers
        self.optimizer = optim.Adam(self.model._modules[self.classifier_key].parameters(), lr=learning_rate)
        
        # default fallback to either pick up a loaded state 
        # or try to load existing checkpoint
        if hasattr(self, 'model_state'):
            print("Loading prefetched model state...")
            self.model.load_state_dict(self.model_state)
        else:
            self.load()

        
    def train(self, epochs=4, logg_every=1):
        
        steps = 0
        # save checkpoint
        self.model.to(self.device)
        self.model.train()
        dataloader = self.images.dataloaders['training']
        total_images = len(dataloader) * dataloader.batch_size
        init_epochs = self.epochs
        
        for e in range(epochs):
            running_loss = 0
            proc_images = 0
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                proc_images += len(images)
                
                self.optimizer.zero_grad()
                
                outputs = self.model.forward(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                if steps % logg_every == 0:
                    print("Epoch: {} of {}, images {} of {}...".format(self.epochs+1, epochs+init_epochs, proc_images, total_images),
                          "Loss: {:.4f}".format(running_loss/logg_every))
                    running_loss = 0  
                steps+=1
                
            self.epochs += 1
            self.save()
            self.validate()
            
    def validate(self, set='validation', data_dir=None):
        
        total = 0
        correct = 0
        self.model.eval()
        self.model.to(self.device)
        try:
            dataloader = self.images.dataloaders[set]
        except:
            self.images = create_dataloaders(data_dir)
            dataloader = self.images.dataloaders[set]

        imgcount = len(dataloader.dataset)
        proc_images = 0
        batch_count = 0
        running_loss = 0
        
        print("Running validation...")
        with torch.no_grad():
            for images, labels in dataloader:
                if self.device == 'cuda':
                    images, labels = images.cuda(), labels.cuda(async=True)

                batch_count += 1
                proc_images += len(images)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #percent = (100 * correct / total)
                #print(f'Accurracy of {proc_images} flower test images: {percent}%')

            percent = (100 * correct / total)
            loss = running_loss / batch_count
            print(f'Accurracy on {imgcount} {set} images: {percent}%, loss: {loss}')       
        
                    
    def save(self, save_dir='checkpoints'):

        # objects save_dir will override parameter..
        if hasattr(self, 'save_dir'):
            save_dir = self.save_dir
         
        filename = save_dir+'/'+self.arch+'.checkpoint'
        if not os.path.exists(filename):
            print(f"Could'nt find any {filename}.","Will be created after first training epoch.")
            if not os.path.exists(save_dir):
                  os.makedirs(save_dir)
            return
            
        checkpoint = {
            'epochs': self.epochs,
            'arch': self.arch,
            'model_state': self.model.state_dict(),
            #'classifier_state': self.model._modules[self.classifier_key].state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'class_to_idx' : self.class_to_idx
        }
        
        torch.save(checkpoint, filename)
        print(f'Saved checkpoint to {filename}')
        
    def load(self, save_dir='checkpoints', checkpoint=None):
        
        # objects save_dir will override parameter..
        if hasattr(self, 'save_dir'):
            save_dir = self.save_dir
            
        filename = checkpoint if checkpoint != None else save_dir+'/'+self.arch+'.checkpoint'
        if not os.path.exists(filename):
            print(f"Could'nt find any {filename}.","Will be created after first training epoch.")
            if not os.path.exists(save_dir):
                  os.makedirs(save_dir)
            return
        
        
        print(f'Loading {filename}')
        c = torch.load(filename)
        try:
            self.model.load_state_dict(c['model_state'])
            #self.optimizer.load_state_dict(c['optimizer_state']
        except:
            self.model_state = c['model_state']
        #self.model._modules[self.classifier_key].load_state_dict(c['classifier_state'])
        self.class_to_idx = c['class_to_idx']
        self.epochs = c['epochs']
        try:
            self.arch = c['arch']
        except:
            None
            
    def predict(self, image, topk=5):
        
        self.model.eval()
        if self.device == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.model.cuda()   
                    
        with torch.no_grad():
            im = Image.open(image)
            npim = process_image( im )
            tensor = torch.Tensor( npim )
            tensor.unsqueeze_(0)
            tensor.cuda()

            out = self.model( tensor )
            pred = torch.exp( out ).data.topk(topk)

            probs = pred[0].cpu().numpy().squeeze()
            idx_to_class = dict(map(reversed, self.class_to_idx.items()))
            classes = [ idx_to_class[i] for i in pred[1].cpu().numpy()[0] ]

            return probs, classes, npim

                
if __name__ == "__main__":
    main()