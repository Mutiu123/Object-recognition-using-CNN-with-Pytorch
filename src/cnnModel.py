import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as FL
import matplotlib.pyplot as plt
import numpy as np 
from cnnArchitecture import ConvNet
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
batch_size = 4
num_epochs = 50
learning_rate = 0.001

transform = transforms.Compose(                                 # The dataset has PILImages of range [0, 1] and it need to 
           [transforms.ToTensor(),                              # transform to the Tensors of normalised range [-1, 1]
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 is 32x32 60000 colour images in 10 classes with 6000 images per classes
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform) 

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform) 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(imag):
    imag = imag / 2 + 0.5 #not normalise image
    np_image = imag.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.show()
    
# obtain some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# display images

imshow(torchvision.utils.make_grid(images))

    
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_num_steps = len(train_loader)     
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)   # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        labels = labels.to(device)   # input_layer: 3 input channels, 6 output channels, 5 kernel size
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 500 ==0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{total_num_steps}, Loss: {loss.item():.4f}') 
            
print('Training is completed')

PATH = './objectRecognition.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    num_correct = 0.0
    num_samples = 0
    num_class_correct = [0 for i in range(10)]
    num_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        num_samples +=labels.size(0)
        num_correct +=(predicted == labels).sum().item()
        
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                num_class_correct[label] += 1
                num_class_samples[label] += 1
                
    acc = num_correct * 100.0 / num_samples
    print(f'Accuracy of the network: {acc}%')
    
    for i in range(10):
        accuracy = num_class_correct[i] * 100.0 / num_class_samples[i]
        print(f'Accuracy of {classes[i]}: {accuracy}%')
    
          