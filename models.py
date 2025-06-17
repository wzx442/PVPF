import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from typing import Optional

class CNN_CIFAR10(nn.Module): # d = 586,250 
    """CNN model specifically designed for CIFAR-10 dataset"""
    def __init__(self, num_classes: int):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    




# region: VCD-FL setting CNN for mnist  
"""
In VCD-FL:The architecture of the CNN is configured as two convolution layers with 5 * 5 convolution kernels, where the first is with 10 channels and the second is with 20 channels, and each is followed by 2 * 2 max pooling layer. Following the convolutional layers, there is a fully connected layer with 50 neurons and an output layer with 10 neurons. The number of parameters for the CNN is 21,780.
"""
class CNN_MNIST(nn.Module): # d = 21,840 
    def __init__(self, num_classes: int):
        super(CNN_MNIST, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=20 * 4 * 4, out_features=50)  # 修正特征尺寸为4x4
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        
        batch_size = x.size(0)
        if x.size(-1) != 4 or x.size(-2) != 4:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
            
        x = x.view(batch_size, -1)
        
        # x = torch.relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)
        return x    


# region: MLP for cifar10 and mnist
class MLP_CIFAR10(nn.Module): # d = 1707274 
    """MLP model specifically designed for CIFAR-10 dataset"""
    def __init__(self, num_classes: int):
        super(MLP_CIFAR10, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    
class MLP_MNIST(nn.Module):
    def __init__(self, num_classes=10, dim_in=784, dim_hidden=200):
        super(MLP_MNIST, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)  # 784 -> 200
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, num_classes)  # 200 -> num_classes

    def forward(self, x):
        x = x.view(-1, 784)  
        x = self.layer_input(x)  
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)  
        return x


def load_model(model_name: str, num_classes: int, dataset: str = 'cifar10', device: Optional[torch.device] = None) -> nn.Module:
    """Load and initialize a specified model.
    
    Args:
        model_name: Name of the model to load ('cnn', 'mlp', or 'resnet18')
        num_classes: Number of output classes 
        dataset: Dataset to use ('cifar10' or 'mnist')
        device: Device to load the model on (default: cuda if available, else cpu)
        
    Returns:
        nn.Module: Initialized model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = dataset.lower()
    model_name = model_name.lower()
    
    # Initialize model based on name and dataset
    if model_name == 'cnn':
        if dataset == 'cifar10':
            model = CNN_CIFAR10(num_classes)
            print("CNN_CIFAR10 model initialized")
        elif dataset == 'mnist':
            model = CNN_MNIST(num_classes)
            print("CNN_MNIST model initialized")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
    elif model_name == 'mlp':
        if dataset == 'cifar10':
            model = MLP_CIFAR10(num_classes)
            print("MLP_CIFAR10 model initialized")
        elif dataset == 'mnist':
            model = MLP_MNIST(num_classes)
            print("MLP_MNIST model initialized")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
            
    elif model_name == 'resnet18':
        # Load ResNet18 with default weights
        model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer based on dataset
        if dataset == 'cifar10':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # d = 3,179,082
            print("ResNet18_CIFAR10 model initialized")
        elif dataset == 'mnist':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # d = 3,177,930
            print("ResNet18_MNIST model initialized")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Modify final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    def apply_grad_clipping(model, optimizer, max_grad_norm=1.0):
        parameters = [p for p in model.parameters() if p.grad is not None]
        for i, p in enumerate(parameters):
            if torch.isnan(p.grad).any():
                print(f"Warning: NaN detected in parameter {i}")
                p.grad = torch.nan_to_num(p.grad, nan=0.0)
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    
    model.apply_grad_clipping = lambda optimizer, max_grad_norm=1.0: apply_grad_clipping(model, optimizer, max_grad_norm)
    
    model = model.to(device)
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Warning: NaN detected in {name}")
            param.data.zero_()
    
    return model