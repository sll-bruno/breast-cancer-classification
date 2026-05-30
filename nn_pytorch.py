import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchNeuralNetwork(nn.Module):
    def __init__(self, n_features, init_method="xavier"):
        super(PyTorchNeuralNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2) #Camada sem ativação. Pytorch usa Softmax com CrossEntropyLoss.
        )
        
        self._initialize_weights(init_method)

    def forward(self, x):
        return self.network(x)

    def _initialize_weights(self, init_method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_method == "he":
                    # Kaiming normal é o equivalente PyTorch para a inicialização He
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif init_method == "xavier":
                    # Xavier normal
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "random":
                    # Inicialização aleatória simples
                    nn.init.normal_(m.weight, mean=0.0, std=0.5)
                
                # Zera os bias
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
