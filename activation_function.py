import numpy as np

class ActivationFunction():
    def forward(self, Z):
        raise NotImplementedError
        
    def backward(self, dA):
        raise NotImplementedError   
    
class ReLU(ActivationFunction):
    """
    Implementação de Foward e Backward para função de ativação ReLU
    """
    def forward(self, inputs):
        """
        Aplicação da função de ativação sobre os valores de input
        ReLU(x) = max(0, x)
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
        return self.output

    def backward(self, dvalues):
        """
        Passa 'para trás' o gradiente apenas onde os neuronios estavam ativos
        Entrada:
        - dvalues: Gradiente da camada seguinte
        Saída:
        - dinputs: Gradiente da camada atual que será passada para a anterior

        O cálculo do gradiente é realizado a partir da regra da cadeia aplicada sobre a ReLU.
        """

        #Copia os valores dos gradientes da camada seguinte
        self.dinputs = dvalues.copy()

        # Aplica a derivada da ReLU: se o valor de entrada for menor ou igual a 0,
        # o gradiente é 0,caso contrário, é mantido o valor do gradiente da camada seguinte.
        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs

    def forward_baseline(self, inputs_base):
        self.inputs_base = inputs_base
        self.output_base = np.maximum(0, inputs_base)
        return self.output_base

    def backward_deepshap(self, incoming_shap):
        delta_y = self.output - self.output_base
        delta_x = self.inputs - self.inputs_base
        
        is_zero = np.isclose(delta_x, 0)
        multiplier = np.zeros_like(self.inputs)
        
        multiplier[~is_zero] = delta_y[~is_zero] / delta_x[~is_zero]
        multiplier[is_zero] = np.where(self.inputs[is_zero] > 0, 1, 0)
        
        return incoming_shap * multiplier
    

class SoftmaxCrossEntropy(ActivationFunction):
    def __init__(self):
        self.A = None 

    def forward(self, Z, Y_true):
        """
        Z e Y_true shape: (num_classes, batch_size)
        """
        # tal do overflow
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        
        # aqui é para calcular as probabilidades
        # Formula do AW
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        
        # Calcula a Cross-Entropy Loss escalar
        m = Y_true.shape[1]
        log_probs = np.log(self.A + 1e-8)
        loss = - (1 / m) * np.sum(Y_true * log_probs)
        
        return loss

    # Observação no Z
    def backward(self, Z, Y_true):
        """
        Retorna o gradiente dZ para a última camada Dense.
        """
        dZ = self.A - Y_true
        return dZ
    
    def predict(self, Z):
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        
        probabilidades = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return probabilidades