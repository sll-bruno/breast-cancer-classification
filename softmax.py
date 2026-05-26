# import numpy as np

# class SoftmaxCrossEntropy:
#     def __init__(self):
#         self.A = None 

#     def forward(self, Z, Y_true):
#         """
#         Z e Y_true shape: (num_classes, batch_size)
#         """
#         # tal do overflow
#         Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
#         exp_Z = np.exp(Z_shifted)
        
#         # aqui é para calcular as probabilidades
#         # Formula do AW
#         self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        
#         # Calcula a Cross-Entropy Loss escalar
#         m = Y_true.shape[1]
#         log_probs = np.log(self.A + 1e-8)
#         loss = - (1 / m) * np.sum(Y_true * log_probs)
        
#         return loss

#     # Observação no Z
#     def backward(self, Z, Y_true):
#         """
#         Retorna o gradiente dZ para a última camada Dense.
#         """
#         dZ = self.A - Y_true
#         return dZ
    
#     def predict(self, Z):
#         Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
#         exp_Z = np.exp(Z_shifted)
        
#         probabilidades = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
#         return probabilidades