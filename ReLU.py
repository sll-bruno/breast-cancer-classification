# import numpy as np

# class ReLU:
#     """
#     Implementação de Foward e Backward para função de ativação ReLU
#     """
#     def forward(self, inputs):
#         """
#         Aplicação da função de ativação sobre os valores de input
#         ReLU(x) = max(0, x)
#         """
#         self.inputs = inputs
#         self.output = np.maximum(0, inputs)

#     def backwards(self, dvalues):
#         """
#         Passa 'para trás' o gradiente apenas onde os neuronios estavam ativos
#         Entrada:
#         - dvalues: Gradiente da camada seguinte
#         Saída:
#         - dinputs: Gradiente da camada atual que será passada para a anterior

#         O cálculo do gradiente é realizado a partir da regra da cadeia aplicada sobre a ReLU.
#         """

#         #Copia os valores dos gradientes da camada seguinte
#         self.dinputs = dvalues.copy()

#         # Aplica a derivada da ReLU: se o valor de entrada for menor ou igual a 0,
#         # o gradiente é 0,caso contrário, é mantido o valor do gradiente da camada seguinte.
#         self.dinputs[self.inputs <= 0] = 0

#         return self.dinputs