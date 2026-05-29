import numpy as np

class NeuralNetwork:
    def __init__(self, layers, loss_activation):
        self.layers = layers
        self.loss_activation = loss_activation

    def fit(self, X, y, learning_rate):

        saida_atual = X
        for layer in self.layers:
            saida_atual = layer.forward(saida_atual)

        loss = self.loss_activation.forward(saida_atual, y)

        gradiente_atual = self.loss_activation.backward(saida_atual, y)

        for layer in reversed(self.layers):
            gradiente_atual = layer.backward(gradiente_atual)

        for layer in self.layers:

            if hasattr(layer, 'update'):
                layer.update(learning_rate)

        return loss

    def predict(self, X):
        saida_atual = X
        for layer in self.layers:
            saida_atual = layer.forward(saida_atual)
        return saida_atual

    def explain_instance(self, X_instance, X_baseline, target_class):
        """
        Calcula os valores SHAP para uma única instância em relação a um baseline.
        Entrada:
            - X_instance: vetor de características da instância a ser explicada (shape: (num_features, 1))
            - X_baseline: vetor de características do baseline (shape: (num_features, 1))
            - target_class: índice da classe para a qual queremos calcular a explicação (int)
        """

        saida_base = X_baseline
        for layer in self.layers:
            saida_base = layer.forward_baseline(saida_base)

        saida_atual = X_instance
        for layer in self.layers:
            saida_atual = layer.forward(saida_atual)

        num_classes = saida_atual.shape[0]
        shap_contrib = np.zeros((num_classes, 1))
        shap_contrib[target_class, 0] = 1.0

        for layer in reversed(self.layers):
            shap_contrib = layer.backward_deepshap(shap_contrib)

        shap_values = shap_contrib * (X_instance - X_baseline)

        return shap_values