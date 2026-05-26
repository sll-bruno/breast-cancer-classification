 
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



        