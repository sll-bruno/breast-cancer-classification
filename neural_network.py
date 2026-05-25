 
class NeuralNetwork:
    def __init__(self, layers, loss_activation):
        self.layers = layers
        self.loss_activation = loss_activation

    def train_step(self, X, y, learning_rate):

        saida_atual = X
        for layer in self.layers:
            saida_atual = layer.foward(saida_atual)

        loss = self.loss_activation.foward(saida_atual, y)

        gradiente_atual = self.loss_activation.backward(saida_atual, y)

        for layer in reversed(self.layers):
            gradiente_atual = layer.backward(gradiente_atual)

        for layer in self.layers:

            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)

        return loss


        