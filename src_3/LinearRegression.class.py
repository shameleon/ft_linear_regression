
class LinearRegression:
    """ no normalization """
    def __init__(self, x_train, y_train):
        """ """
        self.x = x_train
        self.y = y_train
        return None
    
    def forward_propagation(self):
        """update_predicted_output """
        output = 

    def init_training(self):
        """ theta = [biais , weight] 
        self.biais = 0 """
        self.parameters = [0.9 , -0.9]
        self.biais = 0
        self.loss = []
        self.origins = []
        self.slopes = []

    def train_model(self, learning_rate = 0.1, epochs = 5):
        self.alpha = learning_rate
        self.epochs = epochs
        self.init_training()
        for i in range(epochs):

