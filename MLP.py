import numpy as np


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        self.state = 1. / (1. + np.exp(-x))
        return self.state

    def derivative(self):

        return self.state * (1. - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        
        self.state = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return self.state

    def derivative(self):
        
        return 1. - np.square(self.state)


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        
        x[x<0]=0
        self.state = x
        return x
            
    def derivative(self):
        
        self.state[self.state>0] = 1
        return self.state


class Criterion(object):

    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y
        
        
        #LogSumExp trick
        a = np.max(x)
        e = np.exp(x-a)
        s = np.sum(e, axis=1).reshape(e.shape[0],1)
        l = x - np.log(s) - a

        self.sm = np.exp(l)
        self.loss = -1 * np.sum(np.multiply(y, l), axis=1)
        
        return self.loss
        

    def derivative(self):
        
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        self.x = x
 
        if eval:

            self.norm = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.gamma * self.norm + self.beta
        
        else:
            self.mean = (1./self.x.shape[0]) * np.sum(self.x,axis =0)
            self.var = (1./self.x.shape[0]) * np.sum((self.x - self.mean) * (self.x - self.mean), axis=0) 
            self.norm = (self.x - self.mean) / np.sqrt(self.var + self.eps)         
            self.out = self.gamma * self.norm + self.beta

            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var 
            
        return self.out


    def backward(self, delta):
                
        self.dgamma = np.sum( delta * self.norm, axis=0)
        self.dbeta = np.sum( delta, axis=0)
        dnorm = delta * self.gamma
        dvar = -1/2 * np.sum( dnorm * (self.x - self.mean) * np.power((self.var+self.eps),-3/2), axis=0)       
        dmean = -1 * np.sum(dnorm * np.power((self.var+self.eps),-1/2), axis=0) - 2./self.x.shape[0] * dvar * np.sum(self.x - self.mean, axis=0)
        
        return dnorm * np.power((self.var+self.eps), -1/2) + dvar * (2./self.x.shape[0] * (self.x - self.mean)) + dmean * 1./self.x.shape[0]


# d0: the number of inputs d1:the number of units

def random_normal_weight_init(d0, d1):
    return np.random.standard_normal((d0, d1))


def zeros_bias_init(d):
    return np.zeros((1,d))


class MLP(object):

    """
    A simple multilayer perceptron
    """
    
    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        
        self.X_size = [self.input_size] + hiddens + [self.output_size]
        
        self.W = [weight_init_fn(self.X_size[i], self.X_size[i+1] ) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]
        
        self.dW = [np.zeros((self.X_size[i], self.X_size[i+1])) for i in range(len(self.X_size)) if i+1 < len(self.X_size)] 
        self.dW_M = [np.zeros((self.X_size[i], self.X_size[i+1])) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]
        
        self.b = [bias_init_fn(self.X_size[i+1]) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]
        
        self.db = [np.zeros(self.X_size[i+1]) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]
        self.db_M = [np.zeros(self.X_size[i+1]) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(self.X_size[i+1]) for i in range(self.num_bn_layers)]

        
        #input to each layer before affine
        self.input = []

    def forward(self, x):
        
        if not self.train_mode:
            #zero out the input after each batch 
            self.input = []
        
        self.input.append(x)
        
        for i in range(len(self.activations)):
            
            act = self.activations[i]
            weight = self.W[i]
            bias = self.b[i]
            
            if i < self.num_bn_layers:
                
                if self.train_mode:
                    bn = self.bn_layers[i].forward(np.dot(self.input[i], weight) + bias)
                
                else:
                    bn = self.bn_layers[i].forward(np.dot(self.input[i], weight) + bias, eval = True) 

                y = self.activations[i].forward(bn)
                
                
            else:
                y = self.activations[i].forward(np.dot(self.input[i], weight) + bias)
                
            self.input.append(y)

        return y

    def zero_grads(self):
        
        self.dW = [np.zeros((self.X_size[i], self.X_size[i+1])) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]
        self.db = [np.zeros(self.X_size[i+1]) for i in range(len(self.X_size)) if i+1 < len(self.X_size)]
        
        #zero out the input after each batch 
        self.input = []
        
        if self.bn:
            for i in self.bn_layers:
                i.dgamma = np.zeros(i.dgamma.shape)
                i.dbeta = np.zeros(i.dbeta.shape)
            
    def step(self):
        
        for i in range(len(self.W)):
            self.dW_M[i] = self.momentum * self.dW_M[i] - self.lr * self.dW[i]
            self.W[i] = self.W[i] + self.dW_M[i]
            
            self.db_M[i] = self.momentum * self.db_M[i] - self.lr * self.db[i]
            self.b[i] = self.b[i] + self.db_M[i]
                               
        if self.bn:
            for i in self.bn_layers:
                i.gamma = i.gamma - self.lr * i.dgamma
                i.beta = i.beta - self.lr * i.dbeta

    def backward(self, labels):
        
        #calculate total Loss
        self.criterion.forward(self.input[-1], labels)
        
        #backward0 Loss wrt y
        E = self.criterion.derivative()
   
        for i in range(len(self.activations)-1, -1, -1):
        
        #backprop Loss wrt z (input to activation function)
            E = E * self.activations[i].derivative()
        
        #backprop Loss wrt batch norm
            if i < self.num_bn_layers:             
                E = self.bn_layers[i].backward(E)
            
        #backprop Loss wrt weight 
            E_w = np.dot(self.input[i].T, E) 
            self.dW[i] = E_w / self.input[i].shape[0]
            
        #backprop Loss wrt bias
            E_b = np.dot(np.ones(self.input[i].shape[0]), E)
            self.db[i] = E_b / self.input[i].shape[0]

        #backprop Loss wrt input before affine 
            E = np.dot(E, self.W[i].T)
            
        return       

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...
    

    for e in range(nepochs):

        # Per epoch setup ...
        np.random.shuffle(idxs)
        
        batch_loss_train = []
        batch_error_train = []
        batch_loss_eval = []
        batch_error_eval = []

        for b in range(0, len(trainx), batch_size):
            
            # Train ...
            mlp.train()
            mlp.zero_grads()
            mlp.forward(trainx[idxs[b:b + batch_size]])
            mlp.backward(trainy[idxs[b:b + batch_size]])
            mlp.step()
            
            batch_loss_train.append(np.average(mlp.criterion.loss))
            
            labels = np.argmax(trainy[idxs[b:b + batch_size]], axis=1)
            pred_labels = np.argmax(mlp.criterion.sm, axis=1)
            
            batch_error_train.append(np.sum(labels != pred_labels) / batch_size)

        for b in range(0, len(valx), batch_size):
            
            # Val ...
            mlp.eval()
            pred = mlp.forward(valx[b:b + batch_size])
            mlp.criterion.forward(pred, valy[b:b + batch_size])
            
            labels = np.argmax(valy[b:b + batch_size], axis=1)
            pred_labels = np.argmax(mlp.criterion.sm, axis=1)
            
            batch_loss_eval.append(np.average(mlp.criterion.loss))
            batch_error_eval.append(np.sum(labels != pred_labels) / batch_size)
            

        # Accumulate data...
        training_losses.append(sum(batch_loss_train)/len(batch_loss_train))
        training_errors.append(sum(batch_error_train)/len(batch_error_train))
        validation_losses.append(sum(batch_loss_eval)/len(batch_loss_eval))
        validation_errors.append(sum(batch_error_eval)/len(batch_error_eval))
        

    # Cleanup ...
    
    for b in range(0, len(testx), batch_size):

        # Test ...
        mlp.eval()
        pred = mlp.forward(testx[b:b + batch_size])
        pred_labels = np.argmax(pred, axis=1)

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)