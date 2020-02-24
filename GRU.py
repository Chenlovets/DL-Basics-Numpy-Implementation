import numpy as np

HIDDEN_DIM = 4

class Sigmoid:
    def __init__(self):
        pass
    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res
    def backward(self):
        return self.res * (1-self.res)
    def __call__(self, x):
        return self.forward(x)


class Tanh:
    def __init__(self):
        pass
    def forward(self, x):
        self.res = np.tanh(x)
        return self.res
    def backward(self):
        return 1 - (self.res**2)
    def __call__(self, x):
        return self.forward(x)

class Linear():
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)
        
        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        self.db = delta
        self.dW = np.dot(self.x.T, delta)
        dx = np.dot(delta, self.W.T)
        return dx

class GRU_Cell:

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()
        
        

    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        # 
        # output:
        #   - h_t: hidden state at current time-step
        self.hidden_state = h
        self.x_t = x
        
        # zt = σ(Wzhht−1 + Wzxxt)
        self.z_1 = self.Wzh.dot(h)
        self.z_2 = self.Wzx.dot(x)
        self.z_3 = self.z_1 + self.z_2
        self.z_t = self.z_act(self.z_3)
        
        # rt = σ(Wrhht−1 + Wrxxt)
        self.z_4 = self.Wrh.dot(h)
        self.z_5 = self.Wrx.dot(x)
        self.z_6 = self.z_4 + self.z_5
        self.r_t = self.r_act(self.z_6)
        
        # ht~ = tanh(Wh(rt ⊗ ht−1) + Wxxt)
        self.z_7 = self.r_t * h
        self.z_8 = self.Wh.dot(self.z_7)
        self.z_9 = self.Wx.dot(x)
        self.z_10 = self.z_8 + self.z_9
        self.h_before_combine = self.h_act(self.z_10)
        
        # ht =(1−zt) ⊗ ht−1 +zt ⊗ ht~
        self.z_11 = (1 - self.z_t) * h
        self.z_12 = self.z_t * self.h_before_combine
        self.h_t = self.z_11 + self.z_12
        #self.hidden_state = self.h_t
        
        return self.h_t

    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h
        

        dz_11 = delta
        dz_12 = delta
        dz_t = dz_12 * np.transpose(self.h_before_combine)
        dh_before_combined = dz_12 * np.transpose(self.z_t)
        dh = dz_11 * (1 - np.transpose(self.z_t))
        dz_t += dz_11 * (-1) * np.transpose(self.hidden_state)
        dz_10 = dh_before_combined * np.transpose(1 - self.h_before_combine * self.h_before_combine)
        dz_9 = dz_10
        dz_8 = dz_10
        self.x_t = self.x_t.reshape((self.x_t.shape[0],1))
        self.dWx = self.x_t.dot(dz_9).T
        dx = dz_9.dot(self.Wx)
        dz_7 = dz_8.dot(self.Wh)
        self.z_7 = self.z_7.reshape((self.z_7.shape[0],1))
        self.dWh = self.z_7.dot(dz_8).T
        dr_t = dz_7 * np.transpose(self.hidden_state)
        dh += dz_7 * np.transpose(self.r_t)
        dz_6 = dr_t * np.transpose(self.r_t) * np.transpose(1-self.r_t)
        dz_4 = dz_6
        dz_5 = dz_6
        self.dWrx = self.x_t.dot(dz_5).T
        dx += dz_5.dot(self.Wrx)
        self.hidden_state = self.hidden_state.reshape((self.hidden_state.shape[0],1))
        self.dWrh = self.hidden_state.dot(dz_4).T
        dh += dz_4.dot(self.Wrh)
        dz_3 = dz_t * np.transpose(self.z_t) * np.transpose(1-self.z_t)
        dz_1 = dz_3
        dz_2 = dz_3
        self.dWzx = self.x_t.dot(dz_2).T
        dx += dz_2.dot(self.Wzx)
        self.dWzh = self.hidden_state.dot(dz_1).T
        dh += dz_1.dot(self.Wzh)
        
        return dx, dh

# This is the neural net that will run one timestep of the input 
# to test the GRU Cell implementation is correct when used as a GRU.	
class CharacterPredictor(object):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CharacterPredictor, self).__init__()
        # The network consists of a GRU Cell and a linear layer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.rnn = GRU_Cell(input_dim, hidden_dim)
        self.linear = Linear(hidden_dim, num_classes)

    def init_rnn_weights(self, w_hi, w_hr, w_hn, w_ii, w_ir, w_in):
        self.rnn.init_weights(w_hi, w_hr, w_hn, w_ii, w_ir, w_in) 

    def __call__(self, x, h):
        return self.forward(x, h)        

    def forward(self, x, h):
        # A pass through one time step of the input
        y = self.rnn.forward(x, h)
        return self.linear.forward(y.reshape(-1,self.hidden_dim))

# An instance of the class defined above runs through a sequence of inputs to generate the logits for all the timesteps. 
def inference(net, inputs):
    # input:
    #  - net: An instance of CharacterPredictor
    #  - inputs - a sequence of inputs of dimensions [seq_len x feature_dim]
    # output:
    #  - logits - one per time step of input. Dimensions [seq_len x num_classes]
    
    hidden_state = np.zeros(net.hidden_dim)
    logits = np.zeros((inputs.shape[0], net.num_classes))
    
    for i in range(inputs.shape[0]):
        output = net.forward(inputs[i], hidden_state)
        hidden_state = net.rnn.h_t
        logits[i] = output
        
    return logits
