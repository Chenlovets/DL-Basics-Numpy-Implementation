import numpy as np
import math
        

class Conv1D():
    def __init__(self, in_channel, out_channel, 
                 kernel_size, stride):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.W = np.random.randn(out_channel, in_channel, kernel_size)
        self.b = np.zeros(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        self.x = x
        self.batch, __ , self.width = x.shape
        assert __ == self.in_channel, 'Expected the inputs to have {} channels'.format(self.in_channel)
        z = np.zeros((self.batch, self.out_channel, (self.width-self.kernel_size) // self.stride + 1 ))
        
        #loop through each filter
        for j in range(self.out_channel):
            z[:,j] = np.zeros((self.width-self.kernel_size) // self.stride + 1 )
            #loop through each segment
            for i, t in enumerate(range(0, self.width-self.kernel_size+1, self.stride)):
                segment = x[:, :, t:t+self.kernel_size]
                z[:,j, i]=np.einsum('ik,bik->b', self.W[j], segment)
        
        return z


    def backward(self, delta):
        
        num_segments = (self.width-self.kernel_size) // self.stride + 1
        self.db = np.einsum('bik,bik->i', np.ones((self.batch, self.out_channel, num_segments)), delta)

        for j in range(self.out_channel):
            for i in range(self.in_channel):
                for k in range(self.kernel_size):
                    for n, t in enumerate(range(0, self.width-self.kernel_size+1, self.stride)):
                        val = self.x[:, i, t+k]
                        self.dW[j,i,k] += np.einsum('b,b->', delta[:,j, n], val)                 
                    
        dx=np.zeros(self.x.shape)
        for j in range(self.out_channel):
            for i in range(self.in_channel):
                for n, t in enumerate(range(0, self.width-self.kernel_size+1, self.stride)):
                        dx[:, i, t:t+self.kernel_size] += np.einsum('k,b->bk',self.W[j,i], delta[:,j,n])
        
        return dx