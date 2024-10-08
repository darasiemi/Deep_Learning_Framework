import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_prime: (batch_size, hidden_size)
            hidden state at the current time step and current layer
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        """
        
        #print("Wih", self.W_ih.shape)
        #print("x",x.shape)

        h_prime = (np.dot(self.W_ih,x.T)).T +self.b_ih + (np.dot(self.W_hh,h.T)).T + self.b_hh # TODO
        h_prime = self.activation(h_prime)

        # return h_prime
        return h_prime

    def backward(self, delta, h, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size)
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size)
                    Hidden state at previous time step and current layer

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz =delta* self.activation.derivative(h) # TODO
        #print("dz",dz.shape)
        #print("h_prev_l", h_prev_l.shape)
        #print("h_prev_t", h_prev_t.shape)

        # 1) Compute the averaged gradients of the weights and biases
        # print("hprevl",h_prev_l.shape)
        # print("hprevt",h_prev_t.shape)
        # print("h",h.shape)
        # print("dzT",dz.T.shape)
        # print('dh_in', delta.shape)
        # print("batch_size", batch_size)
        # print("Wih", self.dW_ih.shape)
        self.dW_ih += np.dot(dz.T,h_prev_l)/batch_size # TODO
        self.dW_hh += np.dot(dz.T,h_prev_t)/batch_size # TODO
        self.db_ih += np.sum(dz,axis=0)/batch_size # TODO
        self.db_hh += np.sum(dz, axis=0)/batch_size # TODO
        # print(self.dW_hh.shape)

        # # 2) Compute dx, dh
        dx = np.dot(dz,self.W_ih) # TODO
        dh = np.dot(dz,self.W_hh) # TODO

        # 3) Return dx, dh
        return dx, dh
