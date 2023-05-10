import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()
        
        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        #print(self.Wrx.shape)
        #print(self.x.shape)
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r = np.dot(self.Wrx,self.x) + self.brx + np.dot(self.Wrh,self.hidden) + self.brh
        # print("r shape", self.r.shape)
        self.r = self.r_act(self.r)
        self.z = np.dot(self.Wzx, self.x)+ self.bzx + np.dot(self.Wzh, self.hidden) + self.bzh
        
        
        self.z = self.z_act(self.z)
        self.n = np.dot(self.Wnx, self.x) + self.bnx + np.multiply(self.r,(np.dot(self.Wnh,self.hidden )+self.bnh))
        self.n = self.h_act(self.n)
        h_t = (np.multiply((1-self.z), self.n)) + (np.multiply(self.z, self.hidden))
        
        # This code should not take more than 10 lines. 
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
    

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        # This code should not take more than 25 lines.
        # print(delta.shape)
        # print(self.n.shape)
        # print(self.hidden.shape)
        # print("delta",(delta.shape))
        dn = np.multiply(delta,(1-self.z))
    
        print("dn", dn)
        
        dz = np.multiply(delta, (self.hidden-self.n))
        # print("dz", dz)
        # print("dz activation", self.z_act.derivative().shape)
        # print("see",dn*self.h_act.derivative())
        # print("product",(np.dot(self.Wnh,self.hidden)))
        dr =(dn*self.h_act.derivative())*(np.dot(self.Wnh,self.hidden)+self.bnh)
        
        # print("dr",dr)
        # print("ht",np.dot(delta, self.z))
        # print("z",np.dot(self.z_act.derivative()*dz,self.Wzh))
        # print("r",(np.dot(self.r_act.derivative()*dr,self.Wrh)))
        # print("n",np.dot(self.r*self.h_act.derivative()*dn,self.Wnh))
        
        dh = np.multiply(delta, self.z)+\
            (np.dot(self.z_act.derivative()*dz,self.Wzh))+\
            (np.dot(self.r_act.derivative()*dr,self.Wrh))+\
             np.dot(self.r*self.h_act.derivative()*dn,self.Wnh)
        
        # print(self.h_act(self.hidden).shape)
        # print(self.x.shape)
        # print("dr shape",dr.shape)
        # print("first",(dn*self.h_act.derivative(dn)).shape)
        # print("second", (self.x.reshape(self.d,1)).shape)
        # print("check", self.x.T.shape)
        
        # dh = np.zeros((1,self.h))
        # print("orignal dh", dh.shape)
        # print("compare dh", ht_1.shape)

            
        self.dWnx =(dn*self.h_act.derivative()* self.x.reshape(self.d,1)).T
        # print(self.dWnx.shape)
        self.dbnx = dn*(self.h_act.derivative())
        dx  = np.dot((dn*self.h_act.derivative()),self.Wnx)+ np.dot((dz*self.z_act.derivative()),self.Wzx)+np.dot((dr*self.r_act.derivative()),self.Wrx)
        self.dbnh = np.multiply((dn*self.h_act.derivative()),self.r)
        self.dWnh = (dn*self.h_act.derivative()*np.multiply(self.r,self.hidden.reshape(self.h,1))).T
        self.dWzx = (dz*self.z_act.derivative()*self.x.reshape(self.d,1)).T
        self.dWrh = (dr*self.r_act.derivative()*self.hidden.reshape(self.h,1)).T
        self.dWrx = (dr*self.r_act.derivative()*self.x.reshape(self.d,1)).T
        self.dWzh = (dz*self.z_act.derivative()*self.hidden.reshape(self.h,1)).T
        self.dbzx = dz*self.z_act.derivative()
        self.dbrx = dr*self.r_act.derivative()
        self.dbrh = dr*self.r_act.derivative()
        self.dbnx = dn*self.h_act.derivative()
        self.dbzh = dz*self.z_act.derivative()
        
 
        
        
        # print(dx.shape)
        
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        # raise NotImplementedError
