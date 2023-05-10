# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        #print("A shape", self.A.shape)
        #print(A[0])
        #print("W shape", self.W.shape)
        
        stop = self.A.shape[2]-self.W.shape[2]+1
        #print("stop",stop)
        Z = np.zeros((self.A.shape[0],self.W.shape[0],stop))

        for i in range(self.A.shape[0]):
             for w in range(self.W.shape[0]):
                  for j in range(stop):
                       Z[i, w, j] = np.sum(np.multiply(self.A[i][:,j:self.W[w].shape[1]+j], self.W[w])) + self.b[w]
        
        #print("Z shape",Z.shape)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        
        self.dLdW = np.zeros((self.W.shape[0],self.W.shape[1],self.W.shape[2]))
        stride1 = self.A.shape[2]-dLdZ.shape[2]+1
        for i in range(dLdZ.shape[1]):
            for j in range(self.A.shape[1]):
                for k in range(stride1):
                    self.dLdW[i,j,k] = np.sum(np.multiply(self.A[:,j,k:dLdZ.shape[2]+k], dLdZ[:, i, :])) # TODO
                    
        
    
        self.dLdb = dLdZ.sum(axis=(2,0)) # TODO
        
        
       
        flipped_filter = self.W[:,:,::-1]
        new_width = dLdZ.shape[2]+(2*(self.W.shape[2]-1))
        padded_dLdZ = np.zeros((dLdZ.shape[0],dLdZ.shape[1],new_width))
        for i in range(dLdZ.shape[0]):
            padded_dLdZ[i] =np.pad(dLdZ[i],((0,0),((self.W.shape[2]-1),(self.W.shape[2]-1))), mode = "constant", constant_values = 0)

        stride2 = padded_dLdZ.shape[2]-flipped_filter.shape[2]+1
        dLdA = np.zeros((self.A.shape[0],self.A.shape[1],stride2))
    
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                for k in range(stride2):
                    dLdA[i,j,k] =np.sum(np.multiply(padded_dLdZ[i,:,k:flipped_filter.shape[2]+k],flipped_filter[:,j,:]))
        
        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None) # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        # TODO
        one_stride_Z = self.conv1d_stride1.forward(A)
    
        # downsample
        Z = self.downsample1d.forward(one_stride_Z) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        one_stride_backward = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(one_stride_backward) # TODO 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        self.A = A
        stop = self.A.shape[3]-self.W.shape[3]+1
        new_height = self.A.shape[2]-self.W.shape[2]+1
        #print("stop",stop)
        Z = np.zeros((self.A.shape[0],self.W.shape[0],new_height,stop))

        for i in range(self.A.shape[0]):
            for w in range(self.W.shape[0]):
                for h in range(new_height):
                    for j in range(stop):
                        Z[i, w, h, j] = np.sum(np.multiply(self.A[i,:,h:self.W.shape[2]+h,j:self.W.shape[3]+j], self.W[w,:])) + self.b[w] #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        #print("dLdZ shape", dLdZ.shape)
        
        self.dLdW = np.zeros((self.W.shape[0],self.W.shape[1],self.W.shape[2],self.W.shape[3]))
        stride1 = self.A.shape[3]-dLdZ.shape[3]+1
        new_height2 = self.A.shape[2]-dLdZ.shape[2]+ 1
        
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                for h in range(new_height2):
                    for k in range(stride1):
                        #print(np.sum(np.multiply(self.A[:,j,h:dLdZ.shape[2]+h,k:dLdZ.shape[3]+k], dLdZ[:, i, :, :])))
                        self.dLdW[i,j,h,k] = np.sum(np.multiply(self.A[:,j,h:dLdZ.shape[2]+h,k:dLdZ.shape[3]+k], dLdZ[:, i, :, :])) # TODO        
        
        self.dLdb =  dLdZ.sum(axis=(3,2,0)) # TODO
        
        flipped_filter = self.W[:,:,:,::-1][:,:,::-1,:]
        #print("flipped filter height", flipped_filter.shape[2])
        #print("flipped filter width", flipped_filter.shape[3])
        new_width = dLdZ.shape[3]+(2*(self.W.shape[3]-1))
        new_height = dLdZ.shape[2]+(2*(self.W.shape[2]-1))
        padded_dLdZ = np.zeros((dLdZ.shape[0],dLdZ.shape[1],new_height,new_width))
        #print(padded_dLdZ.shape)
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                padded_dLdZ[i,j] = np.pad(dLdZ[i,j],(((self.W.shape[2]-1), (self.W.shape[2]-1)),((self.W.shape[3]-1),(self.W.shape[3]-1))), mode = "constant", constant_values = 0)
        
        #print("paddede dLdZ shape", padded_dLdZ.shape)
        #print("dLdZ shape", dLdZ.shape)


        stride2 = padded_dLdZ.shape[3]-flipped_filter.shape[3]+1
        dLdA = np.zeros((self.A.shape[0],self.A.shape[1],self.A.shape[2],stride2))

        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                for h in range(self.A.shape[2]):
                    for k in range(stride2):
                        #print("padded dLdZ 2", padded_dLdZ[i,:,h:flipped_filter.shape[2]+h,k:flipped_filter.shape[2]+k].shape)
                        #print("flipped filter",flipped_filter[:,j,:,:].shape)
                        #print(np.sum(np.multiply(padded_dLdZ[i,:,h:flipped_filter.shape[2]+h,k:flipped_filter.shape[3]+k],flipped_filter[:,j,:,:])))
                        
                        dLdA[i,j,h,k] =np.sum(np.multiply(padded_dLdZ[i,:,h:flipped_filter.shape[2]+h,k:flipped_filter.shape[3]+k],flipped_filter[:,j,:,:])) # TODO

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None) # TODO
        self.downsample2d = Downsample2d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        one_stride_Z = self.conv2d_stride1.forward(A)
    
        # downsample
        Z = self.downsample2d.forward(one_stride_Z) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        # TODO
        one_stride_backward = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(one_stride_backward) # TODO 

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled =  self.upsample1d.forward(A)#TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA =   self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,weight_init_fn, bias_init_fn):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d =  Upsample2d(upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)  #TODO

        dLdA = self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.shape_of_A = A.shape
        
        Z = np.zeros((A.shape[0],(A.shape[1]*A.shape[2])))
        for i in range (A.shape[0]):
            Z[i] = A[i].flatten() # TODO
        #Z = np.reshape(A, (A.shape[0],(A.shape[1]*A.shape[2])))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        #dLdA = np.reshape(dLdZ,(self.shape_of_A))
        
        dLdA = np.zeros(self.shape_of_A)
        for i in range(dLdA.shape[0]):
           dLdA[i] = dLdZ[i].reshape((self.shape_of_A[1], self.shape_of_A[2]))#TODO

        return dLdA

