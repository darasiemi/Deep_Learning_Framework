import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        #print(self.kernel)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.max_cords = []
        #print(A.shape)
        new_width = A.shape[3]-self.kernel+1
        new_height = A.shape[2] - self.kernel + 1
        Z = np.zeros((A.shape[0],A.shape[1],new_height, new_width))
        #A_duplicate = np.zeros(A.shape)
        
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                for h in range(Z.shape[2]):
                    for k in range(Z.shape[3]):
                        maximum = np.max(A[i,j,h:self.kernel+h,k:self.kernel+k])
                        Z[i,j,h,k] = maximum
                        cordinates = np.where(A[i,j,h:self.kernel+h,k:self.kernel+k] == maximum)
                        cordinate1 = cordinates[0][0]
                        cordinate2 = cordinates[1][0]
                        self.max_cords.append((cordinate1, cordinate2))
 
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        counter = 0 
        dLdA = np.zeros(self.A.shape)
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for h in range(dLdZ.shape[2]):
                    for k in range(dLdZ.shape[3]):
                        set_values = np.zeros((self.kernel,self.kernel))
                        set_values[self.max_cords[counter][0], self.max_cords[counter][1]] = dLdZ[i,j,h,k]
                        #print("Z",Z[i,j,h,k])
                        #print("set values", set_values)
                        #print("1st A", A_duplicate[i,j,h:kernel+h,k:kernel+k])
                
                        dLdA[i,j,h:self.kernel+h,k:self.kernel+k] +=  set_values
                        #print("2nd A", A_duplicate[i,j,h:kernel+h,k:kernel+k])
                        counter += 1
        
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        
        new_width = A.shape[3]-self.kernel+1
        new_height = A.shape[2] - self.kernel + 1
        Z = np.zeros((A.shape[0],A.shape[1],new_height, new_width))
        #A_duplicate = np.zeros(A.shape)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                for h in range(Z.shape[2]):
                    for k in range(Z.shape[3]):
                        mean = np.mean(A[i,j,h:self.kernel+h,k:self.kernel+k])
                        Z[i,j,h,k] = mean
                        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        
        dLdA = np.zeros(self.A.shape)
        for i in range(dLdZ.shape[0]):
            for j in range(dLdZ.shape[1]):
                for h in range(dLdZ.shape[2]):
                    for k in range(dLdZ.shape[3]):
                        dLdA[i,j,h:self.kernel+h,k:self.kernel+k] += (dLdZ[i,j,h,k]/(self.kernel**2))


        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        one_stride_Z = self.maxpool2d_stride1.forward(A)
        
        Z = self.downsample2d.forward(one_stride_Z)
        
        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        one_stride_backward = self.downsample2d.backward(dLdZ)
        
        dLdA = self.maxpool2d_stride1.backward(one_stride_backward)
        
        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        one_stride_Z = self.meanpool2d_stride1.forward(A)
        
        Z = self.downsample2d.forward(one_stride_Z)
        
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        one_stride_backward = self.downsample2d.backward(dLdZ)
        
        dLdA = self.meanpool2d_stride1.backward(one_stride_backward)
        
        return dLdA
