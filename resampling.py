import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A = A
        
        dim = 1
        input_size = A.shape[2]
        width = (self.upsampling_factor*input_size) - (self.upsampling_factor-1)
        Z = np.zeros((A.shape[0],A.shape[1],width))

        for i in range(A.shape[0]): 
            for j in range(A.shape[1]):

                D = np.zeros((dim,((self.upsampling_factor*input_size)-(self.upsampling_factor-1))),dtype=A[i,j].dtype)
                D[:,::self.upsampling_factor] = A[i,j]
                Z[i][j] = np.squeeze(D)
        

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        width = dLdZ.shape[2]
        dim = 1
        down_width = ((width + (self.upsampling_factor-1))//self.upsampling_factor)

        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],down_width))
        for i in range(dLdZ.shape[0]): 
            for j in range(dLdZ.shape[1]):
                D = np.zeros((dim,down_width),dtype=dLdZ[i,j].dtype)
                D = dLdZ[i,j][::self.upsampling_factor]
                dLdA[i,j] = np.squeeze(D) # TODO
        
        #dLdA = self.A #TODO

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.check = A
        
        
        self.width = A.shape[2]
        dim = 1
        down_width = ((self.width + (self.downsampling_factor-1))//self.downsampling_factor)

        Z = np.zeros((A.shape[0],A.shape[1],down_width))
        for i in range(A.shape[0]): 
            for j in range(A.shape[1]):
                D = np.zeros((dim,down_width),dtype=A[i,j].dtype)
                D = A[i,j][::self.downsampling_factor]
                Z[i,j] = np.squeeze(D) # TODO


        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        dim = 1
        input_size = dLdZ.shape[2]
        width = (self.downsampling_factor*input_size) - (self.downsampling_factor-1)
        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],self.width))
    
        for i in range(dLdZ.shape[0]): 
            for j in range(dLdZ.shape[1]):

                D = np.zeros((dim,((self.downsampling_factor*input_size)-(self.downsampling_factor-1))),dtype=dLdZ[i,j].dtype)
                D[:,::self.downsampling_factor] = dLdZ[i,j][:width]
                dLdA[i][j][:width] = np.squeeze(D) #TODO
        
                
        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        dim = 1
        input_size = A.shape[3]
        width = (self.upsampling_factor*input_size) - (self.upsampling_factor-1)
        Z = np.zeros((A.shape[0],A.shape[1],width,width))


        for x in range(A.shape[0]):
            for i in range(A.shape[1]): 
                counter = 0
                for j in range(A.shape[2]):
                    D = np.zeros((dim,((self.upsampling_factor*input_size)-(self.upsampling_factor-1))),dtype=A[x,i,j].dtype)
                    D[:,::self.upsampling_factor] = A[x,i,j]
                    if j == 0:
                        Z[x,i,j] = np.squeeze(D)
                    else:
                        counter+= self.upsampling_factor
                        Z[x,i,counter] = np.squeeze(D)
        
        #print("Z \n", Z)
        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        width = dLdZ.shape[3]
        dim = 1
        down_width = ((width + (self.upsampling_factor-1))//self.upsampling_factor)

        dLdA =  np.zeros((dLdZ.shape[0],dLdZ.shape[1],down_width,down_width))
        for x in range(dLdA.shape[0]):
           for i in range(dLdA.shape[1]):
               counter = 0
               for j in range(dLdA.shape[2]):
                   D = np.zeros((dim,down_width),dtype=dLdZ[x,i,j].dtype)
                   if j == 0:
                       D = dLdZ[x,i,j][::self.upsampling_factor]
                       dLdA[x,i,j] = np.squeeze(D)
                   else:
                       counter+=self.upsampling_factor
                       D = dLdZ[x,i,counter][::self.upsampling_factor]
                       dLdA[x,i,j] = np.squeeze(D)  #TODO

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        
        self.width = A.shape[3]
        dim = 1
        down_width = ((self.width + (self.downsampling_factor-1))//self.downsampling_factor)

        Z =  np.zeros((A.shape[0],A.shape[1],down_width,down_width))
        for x in range(Z.shape[0]):
           for i in range(Z.shape[1]):
               counter = 0
               for j in range(Z.shape[2]):
                   D = np.zeros((dim,down_width),dtype=A[x,i,j].dtype)
                   if j == 0:
                       D = A[x,i,j][::self.downsampling_factor]
                       Z[x,i,j] = np.squeeze(D)
                   else:
                       counter+=self.downsampling_factor
                       D = A[x,i,counter][::self.downsampling_factor]
                       Z[x,i,j] = np.squeeze(D) # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
            
        """
        
        dim = 1
        input_size = dLdZ.shape[3]
        width = (self.downsampling_factor*input_size) - (self.downsampling_factor-1)
        #Height and width are the same as that of A
        dLdA = np.zeros((dLdZ.shape[0],dLdZ.shape[1],self.width,self.width))
    


        for x in range(dLdZ.shape[0]):
            for i in range(dLdZ.shape[1]): 
                counter = 0
                for j in range(dLdZ.shape[2]):
                    D = np.zeros((dim,((self.downsampling_factor*input_size)-(self.downsampling_factor-1))),dtype=dLdZ[x,i,j].dtype)
                    D[:,::self.downsampling_factor] = dLdZ[x,i,j][:width]
                    if j == 0:
                        dLdA[x,i,j][:width] = np.squeeze(D)
                    else:
                        counter+= self.downsampling_factor
                        dLdA[x,i,counter][:width] = np.squeeze(D)  #TODO

        return dLdA
    
