import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2D_Offset(nn.Module):
    def __init__(self,mode='bilinear', padding_mode='border', align_corners=True,device = "cuda:7"):
        '''
        input tensor : NCHW
        :param input_tensor:
        '''
        super(Conv2D_Offset,self).__init__()
        self.mode = mode
        self.paading_mode = padding_mode
        self.align_corners = align_corners
        self.device = device

    def getgrid(self,row,column,dimension,batch_size):
        grid = torch.randn((batch_size,2*9*dimension,row,column),requires_grad=False,device= self.device)
        x_axis = [i for i in range(row)]
        y_axis = [i for i in range(column)]
        row_axis = torch.reshape(torch.tensor(column*x_axis),(row,column))
        row_axis = torch.transpose(row_axis,1,0)
        column_axis = torch.reshape(torch.tensor(row * y_axis), (row, column))
        for i in range(18*dimension):
            if i % 2 == 0:
                grid[:,i,:,:] = column_axis
            else:
                grid[:,i,:,:] = row_axis

        return grid/(((row*column)**0.5)/2) - 1
    def forward(self,x):
        row = x.shape[2]
        column = x.shape[3]
        dimension = x.shape[1]
        batch_size = x.shape[0]
        input_shape = x.shape
        offset_output = torch.zeros((batch_size, 9 * dimension, row, column),requires_grad=False,device=self.device)
        grid = self.getgrid(row,column,dimension,batch_size)
        # grid = nn.Parameter(torch.FloatTensor(grid))
        # dim_count = 0
        # input_dim_count = 0
        for i in range(batch_size*9*dimension):
            batch_num = i//(9*dimension)
            dim_count = i - 9*dimension*batch_num
            input_dim_count = dim_count//9
            offset = grid[batch_num,dim_count*2:(dim_count*2+2),:,:].unsqueeze(dim=0)
            offset = offset.permute(0,2,3,1)
            tmp = (F.grid_sample(x[batch_num,input_dim_count,:,:].unsqueeze(dim = 0).unsqueeze(dim = 0), offset, mode=self.mode, padding_mode=self.paading_mode, align_corners=self.align_corners)).squeeze(dim =0 )
            offset_output[batch_num, dim_count, :, :] = tmp
            # offset_output[batch_num,dim_count,:,:] = (F.grid_sample(x, offset, mode=self.mode, padding_mode=self.paading_mode, align_corners=self.align_corners)).squeeze()
        offset_output = offset_output.permute(0, 2, 3, 1)
        offset_output = torch.reshape(offset_output,(batch_size,-1,3*row,3*column))
        # offset_output = offset_output.permute(0,2,3,1)
        # offset = torch.reshape()
        return offset_output

# b = Conv2D_Offset()
# a = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],dtype=torch.float)
# a = torch.reshape(a,(1,2,3,3))
# b.forward(a)

class Conv2D_Offset(nn.Module):
    def __init__(self,mode='bilinear', padding_mode='border', align_corners=True,device = "cuda:7"):
        '''
        input tensor : NCHW
        :param input_tensor:
        '''
        super(Conv2D_Offset,self).__init__()
        self.mode = mode
        self.paading_mode = padding_mode
        self.align_corners = align_corners
        self.device = device

    def getgrid(self,row,column,dimension,batch_size):
        grid = torch.randn((batch_size,2*9*dimension,row,column),requires_grad=False,device= self.device)
        x_axis = [i for i in range(row)]
        y_axis = [i for i in range(column)]
        row_axis = torch.reshape(torch.tensor(column*x_axis),(row,column))
        row_axis = torch.transpose(row_axis,1,0)
        column_axis = torch.reshape(torch.tensor(row * y_axis), (row, column))
        for i in range(18*dimension):
            if i % 2 == 0:
                grid[:,i,:,:] = column_axis
            else:
                grid[:,i,:,:] = row_axis

        return grid/(((row*column)**0.5)/2) - 1
    def forward(self,x):
        row = x.shape[2]
        column = x.shape[3]
        dimension = x.shape[1]
        batch_size = x.shape[0]
        input_shape = x.shape
        offset_output = torch.zeros((batch_size, 9 * dimension, row, column),requires_grad=False,device=self.device)
        grid = self.getgrid(row,column,dimension,batch_size)
        # grid = nn.Parameter(torch.FloatTensor(grid))
        # dim_count = 0
        # input_dim_count = 0
        for i in range(batch_size*9*dimension):
            batch_num = i//(9*dimension)
            dim_count = i - 9*dimension*batch_num
            input_dim_count = dim_count//9
            offset = grid[batch_num,dim_count*2:(dim_count*2+2),:,:].unsqueeze(dim=0)
            offset = offset.permute(0,2,3,1)
            tmp = (F.grid_sample(x[batch_num,input_dim_count,:,:].unsqueeze(dim = 0).unsqueeze(dim = 0), offset, mode=self.mode, padding_mode=self.paading_mode, align_corners=self.align_corners)).squeeze(dim =0 )
            offset_output[batch_num, dim_count, :, :] = tmp
            # offset_output[batch_num,dim_count,:,:] = (F.grid_sample(x, offset, mode=self.mode, padding_mode=self.paading_mode, align_corners=self.align_corners)).squeeze()
        offset_output = offset_output.permute(0, 2, 3, 1)
        offset_output = torch.reshape(offset_output,(batch_size,-1,3*row,3*column))
        # offset_output = offset_output.permute(0,2,3,1)
        # offset = torch.reshape()
        return offset_output