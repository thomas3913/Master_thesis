import torch
import torch.nn as nn

class my_madmom(nn.Module):

    def __init__(self):
        super(my_madmom,self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20,kernel_size=(3,3),padding="valid")
        self.act1 = torch.nn.ELU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout1 = torch.nn.Dropout(0.15)
        
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=20,kernel_size=(1,10),padding="valid")
        self.act2 = torch.nn.ELU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout2 = torch.nn.Dropout(0.15)
        
        self.conv3 = torch.nn.Conv2d(in_channels=20, out_channels=20,kernel_size=(3,3),padding="valid")
        self.act3 = torch.nn.ELU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(1,3))
        self.dropout3 = torch.nn.Dropout(0.15)
        
        self.name = 'tcn'
        self.dropout_rate = 0.15
        self.activation = 'elu'
        self.dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.kernel_size = 5
        self.num_filters = [20] * len(self.dilations)
        self.padding = 'same'
        
        self.res_x = torch.nn.ModuleList()
        self.layer_list = torch.nn.ModuleList()
        self.layer_list_2 = torch.nn.ModuleList()
        self.activation4 = torch.nn.ModuleList()
        self.dropout4 = torch.nn.ModuleList()
        self.conv1d = torch.nn.ModuleList()
        
        for i in self.dilations:
            self.res_x.append(torch.nn.Conv1d(in_channels=20,out_channels=20, kernel_size=1, padding='same'))
            self.layer_list.append(torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=self.kernel_size, dilation=i, padding=self.padding))
            self.layer_list_2.append(torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=self.kernel_size, dilation=i*2, padding=self.padding))
            self.activation4.append(torch.nn.ELU())
            self.dropout4.append(torch.nn.Dropout(0.15))
            self.conv1d.append(torch.nn.Conv1d(in_channels=40, out_channels=20,kernel_size=1, padding='same'))

        self.activation5 = torch.nn.ELU()

        self.beats1 = torch.nn.Dropout(0.15)
        self.beats2 = torch.nn.Linear(in_features=20,out_features=1)
        self.beats3 = torch.nn.Sigmoid()
        
        self.downbeats1 = torch.nn.Dropout(0.15)
        self.downbeats2 = torch.nn.Linear(in_features=20,out_features=1)
        self.downbeats3 = torch.nn.Sigmoid()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = torch.reshape(x,(20,-1))
                      
        # TCN:
        indx = 0
        for i, num_filters in zip(self.dilations, self.num_filters):
            # feed the output of the previous layer into the next layer
            # increase dilation rate for each consecutive layer
            
            name = self.name + '_dilation_%d' % i
            # 1x1 conv. of input (so it can be added as residual)
            res_x = self.res_x[indx](x)
                        
            conv_1 = self.layer_list[indx](x)
            conv_2 = self.layer_list_2[indx](x)
    
            # concatenate the output of the two dilations
            concat = torch.cat((conv_1,conv_2))
            
            # apply activation function
            x = self.activation4[indx](concat)
            
            # apply spatial dropout
            x = torch.transpose(x,0,1)
            x = self.dropout4[indx](x)
            x = torch.transpose(x,0,1)
            
            # 1x1 conv. to obtain a representation with the same size as the residual
            x = self.conv1d[indx](x)
            
            # add the residual to the processed data and also return it as skip connection
            x = torch.add(x,res_x)

            indx += 1
            
        # activate the output of the TCN stack
        x = self.activation5(x)
        
        beats = self.beats1(x)
        beats = torch.transpose(beats,0,1)
        beats = self.beats2(beats)
        beats = self.beats3(beats)   
        
        downbeats = self.downbeats1(x)
        downbeats = torch.transpose(downbeats,0,1)
        downbeats = self.downbeats2(downbeats)
        downbeats = self.downbeats3(downbeats) 
        
        return beats, downbeats