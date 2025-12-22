"""Construct the Brain-CVAE model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from HSIC import *

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self,  channel=1,
                        disentangle=False,
                        batch_size=64,
                        input_size=64,
                        latent_dim=32,
                        kernel_size=3,
                        intermediate_dim=128,
                        nlayers=2,
                        negative_slope=0.45,
                        bias=True):
        super(VAE, self).__init__()
        self.channel = channel
        self.disentangle = disentangle
        self.batch_size = batch_size
        self.input_size=input_size
        self.latent_dim=latent_dim
        self.kernel_size=kernel_size
        self.intermediate_dim=intermediate_dim
        self.nlayers=nlayers
        self.negative_slope=negative_slope
        self.bias=bias

        self.conv_layers = nn.ModuleList()
        #Convolutional layer of encoder
        for i in range(self.nlayers):
            conv_layer = nn.Conv3d(in_channels=self.channel,
                                   out_channels=self.channel,
                                   kernel_size=self.kernel_size,
                                   stride=2)
            self.conv_layers.append(conv_layer)
            # shape info needed to build decoder model
            self.input_size=1+(self.input_size-self.kernel_size)//2

        #Fully connected layer of encoder
        self.fc_flatten = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_size*self.input_size*self.input_size, out_features=self.intermediate_dim,bias=self.bias),
            nn.ReLU())

        #get mu && log_var
        self.fc_mu=nn.Linear(in_features=self.intermediate_dim,out_features=self.latent_dim,bias=self.bias)
        self.fc_var=nn.Linear(in_features=self.intermediate_dim,out_features=self.latent_dim,bias=self.bias)

        #Fully connected layer of decoder
        self.unflatten=nn.Sequential(
            nn.Linear(in_features=self.latent_dim,out_features=self.intermediate_dim,bias=self.bias),
            nn.LeakyReLU(self.negative_slope),
            nn.Linear(in_features=self.intermediate_dim,out_features=self.input_size*self.input_size*self.input_size,bias=self.bias),
            nn.LeakyReLU(self.negative_slope)
        )

        #Convolutional layer of decoder
        self.conv_layers_decoder = nn.ModuleList()
        for i in range (self.nlayers-1):
            conv_layer_decoder = nn.ConvTranspose3d(in_channels=self.channel,
                                                    out_channels=self.channel,
                                                    kernel_size=self.kernel_size,
                                                    stride=2,
                                                    bias=self.bias)
            self.conv_layers_decoder.append(conv_layer_decoder)

        #Output layer of decoder,using out_padding for dimensionality invariant
        self.fc_output=nn.ConvTranspose3d(in_channels=self.channel,
                                          out_channels=self.channel,
                                          kernel_size=self.kernel_size,
                                          stride=2,
                                          output_padding=1,
                                          bias=self.bias)

    #reparameterization
    def _sampling(self, mu, log_var):
        batch = mu.size(0)
        dim = mu.size(1)
        epsilon = torch.randn(batch, dim).to(device)
        return mu + torch.exp(0.5 * log_var) * epsilon

    #Construct the encoder
    def _encoder(self,x):
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
        x = self.fc_flatten(x)

        mu=self.fc_mu(x)
        log_var=self.fc_var(x)

        return mu,log_var

    # Construct the decoder
    def _decoder(self,x):
        x=self.unflatten(x)
        x=x.reshape(-1,1,self.input_size,self.input_size,self.input_size)   #16*15^3——>16*1*15*15*15

        for conv_layer_decoder in self.conv_layers_decoder:
            x=F.leaky_relu(conv_layer_decoder(x),negative_slope=self.negative_slope)
        x_hat=torch.sigmoid(self.fc_output(x))
        return x_hat

    def _disentangle_func(self, x):
        x1=x[:int(x.size()[0]//2), :]
        x2=x[int(x.size()[0]//2):, :]

        disentangle_loss = hsic_normalized(x1, x2)

        return disentangle_loss


    def forward(self,*args):
        x=args[0]
        self.batch_size=x.size()[0]

        mu,log_var=self._encoder(x)        #encode
        z=self._sampling(mu,log_var)       #sample
        x_hat = self._decoder(z)           #decode

        #disentangle
        if self.disentangle:
            disentangle_loss=self._disentangle_func(z)
        else:
            disentangle_loss = 0

        return x_hat, mu, log_var,disentangle_loss

# Two identical encoders && decoder with twice the input dimension
class CVAE(VAE):
    def __init__(self,disentangle=False,input_size=64):
        super(CVAE, self).__init__()
        self.disentangle=disentangle
        self.input_size=input_size

        self.conv_layers_specific = nn.ModuleList()
        #Convolutional layer of encoder
        for i in range(self.nlayers):
            conv_layer_specific = nn.Conv3d(in_channels=self.channel,
                                   out_channels=self.channel,
                                   kernel_size=self.kernel_size,
                                   stride=2)
            self.conv_layers_specific.append(conv_layer_specific)
            # shape info needed to build decoder model
            self.input_size=1+(self.input_size-self.kernel_size)//2

        #Fully connected layer of encoder
        self.fc_flatten_specific= nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_size*self.input_size*self.input_size, out_features=self.intermediate_dim,bias=self.bias),
            nn.ReLU())

        #get mu && log_var
        self.fc_mu_specific=nn.Linear(in_features=self.intermediate_dim,out_features=self.latent_dim,bias=self.bias)
        self.fc_var_specific=nn.Linear(in_features=self.intermediate_dim,out_features=self.latent_dim,bias=self.bias)

        #Fully connected layer of decoder (with twice input size)
        self.unflatten_2=nn.Sequential(
            nn.Linear(in_features=2*self.latent_dim,out_features=self.intermediate_dim,bias=self.bias),
            nn.LeakyReLU(self.negative_slope),
            nn.Linear(in_features=self.intermediate_dim,out_features=self.input_size*self.input_size*self.input_size,bias=self.bias),
            nn.LeakyReLU(self.negative_slope)
        )

    def encoder_specific(self,x):
        for conv_layer_specific in self.conv_layers_specific:
            x = F.relu(conv_layer_specific(x))
        x = self.fc_flatten_specific(x)

        mu = self.fc_mu_specific(x)
        log_var = self.fc_var_specific(x)
        return mu, log_var

    def decoder(self,x):
        x = self.unflatten_2(x)
        x = x.reshape(-1, 1, self.input_size, self.input_size, self.input_size)  # 16*15^3——>16*1*15*15*15

        for conv_layer_decoder in self.conv_layers_decoder:    #inherited from VAE class
            x=F.leaky_relu(conv_layer_decoder(x),negative_slope=self.negative_slope)
        x_hat=torch.sigmoid(self.fc_output(x))
        return x_hat

    def creat_zero_tensor(self,x):
        return torch.zeros_like(x)

    def disentangle_func(self, x,y):
        disentangle_loss=hsic_normalized(x,y)
        return disentangle_loss

    def forward(self,*args):
        H=args[0]
        M=args[1]
        self.batch_size=H.size()[0]

        # two encoders correspond to two groups: HCs(H/TD/bg) && MDDs(M/DX/tg)
        mu_h, log_var_h = self._encoder(H)
        mu_m, log_var_m = self._encoder(M)
        mu_m_specific,log_var_m_specific=self.encoder_specific(M)   #specific encoder for MDD

        # reparameterization
        z_h = self._sampling(mu_h, log_var_h)                         # S_H in paper
        z_m = self._sampling(mu_m,log_var_m)                          # S_M
        z_m_specific=self._sampling(mu_m_specific,log_var_m_specific) # Z_M

        #feature combination
        zeros=self.creat_zero_tensor(z_h)
        Z_h=torch.cat([z_h,zeros],dim=-1)            #Concatenate HCs feature z_x from the first encoder with 0 vector
        Z_m=torch.cat([z_m,z_m_specific],dim=-1)     #Concatenate MDD feature from two encoders

        x_hat_h = self.decoder(Z_h)
        x_hat_m = self.decoder(Z_m)

        #disentangle
        if self.disentangle:
            disentangle_loss = self.disentangle_func(z_m,z_m_specific)  #HSIC
        else:
            disentangle_loss = 0

        return x_hat_h , x_hat_m, mu_h, mu_m , mu_m_specific, log_var_h, log_var_m, log_var_m_specific, disentangle_loss