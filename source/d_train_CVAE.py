"""Trains VAE and CVAE and save models."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from b_model import CVAE
from c_load_data import HC_subs,MDD_subs
from h_auxilary_functions import save_weight

if __name__ == '__main__':

    print(torch.cuda.is_available())
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

    # Set default parameters
    parser = argparse.ArgumentParser(description="VAE parameters")
    parser.add_argument('--structure',type=str,default='CVAE',help='VAE or CVAE')
    parser.add_argument('--disentangle',type=bool,default=True,help='Contrast learning or not')
    parser.add_argument('--epoch',type=int,default=int(1e2),help='Iteration epoch')
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate')
    parser.add_argument('--input_size',type=int,default=64,help='Dimension of T1 data')
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size')
    parser.add_argument('--alpha', type=float, default=20000, help='Weight of disentangle loss')
    parser.add_argument('--beta',type=float,default=2.0,help=('Weight of KL loss'))
    args = parser.parse_args()

    "Training CVAE"
    if args.structure=='CVAE':
        print("Start training CVAE,disentangle is",args.disentangle)
        model = CVAE(disentangle=args.disentangle,input_size=args.input_size,beta=args.beta).to(device)
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
        loss = list()

        for i in tqdm(range(1,args.epoch)):
            batch_idx_HC = np.random.randint(low=0, high=HC_subs.shape[0], size=args.batch_size)
            batch_idx_MDD = np.random.randint(low=0, high=MDD_subs.shape[0], size=args.batch_size)

            HC_subs_batch = HC_subs[batch_idx_HC, :, :, :]
            MDD_subs_batch = MDD_subs[batch_idx_MDD, :, :, :]

            x_input=torch.from_numpy(HC_subs_batch[:, np.newaxis, :, :,:]).to(device)
            y_input=torch.from_numpy(MDD_subs_batch[:, np.newaxis, :, :,:]).to(device)

            x_hat_x , x_hat_y, mu_x, mu_y , mu_y_specific, log_var_x, log_var_y, log_var_y_specific , tc_loss , disentangle_loss= model(x_input,y_input)

            reconstruction_loss = F.mse_loss(x_input,x_hat_x,reduction='sum')
            reconstruction_loss+=F.mse_loss(y_input,x_hat_y,reduction='sum')
            reconstruction_loss /= args.input_size

            # The total KL loss is the sum of three independent KL losses.
            kl_loss = 0.5 * torch.sum(torch.exp(log_var_x) + torch.pow(mu_x, 2) - 1. - log_var_x)
            kl_loss+= 0.5 * torch.sum(torch.exp(log_var_y) + torch.pow(mu_y, 2) - 1. - log_var_y)
            kl_loss+= 0.5 * torch.sum(torch.exp(log_var_y_specific) + torch.pow(mu_y_specific, 2) - 1. - log_var_y_specific)

            if args.disentangle:
                cvae_loss = reconstruction_loss.mean() + args.beta * kl_loss+ args.alpha * disentangle_loss.mean()
            else:
                cvae_loss = reconstruction_loss.mean() + args.beta * kl_loss

            # backpropagation
            optimizer.zero_grad()
            cvae_loss.backward()
            optimizer.step()

            if np.mod(i, 10) == 0:  # print training loss
                print(cvae_loss.item(),reconstruction_loss.mean().item(), args.beta * kl_loss.item(),
                      args.alpha * disentangle_loss.mean().item())

            path = '../c_torch_weight/CVAE'
            if np.mod(i, 100) == 0:  # Save best model every 100 batchs
                loss.append(cvae_loss)
                save_weight(path, model, cvae_loss, loss)








