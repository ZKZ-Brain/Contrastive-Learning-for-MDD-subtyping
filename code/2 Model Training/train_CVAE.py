"""Train and save model"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from b_model import CVAE
from load_data import HC_subs,MDD_subs
from auxiliary_functions import save_weight

if __name__ == '__main__':

    print(torch.cuda.is_available())
    device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="VAE parameters")
    parser.add_argument('--structure',type=str,default='CVAE',help='VAE or CVAE')
    parser.add_argument('--disentangle',type=bool,default=True,help='Contrast learning or not')
    parser.add_argument('--batch',type=int,default=int(1.5e4),help='Iteration batch')
    parser.add_argument('--lr',type=float,default=5e-3,help='Learning rate')
    parser.add_argument('--input_size',type=int,default=64,help='Dimension of T1 data')
    parser.add_argument('--batch_size',type=int,default=64,help='Batch size')
    parser.add_argument('--alpha',type=float,default=1.5,help=('Weight of KL loss'))
    parser.add_argument('--beta', type=float, default=2e4, help='Weight of disentangle loss')
    args = parser.parse_args()

    "Training CVAE"
    if args.structure=='CVAE':
        print("Start training CVAE,disentangle is",args.disentangle)
        model = CVAE(disentangle=args.disentangle,input_size=args.input_size).to(device)
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
        loss = list()

        for i in tqdm(range(1,args.batch)):
            batch_idx_HC = np.random.randint(low=0, high=HC_subs.shape[0], size=args.batch_size)
            batch_idx_MDD = np.random.randint(low=0, high=MDD_subs.shape[0], size=args.batch_size)

            HC_subs_batch = HC_subs[batch_idx_HC, :, :, :]
            MDD_subs_batch = MDD_subs[batch_idx_MDD, :, :, :]

            H_input=torch.from_numpy(HC_subs_batch[:, np.newaxis, :, :,:]).to(device)
            M_input=torch.from_numpy(MDD_subs_batch[:, np.newaxis, :, :,:]).to(device)

            x_hat_h , x_hat_m, mu_h, mu_m , mu_m_specific, log_var_h, log_var_m, log_var_m_specific, disentangle_loss= model(H_input,M_input)

            reconstruction_loss = F.mse_loss(H_input,x_hat_h,reduction='sum')
            reconstruction_loss+=F.mse_loss(M_input,x_hat_m,reduction='sum')
            reconstruction_loss /= args.input_size

            # The total KL loss is the sum of three independent KL losses.
            kl_loss = 0.5 * torch.sum(torch.exp(log_var_h) + torch.pow(mu_h, 2) - 1. - log_var_h)
            kl_loss+= 0.5 * torch.sum(torch.exp(log_var_m) + torch.pow(mu_m, 2) - 1. - log_var_m)
            kl_loss+= 0.5 * torch.sum(torch.exp(log_var_m_specific) + torch.pow(mu_m_specific, 2) - 1. - log_var_m_specific)

            if args.disentangle:
                cvae_loss = reconstruction_loss.mean() + args.alpha * kl_loss+ args.beta * disentangle_loss.mean()
            else:
                cvae_loss = reconstruction_loss.mean() + args.alpha * kl_loss

            optimizer.zero_grad()
            cvae_loss.backward()
            optimizer.step()

            if np.mod(i, 10) == 0:
                print(cvae_loss.item(),reconstruction_loss.mean().item(), args.alpha * kl_loss.item(),
                      args.beta * disentangle_loss.mean().item())

            path = '../c_torch_weight'
            if np.mod(i, 100) == 0:
                loss.append(cvae_loss)
                save_weight(path, model, cvae_loss, loss)






