import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys

def CondWGAN(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
    num_it1, lr1, lrdecay1, lr_power1, m1, verb1, boot_size1, test1, l1pen1, N1, K1, yt, fac,
    lam_min, lam_max, n01, pen_on1):
    
    print("Training via WGAN")
    
    gpu_ind = int(gpu_ind1)
    if gpu_ind == -1:
        device = 'cpu'
        print("Training G via CPU computing starts.")
        print("WARNING: CPU computing would be very slow!")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda', gpu_ind)
            print("Training G via GPU computing starts!")
            print(device)
        else:
            device = torch.device('cpu')
            print("Training G via CPU computing starts!")
            print("WARNING: CPU computing would be very slow!")
    sys.stdout.flush()
    if torch.is_tensor(X) == False: X = torch.from_numpy(X)
    if torch.is_tensor(y) == False: y = torch.from_numpy(y)
    if torch.is_tensor(Xt) == False: Xt = torch.from_numpy(Xt)
    if torch.is_tensor(yt) == False: yt = torch.from_numpy(yt)
#    print("7019100as111d")
#    sys.stdout.flush()
    X = X.to(device, dtype = torch.float)
    y = y.to(device, dtype = torch.float)
    Xt = Xt.to(device, dtype = torch.float)
    yt = yt.to(device, dtype = torch.float)

    
    L = int(L1)
    k = int(K1)
    zn = int(zn1)
    N = int(N1)
    NN_type = str(NN_type1)
    S = int(S1)
    p = int(p1)
    n = int(n1)
    hidden_size = int(hidden_size1)
    batchnorm_on = int(batchnorm_on1)
    iteration = int(num_it1)
    lr = float(lr1)
    m = int(m1)
    verb = int(verb1)
    size = int(boot_size1)
    lrDecay = int(lrdecay1)
    lrPower = float(lr_power1)
    test = int(test1)
    l1pen = int(l1pen1)
    fac = float(fac)
    lam_min = float(lam_min)
    lam_max = float(lam_max)
    n0 = int(n01)
    pen_on = int(pen_on1)
    #print("as111d")
    #sys.stdout.flush()
    n = X.size()[0]
    p = X.size()[1]
    
    class GNet(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn):
        super(GNet, self).__init__()
        input_dim = p + zn 
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn_x = nn.BatchNorm1d(p)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 

      def forward(self, x, z, batchnorm_on):
        #x = self.bn_x(x)
        #x_m = torch.cat(5*[x, torch.sign(x)*torch.log(torch.abs(x)+1.0)], dim=1)
        #lam_m = torch.cat(5*[0.2*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        out1 = torch.cat([x, z], dim=1)
        #out = self.bn0(out1)
        out = self.relu(self.fc0(out1))
        
        for j in range(L-1):
          out = self.layers[3*j](out)
          out = self.layers[3*j+1](out)
          out = self.layers[3*j+2](out)
          
        out = self.fc_out(out)
        return out      
      
    class DNet(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn):
        super(DNet, self).__init__()
        input_dim = p + 1 
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn_x = nn.BatchNorm1d(p+1)
        #self.bn1 = nn.BatchNorm2d(input_dim)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 

      def forward(self, x, g, batchnorm_on):
        y_m = torch.cat([x, g], dim=1)
        #y_m = self.bn_x(y_m)
        #lam_m = torch.cat(5*[0.2*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        out1 = y_m
        #out = self.bn0(out1)
        out = self.relu(self.fc0(out1))
        
        for j in range(L-1):
          out = self.layers[3*j](out)
          out = self.layers[3*j+1](out)
          out = self.layers[3*j+2](out)
          
        out = self.fc_out(out)
        return out      
  

    def Loss_D(D_fake, D_real):
        out = D_real - D_fake  
        return out.mean()
      
    def Loss_G(D_fake):
        out = D_fake  
        return out.mean()
    
    def gradient_penalty(D, real, fake, device="cpu"):
        epsilon = torch.rand(n0).reshape(n0,1).to(device)
        interpolated = real*epsilon + fake*(1-epsilon)
        mixed_scores = D(X0, interpolated, batchnorm_on)
        
        gradient = torch.autograd.grad(
            inputs = interpolated,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_pen = torch.mean((gradient_norm-1)**2)
        return gradient_pen
        
        
        
        
        
    #Generator initilization
    G = GNet(S, hidden_size, L, batchnorm_on, p, n, zn).to(device)
    D = DNet(S, hidden_size, L, batchnorm_on, p, n, zn).to(device)
    
    optimG= torch.optim.Adam(G.parameters(), lr=lr)
    optimD= torch.optim.Adam(D.parameters(), lr=lr)
    LOSS = torch.zeros(iteration)
    sd_y = torch.std(y).item()
    X = X.view(n,p)
    y = y.view(n,1)
    Xm = torch.cat(N*m*[X]).reshape(N,m,n,p).to(device, dtype=torch.float)
    loss0 = 0.0
    loss00 = 0.0
    pen0 = 0.0
    lam0 = 0.0
    Generator_start = time.perf_counter()
    lr1 = lr
    #lam_min = -10.0
    #lam_max = -1.0 # or 3.0
    K0 = int(n/n0)
    n_lam = 200
    lam_cand = np.linspace(lam_min, lam_max,n_lam)
    print("Training runs!")
    sys.stdout.flush()
    for it in range(iteration):
        
        loss0 = 0.0
        loss00 = 0.0
        pen0 = 0.0
        lam0 = 0.0
        
        lr1 = lr/(float(it+1.0)**lrPower) 
        for param_group in optimD.param_groups:
           sys.stdout = open(os.devnull, 'w')
           param_group["lr"] = lr1
        for param_group in optimG.param_groups:
           sys.stdout = open(os.devnull, 'w')
           param_group["lr"] = lr1
           
        #index = np.arange(n)
        #np.random.shuffle(index)
        #ind_split = np.split(index, K0)
        for h in range(K0):
            
            ind = sample(range(n), n0)
            X0 = X[ind,:].reshape(n0,p)
            y0 = y[ind,:].reshape(n0,1)
            ####################################################
            #for para in D.parameters():
            #    sys.stdout = open(os.devnull, 'w')
             #   para.requires_grad = False
            #for para in G.parameters():
            #    sys.stdout = open(os.devnull, 'w')
            #    para.requires_grad = True
            sys.stdout = sys.__stdout__
            
            lam = torch.zeros(n0,1).to(device)
            Z = torch.randn(n0,zn).to(device, dtype=torch.float)
            
            if pen_on == 1:
              r = torch.rand(1).to(device)
              lam11 = np.random.choice(lam_cand,1)[0]
              if r.item() < 0.1:
                lam11 = lam_min
              if r.item() > 0.9:
                lam11 = lam_max
              lam12 = torch.ones(n0,1)
              lam = (lam11*lam12).to(device) + 0.1*torch.randn(1,1).to(device)
            
            lam11 = lam[0]
            #print("error2")
            #sys.stdout.flush()    
            G.zero_grad()
            Out_G1 = G(X0, Z, batchnorm_on)
            D_fake1 = D(X0, Out_G1, batchnorm_on)
            loss_G = Loss_G(D_fake1)
            pen = torch.zeros(1)
            #print("asdasdTraining runs!")
            #sys.stdout.flush()
            if pen_on == 1:
              Z2 = torch.randn(n0,zn).to(device, dtype=torch.float)
              Out_G2 = G(X0, Z2, batchnorm_on)
              dist = torch.abs(Out_G1 - Out_G2)
              pen = torch.mean( torch.log(  (  dist  / sd_y ) * fac  + 1.0 / fac ) )
              #print("iopkfapowTraining runs!")
              #sys.stdout.flush()
              lam_exp = torch.exp(lam11).reshape(1).to(device)
              pp  = lam_exp * pen / sd_y
              loss_G -= pp.mean()
            loss00 += loss_G.item()
            loss_G.backward()
            optimG.step()
            pen0 += pen/100
            
            #################
            #for para in D.parameters():
            #   sys.stdout = open(os.devnull, 'w')
            #   para.requires_grad = True
            #   para.data.clamp_(-0.001, 0.001)
            #for para in G.parameters():
            #    sys.stdout = open(os.devnull, 'w')
            #    para.requires_grad = False
            sys.stdout = sys.__stdout__
            #for j1 in range(1):
            D.zero_grad()
            Z = torch.randn(n0,zn).to(device, dtype=torch.float)
            Out_G = G(X0, Z, batchnorm_on)
            D_fake = D(X0, Out_G, batchnorm_on)
            D_real = D(X0, y0, batchnorm_on)
            grad_pen = gradient_penalty(D, y0, Out_G, device)
            loss_D = Loss_D(D_fake, D_real) + 0.1*grad_pen
            loss_D.backward() 
            optimD.step()
            loss0 += loss_D.item()/100
            lam0 += lam11/100
        
        LOSS[it] = loss00
        
        if (it+1) % 100==0 and verb == 1:
            percent = float((it+1)*100) /iteration
            arrow   = '-' * int(percent/100 *20 -1) + '>'
            spaces  = ' ' * (20-len(arrow))
            train_time = time.perf_counter() - Generator_start
            print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
            " Current/Initial Loss: {:.4f}/{:.4f}, pen: {:.4f}, log-lam: {:.2f}, Learning rate: {:.5f}, fac: {:.1f}, Training time: {:.1f}"
            .format(float(loss0), LOSS[0], -pen.item(), lam11.item(), lr1, fac, train_time,), end='')
            loss0 = 0.0
            loss00 = 0.0
            pen0 = 0.0
            lam0 = 0.0
            sys.stdout.flush()
    #generation step
#    y_hat = np.zeros((size, n, 1))
#    Size = size
#    sub_size = int(Size/m)
#    Xb = torch.cat(Size*[X]).reshape(sub_size,m,n,p).to(device, dtype=torch.float)
#    Wb = torch.from_numpy(np.random.exponential(scale=1, size=Size*n*S).reshape(sub_size,m,n,S)).to(device, dtype=torch.float)
#    Zb = torch.from_numpy(np.random.uniform(low=0, high=1, size=Size*n*zn).reshape(sub_size,m,n,zn)).to(device, dtype=torch.float)
#    Out = G(Wb, Xb, Zb, batchnorm_on).reshape(sub_size,m,n,1).cpu().detach().numpy()
#    for i in range(sub_size):
#        y_hat[m*i:(m*(i+1)),:,:] = Out[i,:,:,:]
    n_test = Xt.size()[0]
    #print(Out_G1[:,0])
    #print(Out1/2)
    N = 1000
    G.eval()
    lam_cand0 = lam_cand
    with torch.no_grad():
      Z1 = torch.randn(N*n_test,zn).to(device, dtype=torch.float)
      Z2 = torch.randn(N*n_test,zn).to(device, dtype=torch.float)
      
      Xb = torch.cat(N*[Xt], dim=0).to(device, dtype=torch.float)
      yb = torch.cat(N*[yt], dim=0).to(device, dtype=torch.float)
      
      Out = torch.zeros(n_lam,N,n_test)
      Out_discr = torch.zeros(n_lam)
      for i in range(n_lam):
        lam = lam_cand0[i]*torch.ones(n_test*N,1).to(device)
        ########################
        out1 = G(Xb, Z1, batchnorm_on)
        out2 = G(Xb, Z2, batchnorm_on)
        
        #der2 = G(Xb + eps1 + eps2, Z, lam, batchnorm_on) - G(Xb + eps2, Z, lam, batchnorm_on) - G(Xb + eps1, Z, lam, batchnorm_on) + G(Xb, Z, lam, batchnorm_on) 
        d1 = torch.abs(yb - out2)
        d2 = torch.abs(out1 - out2)
        d = 2*d1 - d2
        Out_discr[i] =  torch.mean(d).cpu() 
        #########################
        out1 = out1.reshape(N,n_test)
        out2 = out2.reshape(N,n_test)
        Out[i,:,:] = out1.cpu()
        
      #print(Out[49,:,:])
      Out = Out.detach().numpy()                
      Out_discr = Out_discr.detach().numpy()
    return Out, Out_discr, lam_cand0
        
    
