import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys

def QR_pen_c(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
    num_it1, lr1, lrdecay1, lr_power1, m1, verb1, boot_size1, test1, l1pen1, N1, K1, yt, fac,
    lam_min, lam_max, n01, pen_on1):
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
            try:
              if torch.backends.mps.is_available() == True:
                device = torch.device('mps')
                print("Training G via Apple M1 Metal computing starts!")
              else:
                device = torch.device('cpu')
                print("Training G via CPU computing starts!")
                print("WARNING: CPU computing would be very slow!")
            except:
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
    #X = X.to(device, dtype = torch.float)
    #y = y.to(device, dtype = torch.float)
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
        input_dim = p + 2 + 1
        output_dim = k
        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn_x = nn.BatchNorm1d(p)
        #self.bn1 = nn.BatchNorm2d(input_dim)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 
          self.layers.append( nn.Linear(1, hidden_size) )
          #self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.Linear(1, hidden_size, bias=False) )

      def forward(self, x, a, lam, batchnorm_on):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x, trans_a, 5.0*a, lam], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #out = self.bn0(out)
        out = self.relu(self.fc0(out))
        u = 4
        for j in range(L-1):
          out = self.layers[u*j](out)
          out = self.layers[u*j+1](out) + self.layers[u*j+3](a)
          out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        out = self.fc_out(out)
        #out = torch.sort(out)[0]
        
        return out 
      
    def Loss(x, alpha):
        relu = nn.ReLU()
        out = alpha * relu(x) + (1.0-alpha) * relu(-x)
        return out.mean()
    
    #Generator initilization
    G = GNet(S, hidden_size, L, batchnorm_on, p, n, zn).to(device)
    optimG= torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimG= torch.optim.SGD(G.parameters(), lr=lr, momentum=0.8)
    
    LOSS = torch.zeros(iteration)
    sd_y = torch.std(y).item()
    X = X.view(n,p)
    y = y.view(n,1)
    Xm = torch.cat(N*m*[X]).reshape(N,m,n,p).to(device, dtype=torch.float)
    loss0 = 0.0
    pen0 = 0.0
    lam0 = 0.0
    Generator_start = time.perf_counter()
    lr1 = lr
    n_lam = 100
    lam_cand = torch.linspace(lam_min, lam_max,n_lam)
    print("Training runs!")
    Beta = torch.distributions.beta.Beta(0.5*torch.ones(1,1), 0.5*torch.ones(1,1))
    sys.stdout.flush()
    #print("as111d")
    #sys.stdout.flush()
    alpha05 = 0.5*torch.ones(n0,1).to(device)
    K0 = int(n/n0)    
    
    #X0 = X.to(device, dtype=torch.float)#[ind,:].to(device, dtype=torch.float)
    #y0 = y.to(device, dtype=torch.float)
    
    for it in range(iteration):
        lr1 = lr/(float(it+1.0)**lrPower) 
        for param_group in optimG.param_groups:
           #sys.stdout = open(os.devnull, 'w')
           param_group["lr"] = lr1
        ind = sample(range(n), n0)
        X0 = X[ind,:].to(device, dtype=torch.float)
        y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
        
        index = np.arange(n)
        np.random.shuffle(index)
        ind_split = np.split(index, K0)
        
        for h in range(K0):
            
            #ind = sample(range(n), n0)
            #if n != n0 : ind = ind_split[h]
            
            #X0 = X[ind,:].to(device, dtype=torch.float)
            #y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
            
            lam = torch.zeros(n0,1).to(device)
            if pen_on == 1:
          #r = torch.rand(1).to(device)
          #lam11 = np.random.choice(lam_cand,1)[0]
          #if r.item() < 0.1:
          #  lam11 = lam_min
          #if r.item() > 0.9:
          #  lam11 = lam_max
          #lam12 = torch.ones(n,1)
          #lam = (lam11*lam12).to(device) + 0.2*torch.randn(1,1).to(device)
          #lam11 = lam[0].item()
              lam = np.random.choice(lam_cand,n0)
              lam = torch.from_numpy(lam).reshape(n0,1) + 0.2*torch.randn(n0,1)
              lam11 = lam[0].item()
              lam = lam.to(device)
          
            G.zero_grad()
        #a_samp = Beta.sample()
        #alpha = a_samp * torch.ones(n,1)
        #alpha = alpha.to(device, dtype=torch.float)
            alpha = torch.rand(n0,1)
            #ind = sample(range(n0), int(n0/10))
            #alpha[ind] = 0.5
            #alpha = torch.sort(alpha)[0]
            alpha.requires_grad=True
            X0.requires_grad=True
            lam.requires_grad=True
            
            alpha = alpha.to(device)
            Out_G = G(X0, alpha, lam, batchnorm_on)
            loss = Loss( y0 - Out_G, alpha)
            
            #partial derivative on quantile
            dx_a = torch.autograd.grad(Out_G, alpha, torch.ones_like(Out_G), allow_unused=True, retain_graph=True)[0]            #pwl = torch.sum(-dx_a)
            
            dx_a0 = dx_a
            da_min = torch.min(dx_a)
            da_p = torch.sum(dx_a < 0)/len(dx_a)
            
            #pwl penalty
            b = 1.0#0.00005
            dx_a = -dx_a
            #g_pen = torch.mean(dx_a.clamp_(min=0))
            g_pen = torch.mean(torch.square(dx_a.clamp_(min=0)))
            loss += g_pen*b
            
            pen = torch.zeros(1)
            if pen_on == 1:
              #a1 = torch.rand(n0,1).to(device, dtype=torch.float)
               a2 = torch.rand(n0,1).to(device, dtype=torch.float)
          #Out_G1 = G(X0, alpha, lam, batchnorm_on)
               Out_G2 = G(X0, a2, lam, batchnorm_on)
               dist = torch.abs(Out_G - Out_G2)
          #pen = dist 
               pen = torch.log(  ( dist  / sd_y )*fac  + 1.0 / fac ) 
               lam_exp = torch.exp(lam)
               pp  = (lam_exp * pen).mean()
               loss -= pp
               pen0 += pp.item()/100
#            else:
#                lam11 = lam[0].item()
#                pp = torch.zeros(1)
                
            loss.backward()
            optimG.step()
            
            loss0 += loss.item()/100
            lam0 += lam11/100
            
            #tau_p = torch.arange(0.0, 1.0, 0.05);tau_p= tau_p.reshape(len(tau_p),1).to(device)
            #tau_p.requires_grad=True
            #ind_tau = sample(range(n0), len(tau_p))
            #Out_G1 = G(X0[ind_tau,:], tau_p, lam[ind_tau,:], batchnorm_on)
            #dx_a = torch.autograd.grad(Out_G1, tau_p, torch.ones_like(Out_G1), allow_unused=True, retain_graph=True)[0]
            
            #dx_a0 = dx_a
            #da_min = torch.min(dx_a)
            #da_p = torch.sum(dx_a < 0)/len(dx_a)
            
        LOSS[it] = loss.item()
        
        if (it+1) % 100==0 and verb == 1:
            percent = float((it+1)*100) /iteration
            arrow   = '-' * int(percent/100 *20 -1) + '>'
            spaces  = ' ' * (20-len(arrow))
            train_time = time.perf_counter() - Generator_start
            print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
            " Current/Initial Loss: {:.4f}/{:.4f}, pen: {:.4f}, Curr pen: {:.4f},|| Curr neg-grad: {:.3f}%, Curr mono: {:.3f}, Curr min dx_a: {:.3f} ||, Learning rate: {:.5f}, fac: {:.1f}, Training time: {:.1f}"
            .format(float(loss0), LOSS[0], -float(pen0), -pp.item(), da_p.item()*100, g_pen.item(), da_min.item(),lr1, fac, train_time,), end='')
            loss0 = 0.0
            pen0 = 0.0
            lam0 = 0.0
            sys.stdout.flush()
    
    Out_discr = torch.cat([alpha, dx_a0], dim=1)
    n_test = Xt.size()[0]
    N = 1000
    #device = 'cpu'
    G.eval()#.to(device)
    with torch.no_grad():
      Xt = Xt.to(device, dtype=torch.float)
      Z1 = torch.rand(N*n_test,1).to(device, dtype=torch.float)
      #Z1 = torch.rand(N,n_test)#.to(device, dtype=torch.float)
      #Z1 = torch.sort(Z1, 1)[0]
      #Z1 = Z1.reshape(N*n_test,1).to(device, dtype=torch.float)
      
      #Z_a = 0.5*torch.ones(n_test,1).to(device, dtype=torch.float)
      #Z_2 = 0.9*torch.ones(n_test,1).to(device, dtype=torch.float)
      #Z_3 = 0.95*torch.ones(n_test,1).to(device, dtype=torch.float)
      
      Xb = torch.cat(N*[Xt], dim=0).to(device, dtype=torch.float)
      yb = torch.cat(N*[yt], dim=0).to(device, dtype=torch.float)
      
      Out = torch.zeros(n_lam,N,n_test)
      Out_med = torch.zeros(n_lam,n_test)
      
      tau_cand = torch.arange(0.1, 1.0, 0.1)
      Out_cross = torch.zeros(n_lam,9,n_test)
      #Out_discr = torch.zeros(n_lam)
      for i in range(n_lam):
        #######################
#        lam = lam_cand[i]*torch.ones(N,1).to(device)
#        Z1 = torch.rand(N,1);Z1=torch.sort(Z1)[0]
#        Z1 = Z1.reshape(N,1).to(device, dtype=torch.float)
#        for j in range(n_test):
#            Xt0 = Xt[j, :].reshape(1, p)
#            Xb = torch.cat(N*[Xt0], dim=0).to(device, dtype=torch.float)
#            Out[i,:,j] = G(Xb, Z1, lam, batchnorm_on).cpu().squeeze()
        ########################
        lam = lam_cand[i]*torch.ones(n_test*N,1).to(device)
        out = G(Xb, Z1, lam, batchnorm_on)
        out = out.reshape(N,n_test)
        Out[i,:,:] = out.cpu()
        
        for j in range(len(tau_cand)):
            tau_cur = tau_cand[j]
            Z_cur = tau_cur*torch.ones(n_test,1).to(device, dtype=torch.float)
            lam_a = lam_cand[i]*torch.ones(n_test,1).to(device)
            out = G(Xt, Z_cur, lam_a, batchnorm_on)
        #Out_med[i,:] = out.reshape(n_test).cpu()
            Out_cross[i,j,:] = out.reshape(n_test).cpu()
        
        #out = G(Xt, Z_2, lam_a, batchnorm_on)
        #Out_cross[i,1,:] = out.reshape(n_test).cpu()
        
        #out = G(Xt, Z_3, lam_a, batchnorm_on)
        #Out_cross[i,2,:] = out.reshape(n_test).cpu()

      Out = Out.detach().numpy()    
      Out_med = Out_med.detach().numpy()  
      Out_cross = Out_cross.detach().numpy()  
      
      lam_cand = lam_cand.detach().numpy()   
      Out_discr = Out_discr.cpu().detach().numpy() 
      
    return Out, Out_discr, lam_cand, Out_med, Out_cross
        
    
