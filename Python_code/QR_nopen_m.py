import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys

def QR_nopen_m(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
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
        input_dim = p 

        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.Tanh()#nn.ReLU()#nn.Tanh()#nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)
        
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.fc_out1 = nn.Linear(hidden_size, hidden_size)
        self.fc_out2 = nn.Linear(hidden_size, hidden_size)
        self.z_out = nn.Linear(hidden_size, 1)
        
        #quantile z network
        self.z_0 = nn.Linear(1, hidden_size)
        self.z_1 = nn.Linear(hidden_size, hidden_size)
        self.z_2 = nn.Linear(hidden_size, hidden_size)
        
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 
          #self.layers.append( nn.Linear(1, hidden_size) )
          #self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.Linear(1, hidden_size, bias=False) )

      def forward(self, x, a, batchnorm_on):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #out = self.bn0(out)
        out1 = self.relu( self.fc0(out) )
        out1 = self.relu( self.fc1(out1) )
        out1 = self.relu( self.fc2(out1) )
        #u = 3
        #for j in range(L-1):
        #  out = self.layers[u*j](out)
        #  out = self.layers[u*j+1](out)# + self.layers[u*j+3](a)
        #  out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        
        out_z = self.relu( self.z_0(a) )
        out_z = self.relu( self.z_1(out_z) )
        out_z = self.relu( self.z_2(out_z) )
        
        #out = torch.cat([out1, out_z], dim=1)
        out = out1 + out_z
        out = self.fc_out1(out)
        #out = self.fc_out2( self.relu(out) )
        out = self.z_out( self.relu(out) )
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
            
            ind = sample(range(n), n0)
            if n != n0 : ind = ind_split[h]
            
            X0 = X[ind,:].to(device, dtype=torch.float)
            y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
    
          
            G.zero_grad()
        #a_samp = Beta.sample()
        #alpha = a_samp * torch.ones(n,1)
        #alpha = alpha.to(device, dtype=torch.float)
            alpha = torch.rand(n0,1)
            ind = sample(range(n0), int(n0/10))
            alpha[ind] = 0.5
            alpha = alpha.to(device)
            Out_G = G(X0, alpha, batchnorm_on)
            loss = Loss( y0 - Out_G, alpha)
        
        #Out_G1 = G(X0, alpha05, lam, batchnorm_on)
        #loss += 0.3*Loss( y0 - Out_G1, alpha05 )
    
            loss.backward()
            optimG.step()
        
            loss0 += loss.item()/100
        LOSS[it] = loss.item()
        
        pen0 = 0.0
        lam0 = 0.0
        lam11 = 0.0
        pp = torch.zeros(1)
        if (it+1) % 100==0 and verb == 1:
            percent = float((it+1)*100) /iteration
            arrow   = '-' * int(percent/100 *20 -1) + '>'
            spaces  = ' ' * (20-len(arrow))
            train_time = time.perf_counter() - Generator_start
            print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
            " Current/Initial Loss: {:.4f}/{:.4f}, pen: {:.4f}, Curr pen: {:.4f}, Curr log-lam: {:.2f}, Learning rate: {:.5f}, fac: {:.1f}, Training time: {:.1f}"
            .format(float(loss0), LOSS[0], -float(pen0), -pp.item(), lam11, lr1, fac, train_time,), end='')
            loss0 = 0.0
            pen0 = 0.0
            lam0 = 0.0
            sys.stdout.flush()
    n_test = Xt.size()[0]
    N = 1000
    #device = 'cpu'
    G.eval()#.to(device)
    with torch.no_grad():
      Xt = Xt.to(device, dtype=torch.float)
      Z1 = torch.rand(N*n_test,1).to(device, dtype=torch.float)
      Z_a = 0.5*torch.ones(n_test,1).to(device, dtype=torch.float)
      
      Xb = torch.cat(N*[Xt], dim=0).to(device, dtype=torch.float)
      yb = torch.cat(N*[yt], dim=0).to(device, dtype=torch.float)
      
      Out = torch.zeros(n_lam,N,n_test)
      Out_med = torch.zeros(n_lam,n_test)
      #Out_discr = torch.zeros(n_lam)
      for i in range(1):
        lam = lam_cand[i]*torch.ones(n_test*N,1).to(device)
        ########################
        out = G(Xb, Z1, batchnorm_on)
        out = out.reshape(N,n_test)
        Out[i,:,:] = out.cpu()
        
        lam_a = lam_cand[i]*torch.ones(n_test,1).to(device)
        out = G(Xt, Z_a, batchnorm_on)
        Out_med[i,:] = out.reshape(n_test).cpu()
        
      Out = Out.detach().numpy()    
      Out_med = Out_med.detach().numpy()    
      
      lam_cand = lam_cand.detach().numpy()   
      Out_discr = 0.0
    return Out, Out_discr, lam_cand, Out_med
        
    
