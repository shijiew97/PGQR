import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys

def GQT(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
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
    #device = 'cpu'    
    sys.stdout.flush()
    if torch.is_tensor(X) == False: X = torch.from_numpy(X)
    if torch.is_tensor(y) == False: y = torch.from_numpy(y)
    if torch.is_tensor(Xt) == False: Xt = torch.from_numpy(Xt)
    if torch.is_tensor(yt) == False: yt = torch.from_numpy(yt)

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
    
    dim_s = y.shape[1]
    #print(dim_s)
    #print(p)
        
    class GNet(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn, dim_s):
        super(GNet, self).__init__()
        input_dim = p + dim_s + 2
        output_dim = dim_s
        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        
        self.fc0 = nn.Linear(input_dim, hidden_size)
        #self.fc_out = nn.Linear(hidden_size, dim_s)
        
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn_x = nn.BatchNorm1d(p)
        #self.fc_out = nn.Linear(hidden_size, dim_s)
        
        #self.fc_outs1 = nn.Linear(dim_s, hidden_size)
        #self.fc_outs = nn.Linear(dim_s, 1, bias=False)
        self.fc_outs = nn.Linear(hidden_size, 1)
        #self.fc_outs = nn.Linear(hidden_size, 1, bias=False)
        
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          self.layers.append( nn.ReLU() ) 
          self.layers.append( nn.Linear(1, hidden_size) )
          #self.layers.append( nn.Linear(dim_s, hidden_size) )

      def forward(self, x, a, s):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x, trans_a, 5.0*a, s], dim=1)
        out = self.relu(self.fc0(out))
        u = 4
        for j in range(L-1):
          out = self.layers[u*j](out)
          out = self.layers[u*j+1](out)+self.layers[u*j+3](a)#+self.layers[u*j+4](s)
          out = self.layers[u*j+2](out) 
          
        #out = self.fc_out(out)
        out = self.fc_outs(out)
        #out1 = self.relu(self.fc_outs1(out))
        #u=5
        #for j in range(L-1):
        #  out1 = self.layers[u*j](out1)
        #  out1 = self.layers[u*j+1](out1)+self.layers[u*j+3](a)+self.layers[u*j+4](s)
        #  out1 = self.layers[u*j+2](out1) 
          
        #outs = self.relu(self.fc_outs1(out))
        #outs = self.fc_outs((out1))
        #outs = torch.sum(out*s, dim=1)
        
        return out
      
    def Loss(x, alpha):
        relu = nn.ReLU()
        out = alpha * relu(x) + (1.0-alpha) * relu(-x)
        return out.mean()
    
    #Generator initilization
    G = GNet(S, hidden_size, L, batchnorm_on, p, n, zn, dim_s).to(device)
    optimG= torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimG= torch.optim.SGD(G.parameters(), lr=lr, momentum=0.8)
    LOSS = torch.zeros(iteration)
    sd_y = torch.std(y).item()
    X = X.view(n,p)
    y = y.view(n,2)
    #print("terawe")
    #sys.stdout.flush()
    
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
    #sys.stdout.flush()
    
    K0 = int(n/n0)    
    for it in range(iteration):
        lr1 = lr/(float(it+1.0)**lrPower) 
        for param_group in optimG.param_groups:
           #sys.stdout = open(os.devnull, 'w')
           param_group["lr"] = lr1
        #ind = sample(range(n), n0)
        #X0 = X[ind,:].to(device, dtype=torch.float)
        #y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
        index = np.arange(n)
        np.random.shuffle(index)
        ind_split = np.split(index, K0)
        
        for h in range(K0):
            ind = sample(range(n), n0)
            if n != n0 : ind = ind_split[h]
            X0 = X[ind,:].to(device, dtype=torch.float).reshape(n0,p)
            y0 = y[ind,:].reshape(n0,2).to(device, dtype=torch.float)


            G.zero_grad()
            #alpha = torch.rand(n0,1)
            #ind = sample(range(n0), int(n0/20))
            #alpha[ind] = 0.5
            alpha = torch.ones(n0,1)*torch.rand(1)
            
            s_vec = torch.randn(n0, dim_s).to(device)
            s_vec = s_vec / torch.sqrt(torch.sum(s_vec**2, dim=1)).reshape(n0,1)

            alpha = alpha.to(device)
            Out_G = G(X0, alpha, s_vec)
            loss = Loss( torch.sum(y0*s_vec, dim=1).reshape(n0,1) - Out_G, alpha)
            #print("90123aoiegjoew")
            #sys.stdout.flush()
            loss.backward()
            optimG.step()
            loss0 += loss.item()/100
            
        LOSS[it] = loss.item()
        
        if (it+1) % 100==0 and verb == 1:
            percent = float((it+1)*100) /iteration
            arrow   = '-' * int(percent/100 *20 -1) + '>'
            spaces  = ' ' * (20-len(arrow))
            train_time = time.perf_counter() - Generator_start
            print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
            " Current/Initial Loss: {:.4f}/{:.4f}, pen: {:.4f}, Learning rate: {:.5f}, n: {}, y_dim:{}, Training time: {:.1f}"
            .format(float(loss0), LOSS[0], -float(pen0), lr1, n, dim_s, train_time,), end='')
            loss0 = 0.0
            pen0 = 0.0
            lam0 = 0.0
            sys.stdout.flush()
    n_test = Xt.size()[0]
    N = 1000
    device = 'cpu'
    G.eval().to(device)
    
    with torch.no_grad():
      Xt = Xt.to(device, dtype=torch.float)
      #Z1 = torch.rand(N*n_test,1).to(device, dtype=torch.float)
      Z_med = 0.5*torch.ones(n_test,1).to(device, dtype=torch.float)
      
      Xb = torch.cat([Xt], dim=0).to(device, dtype=torch.float)
      yb = torch.cat([yt], dim=0).to(device, dtype=torch.float)
      alpha_med = 0.5*torch.ones(N,1).to(device, dtype=torch.float)
           
      Generated = out1 = out2 = out3 = Out_med = torch.zeros(N, dim_s)
      Out = torch.zeros(N, n_test, 2)
      
      z = torch.rand(n_test,1).to(device, dtype=torch.float)#*(0.1-0.9)+0.9
      
      s1 = torch.zeros(n_test, dim_s)
      s1[:,0] = 0.0;s1[:,1] = 1.0
      s1 = s1 / torch.sqrt(torch.sum(s1**2, dim=1)).reshape(n_test,1)
      out1 = G(Xb, z, s1)
      
      s2 = torch.zeros(n_test, dim_s)
      s2[:,0] = 1.0;s2[:,1] = 0.0
      s2 = s2 / torch.sqrt(torch.sum(s2**2, dim=1)).reshape(n_test,1)
      out2 = G(Xb, z, s2)
      
      s3 = torch.ones(n_test, dim_s)
      s3 = s3 / torch.sqrt(torch.sum(s3**2, dim=1)).reshape(n_test,1)
      out3 = G(Xb, z, s3)
      
#      print(out2.shape)
#      for i in range(N):
#        z = torch.rand(n_test,1).to(device, dtype=torch.float)*(0.1-0.9)+0.9
#        s0 = torch.randn(n_test, dim_s)
#        s0 = s0 / torch.sqrt(torch.sum(s0**2, dim=1)).reshape(n_test,1)
        
#        out_s = G(Xb, z, s0)[0]
#        Out[i,:,:] = out_s

        
      Out = Out.detach().numpy()   
      #Out_med = out.cpu()
      Generated = Generated.detach().numpy()    
      Out_med = Out_med.detach().numpy()  
      out1 = out1.detach().cpu().numpy()
      out2 = out2.detach().cpu().numpy()
      out3 = out3.detach().cpu().numpy()
      s1 = s1.detach().cpu().numpy()
      s2 = s2.detach().cpu().numpy()
      #lam_cand = lam_cand.detach().numpy()   
      #Out_discr = 0.0
    return Out, Out_med, Generated, out1, out2, out3, s2
        
    
