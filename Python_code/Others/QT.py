import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys

def QT(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
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
    #print("7019100as111d")
    #sys.stdout.flush()
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
    q = p + 1
    #dat = torch.cat([y,X],dim=1)
        
    class GNet(nn.Module):
      def __init__(self, S,  hidden_size, L, n, zn, q):
        super(GNet, self).__init__()
        input_dim = q + 2 
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
          self.layers.append( nn.Linear(q, hidden_size) )

      def forward(self, a, s):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([trans_a, 5.0*a, s], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #out = self.bn0(out)
        out = self.relu(self.fc0(out))
        u = 5
        for j in range(L-1):
          out = self.layers[u*j](out)
          out = self.layers[u*j+1](out) + self.layers[u*j+3](a) + 0.2*self.layers[u*j+4](s)
          out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        out = self.fc_out(out)
        return out 
      
    def Loss(x, alpha):
        relu = nn.ReLU()
        out = alpha * relu(x) + (1.0-alpha) * relu(-x)
        return out.mean()
    
    #Generator initilization
    G = GNet(S, hidden_size, L, n, zn, q).to(device)
    optimG= torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimG= torch.optim.SGD(G.parameters(), lr=lr, momentum=0.8)
    LOSS = torch.zeros(iteration)
    sd_y = torch.std(y).item()
    X = X.view(n,p)
    y = y.view(n,1)
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
    m0 = torch.zeros(1,q).to(dtype=torch.float)
    m0[0,0] = torch.mean(y)
    m0[0,range(1,q)] = torch.mean(X,0).to(dtype=torch.float).reshape(q-1)
    m0 = m0.to(device)
    #print(m0)
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
            y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
            dat0 = torch.cat([y0, X0], dim=1)
            #s_samp = torch.randn(1,q).to(device)
            #s_samp = s_samp / torch.sqrt((s_samp**2).sum()).reshape(1,1)
            #s_samp = s_samp.repeat(n0,1)#torch.cat( n0 * [s_samp], dim = 0)
            #print(s_samp.size())
            #alpha = torch.rand(1,1)*torch.ones(n0,1)
            G.zero_grad()
            alpha = torch.rand(n0,1)
            ind = sample(range(n0), int(n0/20))
            alpha[ind] = 0.5
            s_samp = torch.randn(n0, q).to(device)
            s_samp = s_samp / torch.sqrt((s_samp**2).sum(1)).reshape(n0,1)
            #s_samp -= m0
            alpha = alpha.to(device)
            Out_G =  G(alpha, s_samp)# * s_samp ).sum(1).reshape(n0,1)
            loss = Loss( (dat0 * s_samp).sum(1).reshape(n0,1) - Out_G, alpha)
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
            " Current/Initial Loss: {:.4f}/{:.4f}, pen: {:.4f}, Learning rate: {:.5f}, n: {}, q:{}, Training time: {:.1f}"
            .format(float(loss0), LOSS[0], -float(pen0), lr1, n, q, train_time,), end='')
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
      Z1 = torch.rand(N * n_test,1).to(device, dtype=torch.float)
      Z_med = 0.5*torch.ones(n_test,1).to(device, dtype=torch.float)
      Xb = torch.cat(N*[Xt], dim=0).to(device, dtype=torch.float)
      yb = torch.cat(N*[yt], dim=0).to(device, dtype=torch.float)
      #print("as111d")
      #sys.stdout.flush()
      alpha_med = 0.5*torch.ones(N,1).to(device, dtype=torch.float)
#      s_samp1 = torch.randn(N, q).to(device, dtype=torch.float)
#      s_samp1 = s_samp1 / torch.linalg.norm(s_samp1, p=2, dim = 1)
#      s_samp1 -= m0
    
      s1 = torch.zeros(N,q).to(device)
      s1[:,0] = 1.0
      s1[:,1] = 0.0
      s1 = s1 / torch.sqrt((s1**2).sum(1)).reshape(N,1)
      print(s1[range(10),:])
      alpha1 = torch.rand(N,1).to(device, dtype=torch.float)
      out1 = G(alpha1, s1)

      s2 = torch.zeros(N,q).to(device)
      s2[:,0] = 0.0
      s2[:,1] = 1.0
      s2 = s2 / torch.sqrt((s2**2).sum(1)).reshape(N,1)
      print(s2[range(10),:])
      #alpha2 = torch.rand(N,1).to(device, dtype=torch.float)
      out2 = G(alpha1, s2)
           
      Generated = torch.zeros(N,q)
      Out = torch.zeros(N,n_test)
      #Out_med = torch.zeros(n_test)
      #Out_discr = torch.zeros(n_lam)
      for i in range(N):
        alpha_samp1 = torch.rand(n_test,1).to(device, dtype=torch.float)
        s_test = torch.randn(n_test, q)
        s_test = s_test / torch.sqrt((s_test**2).sum(1)).reshape(n_test,1)
        #s_test -= m0.cpu()
        A = s_test.t() @ s_test
        M = torch.inverse( A  ) @ s_test.t()
        s_test = s_test.to(device)
        M = M.to(device)
        out_s = G(alpha_samp1, s_test)
        gen_samp = M @ out_s
        Generated[i,:] = gen_samp.reshape(q).cpu()

        out = (out_s - torch.sum( Xt * s_test[:,range(1,q)], 1).reshape(n_test,1) ) 
        out /= s_test[:,0].reshape(n_test,1)
        Out[i,:] = out.reshape(n_test).cpu()
      
      out_s = G(Z_med, s_test)
      out = (out_s - torch.sum( Xt * s_test[:,range(1,q)], 1).reshape(n_test,1) ) 
      out /= s_test[:,0].reshape(n_test,1)
      Out_med = out.cpu()
      Generated = Generated.detach().numpy()    
      Out = Out.detach().numpy()    
      Out_med = Out_med.detach().numpy()  
      out1 = out1.detach().cpu().numpy()
      out2 = out2.detach().cpu().numpy()
      
      #lam_cand = lam_cand.detach().numpy()   
      #Out_discr = 0.0
    return Out, Out_med, Generated, out1, out2
        
    
