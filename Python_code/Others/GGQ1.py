import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys
import math

def GGQ1(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
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
    
    #dim_s = y.shape[1]
    #print(dim_s)
    #print(p)
        
    class GNet(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn):
        super(GNet, self).__init__()
        input_dim = p + 1 
        #output_dim = dim_s
        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        
        self.fc0 = nn.Linear(input_dim, hidden_size)
        #self.fc_out = nn.Linear(hidden_size, dim_s)
        
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn_x = nn.BatchNorm1d(p)
        self.fc_out = nn.Linear(hidden_size, 1)
        
        #self.fc_outs1 = nn.Linear(dim_s, hidden_size)
        #self.fc_outs = nn.Linear(dim_s, 1, bias=False)
        #self.fc_outs = nn.Linear(hidden_size, 1)
        #self.fc_outs = nn.Linear(hidden_size, 1, bias=False)
        
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          self.layers.append( nn.ReLU() ) 
          #self.layers.append( nn.Linear(1, hidden_size) )
          #self.layers.append( nn.Linear(dim_s, hidden_size) )

      def forward(self, x, a):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x, a], dim=1)
        out = self.relu(self.fc0(out))
        u = 3
        for j in range(L-1):
          out = self.layers[u*j](out)
          out = self.layers[u*j+1](out)#+self.layers[u*j+3](a)#+self.layers[u*j+4](s)
          out = self.layers[u*j+2](out) 
          
        out = self.fc_out(out)
        #out = self.fc_outs(out)
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
     
    class GNet2(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn):
        super(GNet2, self).__init__()
        input_dim = p + 1

        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.Tanh()#nn.ReLU()#nn.Tanh()#nn.ReLU()#nn.Tanh()#nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)
        
        self.fc001 = nn.Linear(p+2, hidden_size)
        self.fc002 = nn.Linear(p+2, hidden_size)
        self.fc003 = nn.Linear(p+2, hidden_size)
        
        self.fc0001 = nn.Linear(p+1, hidden_size)
        self.fc0002 = nn.Linear(p+1, hidden_size)
        self.fc0003 = nn.Linear(p+1, hidden_size)
        
        self.fc01 = nn.Linear(1, hidden_size)
        self.fc02 = nn.Linear(1, hidden_size)
        
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

      def forward(self, x, a, lam, batchnorm_on):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x, lam], dim=1)
        out00 = torch.cat([x, a, lam], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #out = self.bn0(out)
        out1 = self.relu( self.fc0(out) )
        out1 = self.relu( self.fc1(out1 + 0.8*self.fc0001(out) )) # + 0.1*self.fc0001(out) )
        out1 = self.relu( self.fc2(out1+ 0.8*self.fc0002(out)) )#+ 0.1*self.fc0002(out)) #+ 0.3*self.fc000(out) )
        #u = 3
        #for j in range(L-1):
        #  out = self.layers[u*j](out)
        #  out = self.layers[u*j+1](out)# + self.layers[u*j+3](a)
        #  out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        
        out_z = self.relu( self.z_0(a) )
        out_z = self.relu( self.z_1(out_z + 0.8*self.fc01(a))) #+ 0.1*self.fc01(a))
        out_z = self.relu( self.z_2(out_z + 0.8*self.fc02(a))) #+ 0.1*self.fc02(a))
        
        #out = torch.cat([out1, out_z], dim=1)
        out = out1 + out_z
        #out = self.fc_out1(out + 1.0*self.fc001(out00)) #+ 0.2*self.fc00(out00)
        #out = self.fc_out2( self.relu(out + 0.1*self.fc002(out00)) )
        #out = self.z_out( self.relu(out + 1.0*self.fc003(out00)) )
        
        out = self.fc_out1(out) #+ 0.2*self.fc00(out00)
        out = self.fc_out2( self.relu(out ) )
        out = self.z_out( self.relu(out) )
        #out = torch.sort(out)[0]
        
        return out 
    
    class GNet0(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn):
        super(GNet0, self).__init__()
        input_dim = p+1

        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.Tanh()#nn.ReLU()#nn.Tanh()#nn.ReLU()#nn.Tanh()#nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)
        
        #self.fc001 = nn.Linear(p+2, hidden_size)
        #self.fc002 = nn.Linear(p+2, hidden_size)
        
        #self.fc0001 = nn.Linear(p+1, hidden_size)
        #self.fc0002 = nn.Linear(p+1, hidden_size)
        #self.fc0003 = nn.Linear(p+1, hidden_size)
        
        #self.fc01 = nn.Linear(1, hidden_size)
        #self.fc02 = nn.Linear(1, hidden_size)
        
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

      def forward(self, x, a, lam, batchnorm_on):
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x, lam], dim=1)
        #out00 = torch.cat([x, a, lam], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #out = self.bn0(out)
        out1 = self.relu( self.fc0(out) )
        out1 = self.relu( self.fc1(out1) ) # + 0.1*self.fc0001(out) )
        out1 = self.relu( self.fc2(out1) ) #+ 0.1*self.fc0002(out)) #+ 0.3*self.fc000(out) )
        #u = 3
        #for j in range(L-1):
        #  out = self.layers[u*j](out)
        #  out = self.layers[u*j+1](out)# + self.layers[u*j+3](a)
        #  out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        
        out_z = self.relu( self.z_0(a) )
        out_z = self.relu( self.z_1(out_z) ) #+ 0.1*self.fc01(a))
        out_z = self.relu( self.z_2(out_z) ) #+ 0.1*self.fc02(a))
        
        #out = torch.cat([out1, out_z], dim=1)
        out = out1 + out_z
        out = self.fc_out1(out) #+ 0.2*self.fc00(out00)
        #out = self.fc_out2( self.relu(out + 0.8*self.fc002(out00)) )
        out = self.z_out( self.relu(out) )
        #out = torch.sort(out)[0]
        
        return out 
    
    def Loss(y, yhat, u):
        resid = y-yhat
        out = (torch.abs(resid) + resid*u) * (1)
        return out.mean()

        
    #Generator initilization
    G = GNet0(S, hidden_size, L, batchnorm_on, p, n, zn).to(device)
    optimG= torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimG= torch.optim.SGD(G.parameters(), lr=lr, momentum=0.8)
    LOSS = torch.zeros(iteration)
    sd_y = torch.std(y).item()
    X = X.view(n,p)
    y = y.view(n,1)
    #print("terawe")
    #sys.stdout.flush()
    
    #Xm = torch.cat(N*m*[X]).reshape(N,m,n,p).to(device, dtype=torch.float)
    loss0 = 0.0
    pen0 = 0.0
    lam0 = 0.0
    Generator_start = time.perf_counter()
    lr1 = lr
    n_lam = 100
    const = 1.0
    #lam_cand = torch.linspace(lam_min, lam_max,n_lam)
    print("Training runs!")
    #Beta = torch.distributions.beta.Beta(0.5*torch.ones(1,1), 0.5*torch.ones(1,1))
    sys.stdout.flush()
    #print("as111d")
    #sys.stdout.flush()
    #alpha05 = 0.5*torch.ones(n0,1).to(device)
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


            G.zero_grad()
            #alpha = torch.rand(n0,1)
            #ind = sample(range(n0), int(n0/20))
            #alpha[ind] = 0.5
            #alpha = torch.ones(n0,1)*torch.rand(1)
            
            #u = torch.randn(1, dim_s)
            #cord = torch.randn(n0, dim_s)*(math.pi);u=torch.ones(n0,dim_s)
            #cord = torch.randn(1, dim_s)*(math.pi);u=torch.ones(n0,dim_s)
            #u[:,0] = torch.randn(1)*torch.sin(cord[:,0]);u[:,1] = torch.randn(1)*torch.cos(cord[:,1])
            #u = torch.randn(n0, dim_s)
            #u = u / torch.sqrt(torch.sum(u**2, dim=1)).reshape(n0,1)
            #u = Open_ball(n0, dim_s, const)
            u = 2*torch.rand(n0, 1)-1
            #u = torch.randn(n0, 1)
            u = torch.ones(n0, 1)*u
            u = u.reshape(n0,1).to(device, dtype=torch.float)
            
            
            Out_G = G(X0, u)
            #print(Out_G.shape)
            #loss = Loss( torch.sum(y0*s_vec, dim=1).reshape(n0,1) - Out_G, alpha)
            loss = Loss(y0, Out_G, u)
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
            .format(float(loss0), LOSS[0], -float(pen0), lr1, n, 1, train_time,), end='')
            loss0 = 0.0
            pen0 = 0.0
            lam0 = 0.0
            sys.stdout.flush()
            
    n_test = Xt.size()[0]
    N = 500
    device = 'cpu'
    G.eval().to(device)
    
    with torch.no_grad():
        
      Xt = Xt.to(device, dtype=torch.float)
      #Z1 = torch.rand(N*n_test,1).to(device, dtype=torch.float)
      #Z_med = 0.5*torch.ones(n_test,1).to(device, dtype=torch.float)
      
      Xb = torch.cat([Xt], dim=0).to(device, dtype=torch.float)
      #yb = torch.cat(N*[yt], dim=0).to(device, dtype=torch.float)
      #alpha_med = 0.5*torch.ones(N,1).to(device, dtype=torch.float)
           
      Generated = out1 = out2 = out3 = Out_med = torch.zeros(1)
      Out = torch.zeros(N, n_test)
      
      for i in range(N):
        #u = torch.randn(1, dim_s)
        #cord = torch.randn(n_test, dim_s)*(math.pi);u=torch.ones(n_test,dim_s)
        #cord = torch.randn(1,dim_s)*(math.pi);u=torch.ones(1,dim_s)
        #u[:,0] = torch.randn(1)*torch.sin(cord[:,0]);u[:,1] = torch.randn(1)*torch.cos(cord[:,1])
        #u = torch.randn(n_test, dim_s)
        #u = u / torch.sqrt(torch.sum(u**2, dim=1)).reshape(n_test,1)
        #u = torch.ones(n_test, dim_s)*u
        #const0 = 1.0
        #u = Open_ball(n_test, dim_s, const)
        u = 2*torch.rand(n_test, 1)-1
        #u = torch.randn(n_test)
        u = torch.ones(n_test, 1)*u
        u = u.reshape(n_test,1).to(device, dtype=torch.float)
        
        out_s = G(Xb, u)
        Out[i,:] = out_s.reshape(-1)

        
      Out = Out.detach().numpy()   
      #Out_med = out.cpu()
      Generated = Generated.detach().numpy()    
      Out_med = Out_med.detach().numpy()  

      #lam_cand = lam_cand.detach().numpy()   
      #Out_discr = 0.0
    return Out, Out_med, Generated
        
    
