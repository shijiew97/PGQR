import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample

def GBR2(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
    num_it1, lr1, lrdecay1, lr_power1, m1, verb1, boot_size1, test1, l1pen1, N1, K1):
    
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
    
    if torch.is_tensor(X) == False: X = torch.from_numpy(X)
    if torch.is_tensor(y) == False: y = torch.from_numpy(y)
    if torch.is_tensor(Xt) == False: Xt = torch.from_numpy(Xt)
    
    X = X.to(device, dtype = torch.float)
    y = y.to(device, dtype = torch.float)
    Xt = Xt.to(device, dtype = torch.float)
    
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
    
    class Net(nn.Module):
      def __init__(self, S,  hidden_size, L, batchnorm_on, p, n, zn):
        super(Net, self).__init__()
        input_dim = 10*p + zn
        output_dim = k
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn0 = nn.BatchNorm1d(input_dim)
        #self.bn1 = nn.BatchNorm2d(input_dim)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          #self.layers.append( nn.LayerNorm(hidden_size) )
          self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 

      def forward(self, x, z, batchnorm_on):
        x_m = torch.cat(10*[0.1*x], dim=1)
        out1 = torch.cat([x_m, z], dim=1)
        out = self.bn0(out1)
        out = self.relu(self.fc0(out))
        for j in range(L-1):
          out = self.layers[3*j](out)
          out = self.layers[3*j+1](out)
          out = self.layers[3*j+2](out)
        out = self.fc_out(out)
        return out      

    def Diff(y1,y2):
        out = torch.linalg.norm(y1-y2, dim=1, ord=1)#ord=2
        return out

     
    #Generator initilization
    G = Net(S, hidden_size, L, batchnorm_on, p, n, zn).to(device)
    optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    LOSS = torch.zeros(iteration)
    
    X = X.view(n,p)
    y = y.view(n,1)
    Xm = torch.cat(N*m*[X]).reshape(N,m,n,p).to(device, dtype=torch.float)
    loss0 = 0.0
    Generator_start = time.perf_counter()
    lr1 = lr
    K0 = 3
    n0 = 500
    
    for it in range(iteration):
        
        lr1 = lr/(float(it+1.0)**lrPower) 
        for param_group in optimizer.param_groups:
           param_group["lr"] = lr1
           
        loss = torch.zeros(1).to(device)
        for h in range(K0):
            
            ind = sample(range(n), n0)
            
            X0 = X[ind,:].reshape(n0,p)
            y0 = y[ind,:].reshape(n0,1)
            
            Z = torch.rand(n0,zn).to(device, dtype=torch.float)
            ZZ = torch.rand(n0,zn).to(device, dtype=torch.float)
            

            Out_G1 = G(X0, Z, batchnorm_on)
            #Out_G1 = Sample(Out_G1)
            #print(Out_G1.shape)
        
            dist = Diff(y0, Out_G1)
            Out1 = 2*torch.mean(dist)
        
            Out_ZZ = G(X0, ZZ, batchnorm_on)
            #Out_ZZ = Sample(Out_ZZ)
            
            Out2 = -torch.mean(Diff(Out_ZZ, Out_G1))
            
            loss += Out1 + Out2
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        loss0 += loss.item()/50
        LOSS[it] = loss.item()
        
        if (it+1)%50==0 and verb == 1:
            percent = float((it+1)*100) /iteration
            arrow   = '-' * int(percent/100 *20 -1) + '>'
            spaces  = ' ' * (20-len(arrow))
            train_time = time.perf_counter() - Generator_start
            print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
            " Current/Initial Loss: {:.4f}/{:.4f}, Loss1: {:.4f}, Learning rate: {:.5f}, Training time: {:.1f}"
            .format(float(loss0), LOSS[0], float(torch.mean(Out1).item()), lr1, train_time,), end='')
            loss0 = 0.0
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
    #print(n_test)
    Z = torch.rand(1000*n_test,zn).to(device, dtype=torch.float)
    Xb = torch.cat(1000*[Xt], dim=0).to(device, dtype=torch.float)
    #print(Xb.size())
    #print(Z.size())
    #print(X.size())
    #print(ZZ.size())
    
    #print(Xt.size())
    #print(X.size())
    
    G.eval()
    Out = G(Xb, Z, batchnorm_on).reshape(1000,n_test)
    print(Out.size())
    sys.stdout.flush()
    Out = Out.cpu().detach().numpy()                
    
    Out_discr = 0.0
    lam_cand0 = 0.0
    
    return Out, Out_discr, lam_cand0
        
    
