import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import random
import sys

def QR_pen_w(y, X, Xt, hidden_size1, gpu_ind1, NN_type1, L1, S1, n1, zn1, p1, batchnorm_on1,
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
    #Xt = Xt.to(device, dtype = torch.float)
    #yt = yt.to(device, dtype = torch.float)
    
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
    fac = 10.0#float(fac)
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
        input_dim = p + 2 + 1 + 1
        output_dim = k
        #self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.fc0 = nn.Linear(input_dim, hidden_size)
        self.bn_out = nn.BatchNorm1d(hidden_size+n0)
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.bn_x = nn.BatchNorm1d(p)
        #self.bn1 = nn.BatchNorm2d(input_dim)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.fc_out1 = nn.Linear(hidden_size, hidden_size)
        self.fc_out2 = nn.Linear(hidden_size, n0)
        self.layers = nn.ModuleList()
        #weight w
        #self.fc1_w = nn.Linear(hidden_size, hidden_size)
        #self.fc2_w = nn.Linear(hidden_size, n0)
        #self.fc3_w = nn.Linear(n0, 1)
        self.fc_out_final = nn.Linear(hidden_size+n0, 1)
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 
          self.layers.append( nn.Linear(1, hidden_size) )
          #self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.Linear(1, hidden_size, bias=False) )

      def forward(self, x, w, a, lam, batchnorm_on):
        out_w = w#torch.exp(-w)
        #x = self.bn_x(x)
        #x_m = x #torch.cat([3.0*x], dim=1)
        #lam_m = torch.cat(2*[0.5*lam, 2.0*torch.sign(lam)*torch.log(torch.abs(lam)+1.0)], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        trans_a = - torch.sign(a-0.5)*( torch.log( torch.abs(0.5-a) + 0.00001) + 0.6931472 ) # -log(1/2) = -0.6931472
        out = torch.cat([x, out_w, trans_a, 5.0*a, lam], dim=1)
        #out = torch.cat([x, a, -a, lam, -lam], dim=1)
        #out = self.bn0(out)
        out = self.relu(self.fc0(out))
        u = 4
        for j in range(L-1):
          out = self.layers[u*j](out)
          out = self.layers[u*j+1](out) + self.layers[u*j+3](a)
          out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        
        #out = self.fc_out(out)
        out1 = self.fc_out1(out)
        out2 = self.fc_out2(self.relu( (out) ))*out_w
        out = torch.cat([out1, out2], dim=1)
        out = self.bn_out(out)
        out = self.fc_out_final(out)
        #out = out1 + self.fc3_w(out2)
        return out 
      
    def Loss(x, alpha, w):
        relu = nn.ReLU()
        out = alpha * relu(x) + (1.0-alpha) * relu(-x)
        out = w*out
        return out.mean()
    
    #Generator initilization
    G = GNet(S, hidden_size, L, batchnorm_on, p, n, zn).to(device)
    optimG= torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    #optimG= torch.optim.SGD(G.parameters(), lr=lr, momentum=0.8)
    
    LOSS = torch.zeros(iteration)
    sd_y = torch.std(y).item()
    X = X.view(n,p)
    y = y.view(n,1)
    #Xm = torch.cat(N*m*[X]).reshape(N,m,n,p).to(device, dtype=torch.float)
    loss0 = 0.0
    pen0 = 0.0
    lam0 = 0.0
    Generator_start = time.perf_counter()
    lr1 = lr
    #lam_cand = torch.linspace(lam_min, lam_max,n_lam)
    print("Training runs!")
    Beta = torch.distributions.beta.Beta(0.5*torch.ones(1,1), 0.5*torch.ones(1,1))
    sys.stdout.flush()
    #print("as111d")
    #sys.stdout.flush()
    alpha05 = 0.5*torch.ones(n0,1).to(device)
    K0 = int(n/n0)   
    
    cv_k = int(10)
    N = 1000
    n_w = 100
    n_lam = 100
    lam_cand = torch.linspace(lam_min, lam_max, n_lam)
    
    index = np.arange(n0)
    np.random.shuffle(index)
    ind_split = np.split(index, cv_k)
    
    #ind = sample(range(n), n0)
    #ind = np.asarray(random.sample(range(n), n0)).reshape(-1)
    X0 = X.to(device, dtype=torch.float)#[ind,:].to(device, dtype=torch.float)
    y0 = y.to(device, dtype=torch.float)#[ind,:].reshape(n0,1).to(device, dtype=torch.float)
    #print(len(ind))
    
    for it in range(iteration):
        lr1 = lr/(float(it+1.0)**lrPower) 
        for param_group in optimG.param_groups:
           #sys.stdout = open(os.devnull, 'w')
           param_group["lr"] = lr1
        #ind = sample(range(n), n0)
        #X0 = X[ind,:].to(device, dtype=torch.float)
        #y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
        
        #index = np.arange(n)
        #np.random.shuffle(index)
        #ind_split = np.split(index, K0)
        
        for h in range(K0):
            #ind = ind_split[h]#
            #ind = sample(range(n), n0)
            #X0 = X[ind,:].to(device, dtype=torch.float)
            #y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
            
            Ws = torch.from_numpy(np.ones((n0,1)).reshape(n0,1)).to(device, dtype=torch.float)
            #Ws = torch.from_numpy(np.random.exponential(scale=1, size=n0).reshape(n0,1)).to(device, dtype=torch.float)
            ind0 = ind_split[np.random.randint(0, cv_k)]#
            Ws[ind0,:] = 0.0
            Ws = Ws.to(device)
            
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
            ind1 = sample(range(n0), int(n0/10))
            alpha[ind1] = 0.5
            alpha = alpha.to(device)
            Out_G = G(X0, Ws, alpha, lam, batchnorm_on)
            loss = Loss( y0 - Out_G, alpha, Ws)
        
        #Out_G1 = G(X0, alpha05, lam, batchnorm_on)
        #loss += 0.3*Loss( y0 - Out_G1, alpha05 )
        
            pen = torch.zeros(1)
            if pen_on == 1:
              #a1 = torch.rand(n0,1).to(device, dtype=torch.float)
               Ws2 = torch.from_numpy(np.random.exponential(scale=1, size=S).reshape(S,1)).to(device, dtype=torch.float)
               a2 = torch.rand(n0,1).to(device, dtype=torch.float)
          #Out_G1 = G(X0, alpha, lam, batchnorm_on)
               Out_G2 = G(X0, Ws2, a2, lam, batchnorm_on)
               dist = torch.abs(Out_G - Out_G2)
          #pen = dist 
               pen = torch.log(  ( dist  / sd_y )*fac  + 1.0 / fac ) 
               lam_exp = torch.exp(lam)
               pp  = (lam_exp * pen).mean()
               loss -= pp
               pen0 += pp.item()/100
            loss.backward()
            optimG.step()
        
            loss0 += loss.item()/100
            lam0 += lam11/100
        LOSS[it] = loss.item()
        
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
    #device = 'cpu'
    G.eval()#.to(device)
    with torch.no_grad():
      #Xt = Xt.to(device, dtype=torch.float)
      #Z_a = 0.5*torch.ones(n_test,1).to(device, dtype=torch.float)

      #Xb = torch.cat(N*[Xt], dim=0).to(device, dtype=torch.float)
      #yb = torch.cat(N*[yt], dim=0).to(device, dtype=torch.float)
      

      #yb = torch.cat(N*[y], dim=0).to(device, dtype=torch.float)
      n_test = int(n0/cv_k)
      
      Out = np.zeros(1)
      #Out = np.zeros((n_lam,n_w,cv_k,N,n_test))
      Out_med = torch.zeros(1)
      #Out_discr = torch.zeros(n_lam)
      
      #index = np.arange(n0)
      #ind_split = np.split(index, cv_k)
      U_hat = np.zeros((n_w,cv_k,n_lam))#CR-statistics
      Q_hat = np.zeros((cv_k,n_w,n_lam,n_test))
      
      for k in range(cv_k):
        #lam = lam_cand[i]*torch.ones(n_test*N,1).to(device)
        ########################
        ind0 = ind_split[k]#
        yt = y0[ind0,:]
        
        #Partial data
        #Ws = torch.from_numpy(np.ones((N*n_test,1)).reshape(N*n_test,1)).to(device, dtype=torch.float)
        #Xb = torch.cat(N*[Xt], dim=0).to(device, dtype=torch.float)
        #Xt = X0[ind0,:]
        #Xt = Xt.to(device, dtype=torch.float)
        
        #full data        
        W = torch.ones(n0, 1)
        W[ind0,:] = 0.0
        Ws = W.to(device, dtype=torch.float)
        #Ws = torch.cat(N*[W], dim=0).to(device, dtype=torch.float)
        #Xb = torch.cat(N*[X0], dim=0).to(device, dtype=torch.float)
        
        for j in range(n_w):
            #Ws = torch.from_numpy(np.random.exponential(scale=1, size=N*n_test).reshape(N*n_test,1)).to(device, dtype=torch.float)
            #Ws = torch.from_numpy(np.random.exponential(scale=1, size=N*n_test).reshape(N*n_test,1)).to(device, dtype=torch.float)
            #Z1 = torch.rand(N*n_test,1).to(device, dtype=torch.float)
        
            
            ########################
            for i in range(n_lam):
            #for k in range(cv_k):
                #partial data
                #lam = lam_cand[i]*torch.ones(n_test*N,1).to(device)
                #Z1 = torch.rand(N*n_test,1).to(device, dtype=torch.float)
                
                #full data
                #lam = lam_cand[i]*torch.ones(n0*N,1).to(device)
                #Z1 = torch.rand(N*n0,1).to(device, dtype=torch.float)
                
                lam = lam_cand[i]*torch.ones(n0,1).to(device)
                #Z1 = torch.rand(n0,1).to(device, dtype=torch.float)

                #partial
                #out = G(Xb, Ws, Z1, lam, batchnorm_on).reshape(N,n_test).cpu().detach().numpy()   
                
                #out = out.reshape(N,n0)
                #out = out[N,ind0]
                #out = out.reshape(N,n_test).cpu().detach().numpy()   
                #Out[i,j,k,:,:] = out#.cpu
                out = np.zeros((N,n_test))
                for cc in range(N):
                    Z1 = torch.rand(n0,1).to(device, dtype=torch.float)
                    out[cc,:] = G(X0, Ws, Z1, lam, batchnorm_on).reshape(n0).cpu().detach().numpy()[ind0]   
                #full index
                #out = G(Xb, Ws, Z1, lam, batchnorm_on).reshape(N,n0).cpu().detach().numpy()[ind0,:]   

                #out = out[ind0,:]
                
                #U_hat = np.zeros((n_w,cv_k,n_lam))#CR-statistics
                #Q_hat = np.zeros((cv_k,n_w,n_lam,n_test))
                for c in range(n_test):
                    Q_hat[k,j,i,c] = np.sum(out[:,c] < yt[c,:].cpu().detach().numpy())/N
            
#            Ws_a = torch.from_numpy(np.random.exponential(scale=1, size=S).reshape(N*S,1)).to(device, dtype=torch.float)
#            lam_a = lam_cand[i]*torch.ones(n_test,1).to(device)
#            out = G(Xt, Ws_a, Z_a, lam_a, batchnorm_on)
#            Out_med[i,j,:] = out.reshape(n_test).cpu()
                U_hat[j, k, i] = np.mean(np.abs(np.sort(Q_hat[k, j, i, :]) - np.arange(1,n_test+1)/n_test))
      
      CR = np.mean(U_hat, 1)#shape (n_w, n_lam)
      opt_lam_ind = np.argmin(CR, 1)#shape (n_lam)
      opt_lambda = lam_cand[opt_lam_ind]
      #Out = Out.detach().numpy()    
      Out_med = Out_med.detach().numpy()    
      #out = out.cpu().detach().numpy()
      lam_cand = lam_cand.detach().numpy()   
      opt_lambda = opt_lambda.detach().numpy()
      Out_discr = 0.0
      
    return Out, Out_discr, lam_cand, CR, opt_lambda, ind_split#, U_hat, Q_hat, out
        
    
