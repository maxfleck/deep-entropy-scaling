import torch
import torch.nn as nn

class BNN(nn.Module):
    """
        Implementation of a Basic Neural Network
    """
    def __init__(self, layer_units, 
                 activation=torch.nn.ReLU(),
                 batch_norm:bool=False,):    
        """
            Creates the BON using the following parameters

            Build layer_units before passing:
            inner_layer_units = depth*[width]
            layer_units = [n] + inner_layer_units
            layer_units += [p]            

            Parameters:
            layer_units (list,int) : nodes per layer
            activation      : the activation function to be used
            batch_norm(bool): if True use batch normalization
        """
        super(BNN, self).__init__()

        net_modules = []
        Layers = layer_units
        L = len(Layers)
        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            # print("linear io", input_size, output_size)
            layer = torch.nn.Linear(input_size, output_size)
            net_modules.append(layer)
            if i < L-2:
                if batch_norm:
                    # print("batch_norm size", output_size)
                    layer = torch.nn.LayerNorm(output_size)
                    net_modules.append(layer)
                # print("relu")
                net_modules.append(activation)
        self.bnn = nn.Sequential(*net_modules)
        return

    def forward(self,x):
        return self.bnn(x)


class GenDeepONet(torch.nn.Module):

    def __init__(self, n_branch:int=12, n_trunk:int=12,
                 d_branch:int=1, d_trunk:int=1,
                 w_branch:int=32, w_trunk:int=32,
                 p:int=24,
                 activation=torch.nn.ReLU(),
                 pass_p_branch:bool=False,
                 apply_bias:bool=False,
                ):
        """
            Creates the GenDON using the following parameters

            Parameters:
            n_branch (int  : the input size of the branch network
            n_trunk  (int) : the input size of the trunk network
            d_branch(int)  : number of layers in branch network 
            d_trunk(int)   : number of layers in trunk network 
            w_branch (int) : number of nodes at branch layer
            w_trunk (int)  : number of nodes at trunk layer
            p        (int) : output dimension of networks (to be mapped)
            activation     : the activation function to be used
            pass_p_branch(bool): if True branch output is returned
        """        
        super(GenDeepONet, self).__init__()
    
        p_branch = p + n_trunk
        lu_branch = [n_branch] + d_branch*[w_branch] + [p_branch]
        lu_trunk = [n_trunk] + d_trunk*[w_trunk] + [p]

        self.p = p
        self.p_branch = p_branch
        self.n_branch = n_branch
        self.n_trunk = n_trunk
        self.pass_p_branch = pass_p_branch
    
        self.branch_net = BNN(lu_branch)
        self.trunk_net = BNN(lu_trunk)

        if apply_bias:
            self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)
        self.apply_bias = apply_bias
        return

    def forward(self,x):
        x_branch, x_trunk = torch.split(x, [self.n_branch, self.n_trunk], 1)
        x_branch = self.branch_net(x_branch)
        x_branch, xi_trunk = torch.split(x_branch, [self.p, self.n_trunk], 1)
        x_trunk = self.trunk_net(x_trunk/xi_trunk)
        #y_out = x_branch @ x_trunk
        y_out = torch.sum(x_branch * x_trunk, dim=1)
        if self.apply_bias:
            y_out = y_out + self.bias
        if not self.pass_p_branch:
            return y_out
        else:
            return y_out, x_branch


class DeepESNet(torch.nn.Module):

    def __init__(self, n_branch:int=12,
                 en_n_trunk:int=1,
                 en_d_branch:int=1, en_d_trunk:int=1,
                 en_w_branch:int=32, en_w_trunk:int=32,
                 en_p:int=24,
                 ref_n_trunk:int=1,
                 ref_d_branch:int=1, ref_d_trunk:int=1,
                 ref_w_branch:int=32, ref_w_trunk:int=32,
                 ref_p:int=24,                 
                 activation=torch.nn.ReLU(),
                 pass_p_branch:bool=True,
                ):
        """
            Creates the DeepESNet using the following parameters

            Parameters:
            n_branch (int  : the input size of the networks
            
            en_n_trunk  (int) : the input size of the trunk network
            en_d_branch(int)  : number of layers in branch network 
            en_d_trunk(int)   : number of layers in trunk network 
            en_w_branch (int) : number of nodes at branch layer
            en_w_trunk (int)  : number of nodes at trunk layer
            en_p        (int) : output dimension of networks (to be mapped)

            ref_n_trunk  (int) : the input size of the trunk network
            ref_d_branch(int)  : number of layers in branch network 
            ref_d_trunk(int)   : number of layers in trunk network 
            ref_w_branch (int) : number of nodes at branch layer
            ref_w_trunk (int)  : number of nodes at trunk layer
            ref_p        (int) : output dimension of networks (to be mapped)            
        """        
        super(DeepESNet, self).__init__()
        self.pass_p_branch = pass_p_branch
        self.n_branch = n_branch
        if pass_p_branch:
            en_n_branch = n_branch + en_p 
        else:
            en_n_branch = n_branch
        self.en_n_branch = en_n_branch
        self.en_n_trunk = en_n_trunk
        self.ref_n_trunk = ref_n_trunk

        self.deepES_star = GenDeepONet(n_branch,ref_n_trunk,
                             ref_d_branch,ref_d_trunk,
                             ref_w_branch,ref_w_trunk,
                             ref_p, pass_p_branch=pass_p_branch
                            )        
        
        
        self.deepES = GenDeepONet(en_n_branch,en_n_trunk,
                             en_d_branch,en_d_trunk,
                             en_w_branch,en_w_trunk,
                             en_p
                            )
        return

    def forward_ref(self,x):
        dummy = torch.split(x, [self.ref_n_trunk, self.en_n_trunk, self.n_branch], 1)
        temperature, entropy, embedding = dummy
        
        x_ref = torch.cat([embedding, temperature], 1)
        x_ref, _ = self.deepES_star(x_ref)
        return  torch.unsqueeze(x_ref, 1)
    
    def forward(self,x):
        dummy = torch.split(x, [self.ref_n_trunk, self.en_n_trunk, self.n_branch], 1)
        temperature, entropy, embedding = dummy

        x_ref = torch.cat([embedding, temperature], 1)
        x_ref, x_branch = self.deepES_star(x_ref)
        
        x_en = torch.cat([embedding, x_branch, entropy], 1)
        x_en = self.deepES(x_en)
        return  torch.unsqueeze(x_en + x_ref, 1)
