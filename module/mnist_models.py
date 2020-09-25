import os, sys
import copy as copyroot
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from fastai2.basics import *
from fastai2.vision.all import *


# class BaseNet(torch.nn.Module):
#     def __init__(self, D_in=28, H=28):
#         super(BaseNet, self).__init__()
#         self.flat    = nn.Flatten()
#         self.linear1 = nn.Linear(in_features=D_in**2, out_features=H)
#         self.relu1   = nn.
#         self.linear2 = nn.Linear(in_features=H, out_features=2, bias=False)
#         self.sig     = SigmoidRange(-1., 1)
    
#     def forward(self, x):
#         l0 = self.flat(x)
#         l1 = self.linear1(l0)
#         l2 = self.linear2(l1)
#         y =  self.sig(l2)
#         return y


class FeatsNet(torch.nn.Module):
    
    def __init__(self, D_in=28, H=28, feats=['pix']):
        
        self.feats = feats
        self.D = D_in
        
        super(FeatsNet, self).__init__()
        
        len_ins = 0
        if 'pix'      in self.feats: len_ins += D_in**2
        if 'max'      in self.feats: len_ins += D_in*2
        if 'min'      in self.feats: len_ins += D_in*2
        if 'pts11'    in self.feats: len_ins += 4*2
        if 'pts12'    in self.feats: len_ins += 16*2
        if 'pts22'    in self.feats: len_ins += 16*2
            
        if len_ins == 0: 
            raise Exception('Need a valid code in `feats`')
            
        self.flat    = nn.Flatten()
        
        self.linear1 = nn.Linear(in_features=len_ins, out_features=H)
        self.linear2 = nn.Linear(in_features=H,       out_features=2, 
                                 bias=False)
        
        self.sig     = SigmoidRange(-1., 1)

    def build_pt_feats(self, x, combo='12'):
    
        xi = x.squeeze(0)

        nz   = xi.nonzero()

        mins = nz.min(0)
        maxs = nz.max(0)

        mm    = torch.stack((mins.values, maxs.values))

        mm2 = torch.stack((
        torch.cat((
            torch.tensor([nz[nz[:,1] == e].max(0).values[0] for e in mm[:,1]]),
            torch.tensor([nz[nz[:,1] == e].min(0).values[0] for e in mm[:,1]])
        )),
        torch.cat((
            torch.tensor([nz[nz[:,0] == e].max(0).values[1] for e in mm[:,0]]),
            torch.tensor([nz[nz[:,0] == e].min(0).values[1] for e in mm[:,0]])
        ))
        ))
        mm2 = mm2.T
        
        if combo == '11':
            pts11 = torch.cartesian_prod(mm[:,0], mm[:,1])
            return pts11
            
        if combo == '12':
            pts12 = torch.cat((
                torch.cartesian_prod(mm[:,0],  mm2[:,1]),
                torch.cartesian_prod(mm2[:,0], mm[:,1])
            ))
            return pts12
        
        if combo == '22':
            pts22 = torch.cartesian_prod(mm2[:,0], mm2[:,1])        
            return pts22
        
    def build_feats(self, x):
        
        l0 = torch.tensor([])
        
        if 'pix' in self.feats:
            l0 = self.flat(x)
        
        if 'max' in self.feats:
            max2  = self.flat(x.argmax(dim=2).float()) / float(self.D)
            max3  = self.flat(x.argmax(dim=3).float()) / float(self.D)
            l0 = torch.cat((l0, max2, max3), dim=1)
            
        if 'min' in self.feats:    
            min2  = self.flat(x.argmin(dim=2).float()) / float(self.D)
            min3  = self.flat(x.argmin(dim=3).float()) / float(self.D)
            l0 = torch.cat((l0, min2, min3), dim=1)
            
        if 'pts11' in self.feats:
            tmp = torch.tensor([self.build_pt_feats(e, combo='11').tolist() 
                                for e in x])
            pts = self.flat(tmp.float()) / float(self.D)
            l0 = torch.cat((l0, pts), dim=1)
        
        if 'pts12' in self.feats:
            tmp = torch.tensor([self.build_pt_feats(e, combo='12').tolist() 
                                for e in x])
            pts = self.flat(tmp.float()) / float(self.D)
            l0 = torch.cat((l0, pts), dim=1)
            
        if 'pts22' in self.feats:
            tmp = torch.tensor([self.build_pt_feats(e, combo='22').tolist() 
                                for e in x])
            pts = self.flat(tmp.float()) / float(self.D)
            l0 = torch.cat((l0, pts), dim=1)
            
        return l0
    
    def forward(self, x):
        
        l0 = self.build_feats(x)
        
        l1 = self.linear1(l0)
        l2 = self.linear2(l1)
        y =  self.sig(l2)
        return y


