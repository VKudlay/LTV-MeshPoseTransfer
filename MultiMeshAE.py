import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm

from MeshAE import MeshAE

np.set_printoptions(precision=3)
        
class MultiTaskMeshAE(MeshAE):
    def __init__(self, nv1, nv2, bneck, decoder2_nv2, vs=[256, 64, 16], fs=[16, 64, 256], act_fn=nn.Tanh):
        super().__init__(nv1, nv2, bneck, vs, fs, act_fn)
        self.decoder2 = self.build_decoder(bneck, vs, fs, act_fn, nv2=decoder2_nv2)

    def forward(self, inputs, decoder=0):
        x = self.encoder(inputs.float())
        x = (self.decoder, self.decoder2)[decoder](x)
        return x.type(inputs.dtype)

    def batch_step(self, X, Y, training=False):
        tlen = sum(len(x) for x in X)
        loss = 0
        metrics = {}
        for i, (x, y) in enumerate(zip(X, Y)):
            p = self(x, i)
            loss += self.loss(p, y) * len(x)/tlen
            with torch.no_grad():
                metrics.update({f'{metric.__name__}{i+1}': metric(p, y).numpy().mean() for metric in self.metrics})
        return loss, metrics

class MeshVAE(MeshAE):
    def __init__(self, nv1, nv2, bneck, vs=[256, 64, 16], fs=[16, 64, 256], act_fn=nn.LeakyReLU):
        super().__init__(nv1, nv2, bneck, vs, fs, act_fn)
        
        self.fc_mu = nn.Linear(bneck, bneck)
        self.fc_logvar = nn.Linear(bneck, bneck)

    def encode(self, x):
        z = self.encoder(x)
        return self.fc_mu(z), self.fc_logvar(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, inputs, get_latents=False):
        mu, logvar = self.encode(inputs.float())
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z).type(inputs.dtype)
        return (x_recon, mu, logvar) if get_latents else x_recon

    def batch_step(self, x, y, training=False):
        p, mu, logvar = self(x, get_latents=True)
        loss = self.loss(p, y)
        with torch.no_grad():
            metrics = {metric.__name__ : metric(p, y).numpy().mean() for metric in self.metrics}
        if mu is not None and logvar is not None:
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld /= x.shape[0] * x.shape[1]
            loss += kld
        
        return loss, metrics


class MeshUNet(MeshAE):

    def __init__(self, nv1, nv2, bneck, vs=[256, 64, 16], fs=[16, 64, 256], act_fn=nn.LeakyReLU, skip_idx=[1], dropout=0.1):
        self.skip_idx = skip_idx
        super().__init__(nv1, nv2, bneck, vs, fs, act_fn)
        self.drop = nn.Dropout(dropout)
        self.acts = [lambda x: x] # [torch.sin, torch.cos]
    
    def build_decoder(self, bneck, vs, fs, act_fn):
        tblocks = [TConvBlock((v1,v2), (f1*(1+((len(vs)-1-i) in self.skip_idx)),f2), act_fn=act_fn) 
            for i,(v1,v2,f1,f2) in enumerate(zip(vs[::-1], vs[::-1][1:] + [self.nv2], fs[::-1][:-1], fs[::-1][1:]))]
        return nn.Sequential(
            nn.Linear(bneck, vs[-1] * fs[-1]),
            Reshape(fs[-1], vs[-1]),
            *tblocks,
            Permute(1, 0))

    def forward(self, inputs):
        x = inputs.float()
        skips = []
        i = 0
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, SConvBlock):
                if i in self.skip_idx:
                    skips += [self.drop(x)]
                i += 1
        
        for layer in self.decoder:
            if isinstance(layer, TConvBlock):
                i -= 1
                if i in self.skip_idx:
                    act = self.acts[i % len(self.acts)]
                    skip = act(skips[-1])
                    x = torch.cat((x, skip), dim=1)
                    skips = skips[:-1]
            x = layer(x)
        return x.type(inputs.dtype)
    
##########################################################################################

class MeshDataset(Dataset):
    def __init__(self, obj1_data, obj2_data):
        self.x = obj1_data
        self.y = obj2_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = ToTensor()(self.x[idx]).squeeze(0)
        y = ToTensor()(self.y[idx]).squeeze(0)
        return x, y
