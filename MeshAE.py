import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm

np.set_printoptions(precision=3)

class Trainer():

    def __init__(self, model):
        self.model = model

    def compile(self, 
        optimizer, 
        scheduler  = None, 
        sched_step = lambda s,l,m: s.step()
    ):
        self.optimizer  = optimizer
        self.scheduler  = scheduler 
        self.sched_step = sched_step if scheduler is not None else lambda s,l,m: None
        self.sched_step = partial(self.sched_step, self.scheduler)
        self.compiled   = True

    def train(self, D0, D1, epochs):
        assert getattr(self, 'compiled', False), f"{self} must be compiled before training"
        stats = defaultdict(lambda: [])
        pbar = tqdm(range(1, epochs+1))
        for epoch in pbar:
            losses  = []
            # metrics = []
            for b, data in enumerate(D0):
                tloss, tmetric = self.train_step(data)
                losses  += [tloss.detach().numpy().mean()]
                # metrics += [tmetric.detach().numpy().mean()]
            stats['loss'] += [np.mean(losses)]
            # stats['metr'] += [np.mean(metrics)]

            losses  = []
            # metrics = []
            for b, data in enumerate(D1): 
                vloss, vmetric = self.valid_step(data)
                losses  += [vloss.detach().numpy().mean()]
                # metrics += [vmetrics.detach().numpy().mean()]
            stats['vloss'] += [np.mean(losses)]
            # stats['metr'] += [np.mean(metrics)]

            trial_stat = {k : np.mean(v[-100:]) for k,v in stats.items()}
            if self.scheduler is not None:
                trial_stat['lr'] = self.scheduler.state_dict()["_last_lr"][0]
            pbar.set_description("| ".join([f"{k} = {v:0.3e} " for k,v in trial_stat.items()]))
            if (pbar.n % 1000) == 0 and pbar.n > 0: 
                print()
            if pbar.n > 1000 and (0.9999 < np.mean(stats['loss'][-101:]) / (np.mean(stats['loss'][-1001:])) < 1):
                print("\nTerminating early")
                break

        return stats

    def train_step(self, data):
        self.model.train()
        return self.batch_step(data, training=True)

    @torch.no_grad()
    def valid_step(self, data):
        self.model.eval()
        return self.batch_step(data, training=False)

    def batch_step(self, data, training=False):
        x, y = [v.to(self.model.device) for v in data]
        if training: 
            self.optimizer.zero_grad()
        loss, metrics = self.model.batch_step(x, y, training=False)
        if training: 
            loss.backward()
            self.optimizer.step()
            self.sched_step(loss, metrics)
        return loss, metrics


class SConvBlock(nn.Module):
    def __init__(self, vs, fs, vstride=2, conv_fn=nn.Conv1d, act_fn=nn.Tanh, padding='valid', drop=0):
        super().__init__()
        self.conv = conv_fn(fs[0], fs[1], kernel_size=(vstride), stride=(vstride), padding=padding) 
        self.down = nn.Linear(vs[0]//vstride, vs[1])
        self.act  = act_fn()
        # self.drop = nn.Dropout(drop)
        # self.bn   = nn.BatchNorm1d(fs[1])
         
    def forward(self, inputs):
        # print(inputs.shape, '->', self.conv, '->')
        x = self.conv(inputs)
        # print(x.shape, '->', self.down, '->')
        x = self.down(x)
        # print(x.shape)
        x = self.act(x)
        # x = self.drop(x)
        # x = self.bn(x)
        return x
    
class TConvBlock(SConvBlock):
    def __init__(self, vs, fs, vstride=2, conv_fn=nn.ConvTranspose1d, act_fn=nn.Tanh, padding=0):
        super().__init__(vs, fs, vstride, conv_fn, act_fn, padding)
        self.down = nn.Linear(vs[0]*vstride, vs[1])

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.axes = args

    def forward(self, x):
        return x.permute(0, *[x+1 for x in self.axes])

##########################################################################################

class MeshAE(pl.LightningModule):
    def __init__(self, nv1, nv2, bneck, vs=[256, 64, 16], fs=[16, 64, 256], act_fn=nn.Tanh):
        super().__init__()
        self.nv1 = nv1
        self.nv2 = nv2
        v_seq = vs
        f_seq = [3] + fs
        self.encoder = self.build_encoder(bneck, v_seq, f_seq, act_fn)
        self.decoder = self.build_decoder(bneck, v_seq, f_seq, act_fn)

    def build_encoder(self, bneck, vs, fs, act_fn):
        cblocks = [SConvBlock((v1,v2), (f1,f2), act_fn=act_fn) 
            for v1,v2,f1,f2 in zip([self.nv1] + vs[:-1], vs, fs[:-1], fs[1:])]
        return nn.Sequential(
            Permute(1, 0),
            *cblocks, 
            nn.Flatten(),
            nn.Linear(vs[-1] * fs[-1], bneck))

    def build_decoder(self, bneck, vs, fs, act_fn):
        tblocks = [TConvBlock((v1,v2), (f1,f2), act_fn=act_fn) 
            for v1,v2,f1,f2 in zip(vs[::-1], vs[::-1][1:] + [self.nv2], fs[::-1][:-1], fs[::-1][1:])]
        return nn.Sequential(
            nn.Linear(bneck, vs[-1] * fs[-1]),
            Reshape(fs[-1], vs[-1]),
            *tblocks,
            Permute(1, 0))

    def compile(self, loss=torch.nn.MSELoss(), metrics=[], device=torch.device("cpu")):
        self.loss = loss
        self.metrics = metrics
        self.to(device)
        
    def forward(self, inputs):
        x = self.encoder(inputs.float())
        x = self.decoder(x)
        return x.type(inputs.dtype)
    
    def batch_step(self, x, y, training=False):
        p = self(x)
        loss = self.loss(p, y)
        with torch.no_grad():
            metrics = {metric.__name__ : metric(p, y).numpy().mean() for metric in self.metrics}
        return loss, metrics
        
class MeshVAE(MeshAE):
    def __init__(self, nv1, nv2, bneck, vs=[256, 64, 16], fs=[16, 64, 256], act_fn=nn.Tanh):
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

    def __init__(self, nv1, nv2, bneck, vs=[256, 64, 16], fs=[16, 64, 256], act_fn=nn.Tanh, skip_idx=[1]):
        self.skip_idx = skip_idx
        super().__init__(nv1, nv2, bneck, vs, fs, act_fn)
        self.drop = lambda x: x # nn.Dropout(0.1)
    
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
                    skips += [x]
                i += 1

        for layer in self.decoder:
            if isinstance(layer, TConvBlock):
                i -= 1
                if i in self.skip_idx:
                    skip = self.drop(skips[-1])
                    x = torch.cat((x, skip), dim=1)
                    skips = skips[:-1]
            x = layer(x)
        return x.type(inputs.dtype)
    
##########################################################################################

class MeshDataset(Dataset):
    def __init__(self, obj1_data, obj2_data):
        self.obj1_data = obj1_data
        self.obj2_data = obj2_data

    def __len__(self):
        return len(self.obj1_data)

    def __getitem__(self, idx):
        x = ToTensor()(self.obj1_data[idx]).squeeze(0)
        y = ToTensor()(self.obj2_data[idx]).squeeze(0)
        return x, y
