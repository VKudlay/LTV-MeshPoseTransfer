## OPEN THIS FILE IN BLENDER SCRIPT EDITOR

## TODO: 
##  Make the program non-blocking (i.e. let the optimization run while still allowing blender interactions)
##  Incorporate visualization into training callback to visualize training progressively on multiple expressions.
##  Consider other modifications (VAE, special training, etc.)
import pip
#pip.main("install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117".split())
#pip.main("install numpy scipy tqdm".split())

import torch
import bpy
import numpy as np
import sys
from torch.utils.data import DataLoader, ConcatDataset

## Make sure current directory is available for imports and refresh any import history
sys.path += [] if sys.path[-1]=='.' else ['.']
try: del sys.modules['BMesh']
except: pass
try: del sys.modules['MeshAE']
except: pass
try: del sys.modules['BlendshapeSlider']
except: pass

from BMesh import *
from MeshAE import *

## Enter Vertex Paint mode (not necessary at this point. Feel free to change)
# bpy.ops.object.mode_set(mode='VERTEX_PAINT')

ref_face = 'Face.Input'
new_face = 'Face'

with ( 
    BMesh(bpy.data.objects[ref_face]) as obj1,
    BMesh(bpy.data.objects[new_face]) as obj2,
):  
    t1_keys = [k for k in obj2.keys if k != 'Basis' and k in obj1.keys and k.startswith("Fcl_")]
    t2_keys = [k for k in obj1.keys if k != 'Basis' and k not in t1_keys and not k.startswith("_")]
        
    cut_keys = ['Fcl_ALL_Angry', 'Fcl_MTH_Angry', 'Fcl_EYE_Close_R', 'Fcl_MTH_SkinFung_R']
    cut_keys = [key for key in cut_keys if key in t1_keys]
    train_keys = [key for key in t1_keys if key not in cut_keys]
    valid_keys = [key for key in t1_keys if key not in train_keys]
    kno_keys = ['Basis'] + train_keys + valid_keys
    all_keys = kno_keys + t2_keys

    ## https://hal.science/hal-03716435/file/Representation_learning_of_3D_meshes_using_an_Autoencoder_in_the_spectral_domain.pdf

    use_diff = True ## Train to predict per-vertex position or per-vertex displacement from basis
    load_pretrained = False
    batch_size = 128
    if use_diff: ## Settings that seem to work for me
        MeshXAE = MeshUNet  # MeshAE
        k_dims = 1024       ## Graph Laplacian eigenvector count
        bneck = 128         ## Bottleneck dimension
        epochs = 5000
        lr = 5e-3
        sched_decay = 0.2   ## Scheduler settings
        sched_patience = 500
        sched_thresh = 1e-3
        ae_kwargs = {
            'vs' : [256, 64, 16, 4],  ## "Latent Vertex" dimension progression
            'fs' : [4, 16, 64, 256],  ## Feature dimension progression
            'num_encoders' : 1,
            'num_decoders' : 2,
            'bnorm' : True,
            'act_fn' : torch.nn.Tanh
        }
        arches = [(0,0), (0,1)]
    
    loss_fn = torch.nn.L1Loss() # torch.nn.MSELoss()

    fwd1, inv1 = obj1.get_spectral_transforms(k_dims)
    fwd2, inv2 = obj2.get_spectral_transforms(k_dims)
    
    if use_diff:    vfunc = lambda pose_vs: [pose_v - pose_vs[0] for pose_v in pose_vs[1:]]
    else:           vfunc = lambda pose_vs: pose_vs[1:]
    
    def preprocess_data(obj, keys, sfunc=lambda x:x, get_masked=False):
        values = vfunc([obj.get_vertex_positions(key) for key in ([0]+keys)])
        pos_x = values[0][:, :1] < 0
        out = np.stack([sfunc(v) for v in values], axis=0)
        if get_masked:
            out1 = np.stack([sfunc(v * (pos_x    )) for v in values], axis=0)
            out2 = np.stack([sfunc(v * (1 - pos_x)) for v in values], axis=0)
            return out, out1, out2
        return out
    
    ## Training/validation for task 3: Lo+Hi -> Lo
    tdata = BlendshapeDataset(
        preprocess_data(obj1, train_keys, fwd1, True), 
        preprocess_data(obj2, train_keys, fwd2, True), train_keys)
    vdata = BlendshapeDataset(
        preprocess_data(obj1, valid_keys, fwd1, True),
        preprocess_data(obj2, valid_keys, fwd2, True), valid_keys)
    idata = BlendshapeDataset(
        preprocess_data(obj1, t2_keys, fwd1, True),
        preprocess_data(obj1, t2_keys, fwd1, True), all_keys)

    adata = MeshDataset(
        preprocess_data(obj1, kno_keys, fwd1),
        preprocess_data(obj2, kno_keys, fwd2), kno_keys)        
    kdata = MeshDataset(
        preprocess_data(obj1, t2_keys, fwd1),
        preprocess_data(obj2, t2_keys, fwd2), t2_keys)
    
    ## Effective domain of t1 data, t2 data in obj1, t2 data in obj3k, basis in objn
    d1, d2 = adata.x, adata.y

    tdata.fit(d1, d2) ## obj1-t1 -> obj2-t2 (for training)
    vdata.fit(d1, d2) ## obj1-t1 -> obj2-t2 (for validation)
    kdata.fit(d1, d2) ## obj1-t1 -> obj2-t2 (for known error)
    idata.fit(d1, d1) 
    adata.fit(d1, d2)

    tloader = DataLoader(tdata, batch_size=batch_size, shuffle=False)
    vloader = DataLoader(vdata, batch_size=batch_size, shuffle=False)
    iloader = DataLoader(idata, batch_size=batch_size, shuffle=False) 
    kloader = DataLoader(kdata, batch_size=1, shuffle=False)
    aloader = DataLoader(adata, batch_size=1, shuffle=False)
        
    def generate_model_filename(arch, k_dims, bneck, epochs, lr):
        return f"{arch}_k_dims-{k_dims}_bneck-{bneck}_epochs-{epochs}_lr-{lr:.1e}.pth"
    
    def load_model(model_class, device, model_filename):
        model = model_class(nv1=k_dims, nv2=k_dims, bneck=bneck, **ae_kwargs)
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.to(device)
        return model

    # Instantiate and train the MeshAE model or load a pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_filename = generate_model_filename(MeshXAE.__name__, k_dims, bneck, epochs, lr)
    if load_pretrained:
        model = load_model(MeshXAE, device, model_filename)
        print(f"Loaded pre-trained model from {model_filename}")
    else:
        # Instantiate and train the MeshAE model
        model = MeshXAE(nv1=k_dims, nv2=k_dims, bneck=bneck, **ae_kwargs)
        model.compile(loss=loss_fn, device=device)
        trainer = Trainer(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.compile(
            optimizer = optimizer,
            scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sched_decay, patience=sched_patience, threshold=sched_thresh), 
            sched_step = lambda s,l,m: s.step(l))
        try: 
            trainer.train([tloader, iloader], vloader, arches=arches, epochs=epochs)
        except: pass
        # Save the trained model
        model_filename = generate_model_filename(MeshXAE.__name__, k_dims, bneck, epochs, lr)
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    # Make predictions on the validation set
    with torch.no_grad():
        model.eval()
        pred_diffs = []
        for key, data in zip(t2_keys, kloader):
            outputs = tdata.denorm(model(data[0].to(device)), 'y')
            pred_diffs += [inv2(outputs)]
        pred_diffs = np.concatenate(pred_diffs, axis=0)

    # Add the predicted deformations as shape keys on obj2p
    pred_keys = t2_keys
   
    for i, (pred_diff, pkey) in enumerate(zip(pred_diffs, pred_keys)):
        obj2.update_shape_key(pkey, pred_diff, offset=use_diff)
    
    obj1.set_shape_key(0)
    obj2.set_shape_key(0)
    obj2.activate()
    
## Registers BlendShape Slider; Go to Object Mode in viewport and type "n"
from BlendshapeSlider import register, unregister
register([ref_face, new_face]) 