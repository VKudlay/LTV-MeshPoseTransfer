## OPEN THIS FILE IN BLENDER SCRIPT EDITOR

## TODO: 
##  Make the program non-blocking (i.e. let the optimization run while still allowing blender interactions)
##  Incorporate visualization into training callback to visualize training progressively on multiple expressions.
##  Consider other modifications (VAE, special training, etc.)

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

## Registers BlendShape Slider; Go to Object Mode in viewport and type "n"
from BlendshapeSlider import register, unregister
register() 

## Enter Vertex Paint mode (not necessary at this point. Feel free to change)
# bpy.ops.object.mode_set(mode='VERTEX_PAINT')

with ( 
    BMesh(bpy.data.objects['Face']) as obj1,
    BMesh(bpy.data.objects['Face.001']) as obj2,
    BMesh(bpy.data.objects['Face.002']) as objp,
    BMesh(bpy.data.objects['head']) as objk,
):  
    t1_keys = [k for k in obj2.keys if k != 'Basis']
    t2_keys = [k for k in objk.keys if k != 'Basis']
    un_keys = [k for k in obj2.keys if k not in obj1.keys]
    kno_keys = [k for k in obj2.keys if k not in un_keys]
    
    trial = 2
#    if trial == 1:
#        train_keys = [
#            'Basis', # 'Fcl_ALL_Angry', 'Fcl_MTH_Angry',
#            'Fcl_ALL_Fun', 'Fcl_ALL_Joy', 'Fcl_ALL_Sorrow', 'Fcl_ALL_Surprised',
#            'Fcl_EYE_Close_R', 'Fcl_EYE_Close_L', 'Fcl_MTH_Up',
#            'Fcl_MTH_Down', 'Fcl_MTH_Small', 'Fcl_MTH_Large', 'Fcl_MTH_O', 'Fcl_MTH_I',
#            'Fcl_EYE_Iris_Hide',
#            'Fcl_MTH_SkinFung', 'Fcl_MTH_SkinFung_R',
#            'Fcl_HA_Short', 'Fcl_HA_Hide'
#        ]
#        valid_keys = [key for key in obj2.keys if key not in train_keys and key in obj1.keys]
    if trial == 2:
        cut_keys = ['Fcl_ALL_Angry', 'Fcl_MTH_Angry', 'Fcl_EYE_Close_R', 'Fcl_MTH_SkinFung_R']
        train_keys = [key for key in t1_keys if key not in (cut_keys + un_keys)]
        valid_keys = [key for key in t1_keys if key not in (train_keys + un_keys)]
    all_keys = t1_keys + t2_keys

    ## https://hal.science/hal-03716435/file/Representation_learning_of_3D_meshes_using_an_Autoencoder_in_the_spectral_domain.pdf

    use_diff = True ## Train to predict per-vertex position or per-vertex displacement from basis
    load_pretrained = True
    if use_diff: ## Settings that seem to work for me
        MeshXAE = MeshUNet # MeshAE
        k_dims = 2048       ## Graph Laplacian eigenvector count
        bneck = 64          ## Bottleneck dimension
        vs=[256, 64, 16, 4]    ## "Latent Vertex" dimension progression
        fs=[4, 16, 64, 256]    ## Feature dimension progression
        epochs = 10000
        lr = 1e-3
        act_fn = torch.nn.Tanh
        sched_decay = 0.5   ## Scheduler settings
        sched_patience = 500
        sched_thresh = 1e-4
        ae_kwargs = {
            'num_encoders' : 1,
            'num_decoders' : 2,
            'skip_idx' : [1, 2]
        }
        arches = [(0,0), (0,1)]
#    if use_diff: ## Settings that seem to work for me
#        MeshXAE = MeshUNet # MeshAE
#        k_dims = 2048       ## Graph Laplacian eigenvector count
#        bneck = 64          ## Bottleneck dimension
#        vs=[256, 64, 16, 4]    ## "Latent Vertex" dimension progression
#        fs=[4, 16, 64, 256]    ## Feature dimension progression
#        epochs = 20000
#        lr = 1e-3
#        act_fn = torch.nn.LeakyReLU
#        sched_decay = 0.5   ## Scheduler settings
#        sched_patience = 500
#        sched_thresh = 1e-5
#        ae_kwargs = {'skip_idx' : [1, 2, 3]}
#    else:  ## Paper Settings: Very slow to converge (or does it even? idk)
#        MeshXAE = MeshVAE
#        k_dims = 2048
#        bneck = 64
#        epochs = 5000      ## Paper didn't give epoch count; just ran for 20 hours 
#        lr = 1e-4
#        vs=[256, 128, 64, 32, 16]
#        fs=vs[::-1]
#        act_fn = torch.nn.LeakyReLU
#        sched_decay = 0.5
#        sched_patience = 100 ## idk lol. 3 seems excessively small
#        sched_thresh = 1e-4
#        ae_kwargs = {}
    
    loss_fn = torch.nn.L1Loss() # torch.nn.MSELoss()

    fwd1, inv1 = obj1.get_spectral_transforms(k_dims)
    fwd2, inv2 = obj2.get_spectral_transforms(k_dims)
    fwdk, invk = objk.get_spectral_transforms(k_dims)
    
    if use_diff:    vfunc = lambda pose_vs: [pose_v - pose_vs[0] for pose_v in pose_vs[1:]]
    else:           vfunc = lambda pose_vs: pose_vs[1:]
    
    def preprocess_data(obj, keys, sfunc=lambda x:x):
        values = vfunc([obj.get_vertex_positions(key) for key in ([0]+keys)])
        return np.stack([sfunc(v) for v in values], axis=0)
    
    ## Training/validation for task 3: Lo+Hi -> Lo
    tdata = MeshDataset(
        preprocess_data(obj1, train_keys, fwd1), 
        preprocess_data(obj2, train_keys, fwd2))
    vdata = MeshDataset(
        preprocess_data(obj1, valid_keys, fwd1),
        preprocess_data(obj2, valid_keys, fwd2))
    
    ## TODO: Training/validation for task 2: Lo+Hi -> Hi using model1 encoder, new decoder decoder
    udata = MeshDataset(
        preprocess_data(obj1, t2_keys, fwd1),
        preprocess_data(objk, t2_keys, fwdk))
    
    all_x = tdata.x, vdata.x, udata.x
    all_y = tdata.y, vdata.y, udata.y
    for dset in (tdata, vdata, udata):
        dset.fit(all_x, all_y)

    tloader = DataLoader(tdata, batch_size=len(tdata), shuffle=False)
    vloader = DataLoader(vdata, batch_size=len(vdata), shuffle=False)
    uloader = DataLoader(udata, batch_size=len(udata), shuffle=False)
    aloader = DataLoader(ConcatDataset([tdata, vdata, udata]), batch_size=1, shuffle=False)
    
    
    def generate_model_filename(arch, k_dims, bneck, vs, fs, epochs, lr):
        return f"{arch}_k_dims-{k_dims}_bneck-{bneck}_epochs-{epochs}_lr-{lr:.1e}.pth"
    
    def load_model(model_class, device, model_filename):
        model = model_class(nv1=k_dims, nv2=k_dims, bneck=bneck, vs=vs, fs=fs, act_fn=act_fn, **ae_kwargs)
        model.load_state_dict(torch.load(model_filename, map_location=device))
        model.to(device)
        return model

    # Instantiate and train the MeshAE model or load a pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_filename = generate_model_filename(MeshXAE.__name__, k_dims, bneck, vs, fs, epochs, lr)
    if load_pretrained:
        model = load_model(MeshXAE, device, model_filename)
        print(f"Loaded pre-trained model from {model_filename}")
    else:
        # Instantiate and train the MeshAE model
        model = MeshXAE(nv1=k_dims, nv2=k_dims, bneck=bneck, vs=vs, fs=fs, act_fn=act_fn, **ae_kwargs)
        model.compile(loss=loss_fn, device=device)
        trainer = Trainer(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.compile(
            optimizer = optimizer,
            scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sched_decay, patience=sched_patience, threshold=sched_thresh), 
            sched_step = lambda s,l,m: s.step(l))
        trainer.train((tloader, uloader), vloader, arches=arches, epochs=epochs)
        # Save the trained model
        model_filename = generate_model_filename(MeshXAE.__name__, k_dims, bneck, vs, fs, epochs, lr)
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    # Make predictions on the validation set
    with torch.no_grad():
        model.eval()
        true_diffs = []
        pred_diffs_all = []
        pred_diffs_kno = []
        for key, data in zip(all_keys, aloader):
            outputs = model(data[0].to(device))
            pred_diffs_all += [inv2(outputs.cpu().numpy())]
            if key in t1_keys:
                pred_diffs_kno += [pred_diffs_all[-1]]
        pred_diffs_all = np.concatenate(pred_diffs_all, axis=0)
        pred_diffs_kno = np.concatenate(pred_diffs_kno, axis=0)

    # Add the predicted deformations as shape keys on objp
    pred_keys = ["-".join([key] + (["Pred"] * (key not in train_keys))) for key in all_keys]
   
    # # Calculate error between predicted and true shape keys of obj2
    true_diffs_kno = preprocess_data(obj2, kno_keys)
    errors = np.abs(pred_diffs_kno - true_diffs_kno)
   
    objp.clear_shape_keys()
    for i, (pred_diff, pkey, tkey) in enumerate(zip(pred_diffs_all, pred_keys, all_keys)):
        objp.update_shape_key(pkey, pred_diff, offset=use_diff)
        if i < len(errors):
            objp.color_vertices(errors[i], scale=True, layer_name=pkey)
    
    obj1.set_shape_key(0)
    obj2.set_shape_key(0)
    objp.set_shape_key(0)
    objk.set_shape_key(0)
    objp.activate()