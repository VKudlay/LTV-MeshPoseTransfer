## OPEN THIS FILE IN BLENDER SCRIPT EDITOR

## TODO: 
##  Make the program non-blocking (i.e. let the optimization run while still allowing blender interactions)
##  Incorporate visualization into training callback to visualize training progressively on multiple expressions.
##  Consider other modifications (VAE, special training, etc.)

import torch
import bpy
import numpy as np
import sys
from torch.utils.data import DataLoader

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
from BlendshapeSlider import register as register_bs
register_bs() 

## Enter Vertex Paint mode (not necessary at this point. Feel free to change)
# bpy.ops.object.mode_set(mode='VERTEX_PAINT')

def preprocess_data(obj, keys, func=lambda x: x, use_diff=False):
    rest_v = obj.get_vertex_positions(0)
    pose_vs = [obj.get_vertex_positions(key) for key in keys]
    values = pose_vs if not use_diff else [pose_v - rest_v for pose_v in pose_vs]
    coefs = [func(v) for v in values]
    return np.stack(coefs, axis=0)

with ( 
    BMesh(bpy.data.objects['Face']) as obj1,
    BMesh(bpy.data.objects['Face.001']) as obj2,
    BMesh(bpy.data.objects['Face.002']) as objp
):  
    trial = 2
    if trial == 1:
        train_keys = [
            'Basis',
            # 'Fcl_ALL_Angry', 'Fcl_MTH_Angry',
            'Fcl_ALL_Fun', 'Fcl_ALL_Joy', 'Fcl_ALL_Sorrow', 'Fcl_ALL_Surprised',
            'Fcl_EYE_Close_R', 'Fcl_EYE_Close_L', 'Fcl_MTH_Up',
            'Fcl_MTH_Down', 'Fcl_MTH_Small', 'Fcl_MTH_Large', 'Fcl_MTH_O', 'Fcl_MTH_I',
            'Fcl_EYE_Iris_Hide',
            'Fcl_MTH_SkinFung', 'Fcl_MTH_SkinFung_R',
            'Fcl_HA_Short', 'Fcl_HA_Hide'
        ]
        test_keys = [key for key in obj2.keys if key not in train_keys and key in obj1.keys]
    if trial == 2:
        cut_keys = ['Fcl_ALL_Angry', 'Fcl_MTH_Angry', 'Fcl_EYE_Close_R', 'Fcl_MTH_SkinFung_R']
        train_keys = [key for key in obj2.keys if key not in cut_keys and key in obj1.keys]
        test_keys = [key for key in obj2.keys if key in obj1.keys]
    all_keys = train_keys + test_keys

    ## https://hal.science/hal-03716435/file/Representation_learning_of_3D_meshes_using_an_Autoencoder_in_the_spectral_domain.pdf

    use_diff = True ## Train to predict per-vertex position or per-vertex displacement from basis
    if use_diff: ## Settings that seem to work for me
        MeshXAE = MeshUNet # MeshAE
        k_dims = 2048       ## Graph Laplacian eigenvector count
        bneck = 64          ## Bottleneck dimension
        vs=[256, 64, 32, 16]    ## "Latent Vertex" dimension progression
        fs=[16, 32, 64, 256]    ## Feature dimension progression
        epochs = 20000
        lr = 1e-3
        act_fn = torch.nn.LeakyReLU
        sched_decay = 0.5   ## Scheduler settings
        sched_patience = 500
        sched_thresh = 1e-3
    else:  ## Paper Settings: Very slow to converge (or does it even? idk)
        MeshXAE = MeshVAE
        k_dims = 2048
        bneck = 64
        epochs = 5000      ## Paper didn't give epoch count; just ran for 20 hours 
        lr = 1e-4
        vs=[256, 128, 64, 32, 16]
        fs=vs[::-1]
        act_fn = torch.nn.LeakyReLU
        sched_decay = 0.5
        sched_patience = 100 ## idk lol. 3 seems excessively small
        sched_thresh = 1e-4
    
    loss_fn = torch.nn.L1Loss()
#    loss_fn = torch.nn.MSELoss()

    fwd1, inv1 = obj1.get_spectral_transforms(k_dims)
    fwd2, inv2 = obj2.get_spectral_transforms(k_dims)
    
    obj1_data = preprocess_data(obj1, train_keys, fwd1, use_diff=use_diff)
    obj2_data = preprocess_data(obj2, train_keys, fwd2, use_diff=use_diff)
    tdataset = MeshDataset(obj1_data, obj2_data)

    obj1_data2 = preprocess_data(obj1, test_keys, fwd1, use_diff=use_diff)
    obj2_data2 = preprocess_data(obj2, test_keys, fwd2, use_diff=use_diff)
    vdataset = MeshDataset(obj1_data2, obj2_data2)

    tloader = DataLoader(tdataset, batch_size=len(tdataset), shuffle=False)
    vloader = DataLoader(vdataset, batch_size=len(vdataset), shuffle=False)

    # Instantiate and train the MeshAE model
    model = MeshXAE(
        nv1 = k_dims, 
        nv2 = k_dims, 
        bneck = bneck,
        vs = vs, 
        fs = fs
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.compile(loss=loss_fn, device=device)
    trainer = Trainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer.compile(
        optimizer = optimizer,
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sched_decay, patience=sched_patience, threshold=sched_thresh), 
        sched_step = lambda s,l,m: s.step(l)
    )
    trainer.train(tloader, vloader, epochs=epochs)

    # Make predictions on the validation set
    with torch.no_grad():
        model.eval()
        true_diffs = []
        pred_diffs = []
        for data in tloader:
            outputs = model(data[0].to(device))
            true_diffs += [inv2(data[1].cpu().numpy())]
            pred_diffs += [inv2(outputs.cpu().numpy())]
        for data in vloader:
            outputs = model(data[0].to(device))
            true_diffs += [inv2(data[1].cpu().numpy())]
            pred_diffs += [inv2(outputs.cpu().numpy())]
        true_diffs = np.concatenate(true_diffs, axis=0)
        pred_diffs = np.concatenate(pred_diffs, axis=0)

    # Add the predicted deformations as shape keys on objp
    pred_keys = [obj1.get_key(key)[1] for key in all_keys]
    pred_keys = ["-".join([key] + ["Pred"] * (key not in train_keys)) for key in pred_keys]
   
    objp.clear_shape_keys()
    for i, (pred_diff, true_diff, key) in enumerate(zip(pred_diffs, true_diffs, pred_keys)):
        objp.update_shape_key(key, pred_diff, offset=use_diff)
    
    # Calculate error between predicted and true shape keys of obj2
    true_shape_keys = [obj2.get_vertex_positions(key) for key in all_keys]
    errors = np.abs(true_shape_keys - pred_diffs)

    # Set color of objp vertices based on the error
    for i, (error_color, key) in enumerate(zip(errors, pred_keys)):
        objp.set_shape_key(key)
        objp.color_vertices(error_color, scale=True)
    objp.activate()