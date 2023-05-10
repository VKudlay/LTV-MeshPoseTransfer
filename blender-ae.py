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

with ( 
    BMesh(bpy.data.objects['Face']) as obj1k,
    BMesh(bpy.data.objects['Face.True']) as obj2t,
    BMesh(bpy.data.objects['Face.Pred']) as obj2p,
    BMesh(bpy.data.objects['head.true']) as obj3k,
    BMesh(bpy.data.objects['head.pred']) as obj3p,
    # BMesh(bpy.data.objects['Lumine.001']) as objn,
):  
    t1_keys = [k for k in obj2t.keys if k != 'Basis' and k in obj1k.keys]
    t2_keys = [k for k in obj3k.keys if k != 'Basis' and k in obj1k.keys]
        
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
#        valid_keys = [key for key in obj2t.keys if key not in train_keys and key in obj1k.keys]
    if trial == 2:
        cut_keys = ['Fcl_ALL_Angry', 'Fcl_MTH_Angry', 'Fcl_EYE_Close_R', 'Fcl_MTH_SkinFung_R']
        cut_keys = [key for key in cut_keys if key in t1_keys]
        train_keys = [key for key in t1_keys if key not in cut_keys]
        valid_keys = [key for key in t1_keys if key not in train_keys]
    kno_keys = ['Basis'] + train_keys + valid_keys + cut_keys
    all_keys = kno_keys + t2_keys

    ## https://hal.science/hal-03716435/file/Representation_learning_of_3D_meshes_using_an_Autoencoder_in_the_spectral_domain.pdf

    use_diff = True ## Train to predict per-vertex position or per-vertex displacement from basis
    load_pretrained = False
    batch_size = 256
    if use_diff: ## Settings that seem to work for me
        MeshXAE = MeshVUNet # MeshAE
        k_dims = 1024       ## Graph Laplacian eigenvector count
        bneck = 512         ## Bottleneck dimension
        vs=[256, 64, 32, 16, 8]    ## "Latent Vertex" dimension progression
        fs=[8, 16, 32, 64, 256]    ## Feature dimension progression
        epochs = 10000
        lr = 1e-3
        act_fn = torch.nn.Tanh
        sched_decay = 0.2   ## Scheduler settings
        sched_patience = 500
        sched_thresh = 1e-3
        ae_kwargs = {
            'num_encoders' : 2,
            'num_decoders' : 2,
#            'skip_idx' : [1, 2, 3]
        }
        arches = [
            (0,0), 
            (0,1)
        ]
#    else:  ## Paper Settings: Very slow to converge (or does it even? idk)
#        MeshXAE = MeshUVAE # MeshAE
#        k_dims = 2048       ## Graph Laplacian eigenvector count
#        bneck = 256          ## Bottleneck dimension
#        vs=[256, 64, 16, 8]    ## "Latent Vertex" dimension progression
#        fs=[8, 16, 64, 256]    ## Feature dimension progression
#        epochs = 10000
#        lr = 5e-3
#        act_fn = torch.nn.Tanh
#        sched_decay = 0.5   ## Scheduler settings
#        sched_patience = 300
#        sched_thresh = 1e-3
#        ae_kwargs = {
#            'num_encoders' : 1,
#            'num_decoders' : 2,
#            # 'skip_idx' : [1, 2, 3]
#        }
#        arches = [(0,0), (0,1)]
    
    loss_fn = torch.nn.L1Loss() # torch.nn.MSELoss()

    fwd1, inv1 = obj1k.get_spectral_transforms(k_dims)
    fwd2, inv2 = obj2t.get_spectral_transforms(k_dims)
    fwd3, inv3 = obj3k.get_spectral_transforms(k_dims)
    # fwdn, invn = objn.get_spectral_transforms(k_dims)
        
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
        preprocess_data(obj1k, train_keys, fwd1, True), 
        preprocess_data(obj2t, train_keys, fwd2, True), train_keys)
    vdata = BlendshapeDataset(
        preprocess_data(obj1k, valid_keys, fwd1, True),
        preprocess_data(obj2t, valid_keys, fwd2, True), valid_keys)
    udata = BlendshapeDataset(
        preprocess_data(obj1k, t2_keys, fwd1, False),
        preprocess_data(obj3k, t2_keys, fwd3, False), t2_keys)
    # ndata1 = BlendshapeDataset(
    #     preprocess_data(objn, ['Basis'], fwdn, True),
    #     preprocess_data(obj2t, ['Basis'], fwd2, True), ['Basis'])
    # ndata2 = BlendshapeDataset(
    #     preprocess_data(objn, ['Basis'], fwdn),
    #     preprocess_data(obj3k, ['Basis'], fwd3), ['Basis'])

    kdata = MeshDataset(
        preprocess_data(obj1k, kno_keys, fwd1),
        preprocess_data(obj2t, kno_keys, fwd2), kno_keys)
    adata_v = preprocess_data(obj1k, all_keys, fwd1)
    adata = MeshDataset(adata_v, adata_v, all_keys)
    
    ## Effective domain of t1 data, t2 data in obj1k, t2 data in obj3k, basis in objn
    d1, d2, d3 = kdata.x, kdata.y, udata.y
    # d4 = ndata1.x
    
    tdata.fit(d1, d2) ## obj1k-t1 -> obj2t-t2 (for training)
    vdata.fit(d1, d2) ## obj1k-t1 -> obj2t-t2 (for validation)
    kdata.fit(d1, d2) ## obj1k-t1 -> obj2t-t2 (for known error)
    adata.fit(d1, d1) ## obj1k-t1 -> obj1k-t1 (for visualization)
    udata.fit(d1, d3) ## obj1k-t1 -> obj3k-t1 (for supervision)
    # ndata1.fit(d4, d2)
    # ndata2.fit(d4, d3)
    
    tloader = DataLoader(tdata, batch_size=batch_size, shuffle=False)
    vloader = DataLoader(vdata, batch_size=batch_size, shuffle=False)
    uloader = DataLoader(udata, batch_size=batch_size, shuffle=False) 
    kloader = DataLoader(kdata, batch_size=1, shuffle=False)
    aloader = DataLoader(adata, batch_size=1, shuffle=False)
    # nloader1 = DataLoader(ndata1, batch_size=1, shuffle=False)
    # nloader2 = DataLoader(ndata2, batch_size=1, shuffle=False)
        
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
        try: 
            trainer.train((tloader, uloader), vloader, arches=arches, epochs=epochs)
        except: pass
        # Save the trained model
        model_filename = generate_model_filename(MeshXAE.__name__, k_dims, bneck, vs, fs, epochs, lr)
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved as {model_filename}")

    # Make predictions on the validation set
    with torch.no_grad():
        model.eval()
        pred_diffs_all = []
        pred_diffs_kno = []
        pred_diffs_ob3 = []
        for key, data in zip(all_keys, aloader):
            outputs = tdata.denorm(model(data[0].to(device)), 'y')
            pred_diffs_all += [inv2(outputs)]
        for key, data in zip(kno_keys, kloader):
            outputs = tdata.denorm(model(data[0].to(device)), 'y')
            pred_diffs_kno += [inv2(outputs)]
        for key, data in zip(t2_keys, uloader):
            outputs = udata.denorm(model(data[0].to(device), dec=1), 'y')
            pred_diffs_ob3 += [inv3(outputs)]
        pred_diffs_all = np.concatenate(pred_diffs_all, axis=0)
        pred_diffs_kno = np.concatenate(pred_diffs_kno, axis=0)
        pred_diffs_ob3 = np.concatenate(pred_diffs_ob3, axis=0)

    # Add the predicted deformations as shape keys on obj2p
    pred_keys = ["-".join([key] + (["Pred"] * (key not in train_keys))) for key in all_keys]
   
    # # Calculate error between predicted and true shape keys of obj2t
    true_diffs_kno = preprocess_data(obj2t, kno_keys)
    errors = np.abs(pred_diffs_kno - true_diffs_kno)
   
    obj2p.clear_shape_keys()
    for i, (pred_diff, pkey) in enumerate(zip(pred_diffs_all, pred_keys)):
        obj2p.update_shape_key(pkey, pred_diff, offset=use_diff)
        kkey = pkey.replace("-Pred", "")
        if kkey in kno_keys:
            obj2p.color_vertices(errors[kno_keys.index(kkey)], scale=True, layer_name=pkey)
        if kkey in t2_keys:
            obj3p.update_shape_key(kkey, pred_diffs_ob3[t2_keys.index(kkey)], offset=use_diff)
    
    obj1k.set_shape_key(0)
    obj2t.set_shape_key(0)
    obj2p.set_shape_key(0)
    obj3k.set_shape_key(0)
    obj3p.set_shape_key(0)
    obj2p.activate()
    
## Registers BlendShape Slider; Go to Object Mode in viewport and type "n"
from BlendshapeSlider import register, unregister
register() 