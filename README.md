## Blendshape Transfer - Spectral DL Side
Trying to transfer shapekeys from one mesh to another mesh without necessary vertex alignment. This is the deep learning side of the project using spectral-domain conversion. 

**Other side of the project:** Will link to it when they compile it.  

----

## Presentation Slides: 
- https://docs.google.com/presentation/d/1_PDcN-u-dA_nowIVteLG_1u8tL9WCKV9_CpH4GFptUs/edit?usp=sharing

## Blend Files (chonky)
- https://drive.google.com/drive/folders/1eocGLmWBBebbN7GEmpBsApCuzleFP464?usp=sharing

### References: 
- [Representation Learning of 3D Meshes Using an Autoencoder in the Spectral Domain](https://hal.science/hal-03716435/file/Representation_learning_of_3D_meshes_using_an_Autoencoder_in_the_spectral_domain.pdf)
  - Inspiration for model
- [VRoid Studio](https://vroid.com/en/studio)
  - Most of the meshes were made in [VRoid Studio](https://vroid.com/en/studio), so they were not major efforts/contributions on my part.
- [How to Create ARKit Face Tracking for 3D VTubing](https://www.youtube.com/watch?v=19H82IkEx9k)
  - Great starter/motivating reference, and which links out to "Baldy" AKA simple face mesh with 52 blendshapes, which traces back to an ARKit/rigging community on Discord. 
- [ARKit Face Blendshapes](https://arkit-face-blendshapes.com)
- [@hinzka/52blendshapes-for-VRoid-face](https://github.com/hinzka/52blendshapes-for-VRoid-face)
  - This is where we got the starter ARKit + Fcl model to transfer from
- [GitHub - kugelrund/mesh_segmentation: A python addon for mesh segmentation in blender using spectral clustering methods](https://github.com/kugelrund/mesh_segmentation)
  - Starter code for segmentation routine (had to build upon it to support blender incorporation and vertex-based graph stuff)
- On my end, I used ChatGPT occasionally to speed up development.
  - Was especially helpful for blender-related boilerplate and refactoring.
  - Wasn't very helpful for model tuning and math debugging.
