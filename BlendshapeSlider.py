import bpy
from bpy.props import FloatProperty, EnumProperty, BoolProperty
from BMesh import BMesh
from collections import defaultdict
import numpy as np

target_names = ['Face', 'Face.True', 'Face.Pred', 'head.true', 'head.pred']

blendshape_keys = {}

def objgen(obj_names=[], needs_shapekeys=False):
    namelist = obj_names if len(obj_names) else target_names
    for obj_name in namelist:
        if obj_name not in bpy.data.objects.keys():
            print(f"[!] {obj_name} failed to load; skipping"); continue
        obj = bpy.data.objects.get(obj_name)
        if needs_shapekeys and not obj.data.shape_keys:
            print(f"[!] {obj_name} has no shape keys; skipping"); continue
        yield obj_name, obj

curr_key = ""
ckeyrefs = []
def update_blend_shape(self, context):
    global curr_key, ckeyrefs
    blendshape_value = self.blendshape_value
    blendshape_key = self.blendshape_key
    if curr_key != blendshape_key:
        for objname, obj in objgen():
            if obj.data.shape_keys:
                blendshape_keys[objname] = {key.name: key for key in obj.data.shape_keys.key_blocks}
        ckeyrefs = []
        for obj_name, keys in blendshape_keys.items():
            obj = bpy.data.objects.get(obj_name)
            vcols = obj.data.vertex_colors
            for col in vcols.values():
                col.active = False
            for keyname, keyref in keys.items():
                # if obj_name == 'Face.002':
                    # print(keyname, blendshape_key, keyname in blendshape_key, blendshape_key in keyname)
                if keyname.replace("-Pred", "") == blendshape_key or blendshape_key.replace("-Pred", "") == keyname: 
                    ckeyrefs += [keyref]
                    keyref.value = blendshape_value
                    if keyname in vcols.keys(): 
                        vcols.get(keyname).active = 1
                else: 
                    keyref.value = 0
        curr_key = blendshape_key

    for keyref in ckeyrefs:
        keyref.value = blendshape_value
        
def get_shape_keys(self, context):
    shape_keys = []
    true_count = defaultdict(lambda: 0)
    pred_count = defaultdict(lambda: 0)

    for obj_name, keys in blendshape_keys.items():
        for key in keys.keys():
            if key == "Basis":
                continue
            elif key.endswith("-Pred"):
                pred_count[key[:-5]] += 1
            else:
                true_count[key] += 1
                if key not in shape_keys:
                    shape_keys += [key]
    tc, pc = true_count, pred_count
    return [(key, f"{key} ({tc[key]}T/{pc[key]}P)", "") for key in shape_keys if (tc[key] + pc[key]) > 1]

##########

class SelectDeformedOperator(bpy.types.Operator):
    bl_idname = "object.select_deformed"
    bl_label = "Select Deformed Vertices"

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and context.object is not None

    def execute(self, context):
        obj = context.object
        if obj.type != 'MESH' or obj.data.shape_keys is None:
            self.report({'WARNING'}, "Active object is not a mesh or has no shape keys")
            return {'CANCELLED'}

        sk = obj.data.shape_keys
        kb = sk.key_blocks.get(context.scene.blendshape_key)
        if kb is None:
            self.report({'WARNING'}, "No shape key found with the specified name")
            return {'CANCELLED'}

        threshold = context.scene.deform_threshold

        # Switch to edit mode to be able to select vertices
        bpy.ops.object.mode_set(mode='EDIT')

        # Deselect all vertices
        bpy.ops.mesh.select_all(action='DESELECT')

        # Switch back to object mode to be able to access vertices data
        bpy.ops.object.mode_set(mode='OBJECT')

        # Select vertices that have been deformed by the shape key
        for i, vertex in enumerate(obj.data.vertices):
            deform = kb.data[i].co - vertex.co
            if deform.length > threshold:
                vertex.select = True

        # Switch back to edit mode to show the selection
        bpy.ops.object.mode_set(mode='EDIT')

        return {'FINISHED'}

##########

class BlendShapePanel(bpy.types.Panel):
    bl_label = "Blend Shape Control"
    bl_idname = "OBJECT_PT_blend_shape_control"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Blend Shapes'

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'PAINT_VERTEX'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "blendshape_key", text="Shape Key")
        layout.prop(scene, "blendshape_value", slider=True)
        layout.prop(scene, "deform_threshold", text="Deformation Threshold")
        layout.operator("object.select_deformed", text="Select Deformed Vertices")

def register(obj_names):
    # Pre-compute the shape keys and store their references
    global blendshape_keys, target_names
    target_names = obj_names
    blendshape_keys = {}
    bpy.utils.register_class(BlendShapePanel)
    bpy.types.Scene.blendshape_value = FloatProperty(
        name="Blend Shape Value",
        description="Control the blend shape value",
        min=0, max=1, default=0,
        update=update_blend_shape,
    )
    bpy.types.Scene.blendshape_key = EnumProperty(
        name="Blend Shape Key",
        description="Select the blend shape key to control",
        items=get_shape_keys,
    )
    bpy.types.Scene.deform_threshold = FloatProperty(
        name="Deformation Threshold",
        description="Vertices with deformation above this threshold will be selected",
        min=0, max=1, default=0.1,
    )
    for objname, obj in objgen():
        if obj.data.shape_keys:
            blendshape_keys[objname] = {key.name: key for key in obj.data.shape_keys.key_blocks}
    bpy.utils.register_class(SelectDeformedOperator)

def unregister():
    global handlers
    bpy.utils.unregister_class(BlendShapePanel)
    bpy.utils.unregister_class(SelectDeformedOperator)
    del bpy.types.Scene.blendshape_value
    del bpy.types.Scene.blendshape_key

def rename_shape_keys(obj_name, rename_dict=[], substring_dict={"_L":"Left", "_R":"Right"}):
    # # Example usage
    # target_mesh_name = "YourMeshObjectName"
    # rename_shape_keys(target_mesh_name, { "browDown_L": "browDownLeft", "browDown_R": "browDownRight" })
    for objname, obj in objgen([obj_name], needs_shapekeys=True):
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name in rename_dict:
                key_block.name = rename_dict[key_block.name]
            for s1, s2 in substring_dict.items():
                if s1 in key_block.name:
                    key_block.name = key_block.name.replace(s1, s2)

def unregister_class_by_name(class_name):
    cls = getattr(bpy.types, class_name, None)
    if cls: bpy.utils.unregister_class(cls)
    else: print(f"Class {class_name} not found.")

if __name__ == "__main__":
    register()