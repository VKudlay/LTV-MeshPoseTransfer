import bpy
from bpy.props import FloatProperty, EnumProperty # , CollectionProperty, PointerProperty
# from bpy.types import Operator, Panel, PropertyGroup, UIList
from BMesh import BMesh
from collections import defaultdict

target_names = ['Face', 'Face.001', 'Face.002']

def update_blend_shape(self, context):
    blendshape_value = self.blendshape_value
    blendshape_key = self.blendshape_key

    for obj_name in target_names:
        if obj_name not in bpy.data.objects.keys():
            print(f"[!] {obj_name} failed to load; skipping"); continue
        with BMesh(bpy.data.objects.get(obj_name)) as obj:
            if blendshape_key in obj.keys:
                curr_shapekey = blendshape_key
            elif f"{blendshape_key}-Pred" in obj.keys:
                curr_shapekey = f"{blendshape_key}-Pred"
            else: continue
            obj.set_shape_key(curr_shapekey, blendshape_value)

def get_shape_keys(self, context):
    shape_keys = []
    true_count = defaultdict(lambda: 0)
    pred_count = defaultdict(lambda: 0)
    for obj_name in target_names:
        if obj_name not in bpy.data.objects.keys():
            print(f"[!] {obj_name} failed to load; skipping"); continue
        with BMesh(bpy.data.objects.get(obj_name)) as obj:
            for key in obj.keys:
                if key == "Basis": 
                    continue
                elif key.endswith("-Pred"): 
                    pred_count[key[:-5]] += 1
                else:
                    true_count[key] += 1
                    if key not in shape_keys:
                        shape_keys += [key]
    tc, pc = true_count, pred_count
    return [(key, f"{key} ({tc[key]}T/{pc[key]}P)", "") for key in shape_keys if (tc[key]+pc[key]) > 2]

class BlendShapePanel(bpy.types.Panel):
    bl_label = "Blend Shape Control"
    bl_idname = "OBJECT_PT_blend_shape_control"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Blend Shapes'
    bl_context = 'objectmode'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "blendshape_key", text="Shape Key")
        layout.prop(scene, "blendshape_value", slider=True)

def register():
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

def unregister():
    bpy.utils.unregister_class(BlendShapePanel)
    del bpy.types.Scene.blendshape_value
    del bpy.types.Scene.blendshape_key

if __name__ == "__main__":
    register()