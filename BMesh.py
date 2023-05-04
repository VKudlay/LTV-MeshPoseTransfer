import mathutils as mu
import numpy as np
import bmesh
import bpy
import sys
from collections import deque
from segmentation import get_gl_eigvs

import sys
sys.path += [] if sys.path[-1]=='.' else ['.']

def clear_import_cache(key):
    try: del sys.modules[key]   
    except: pass

class BMeshObject:
    def __init__(self, obj=None, mesh=None):
        self.obj = obj
        self.bm = mesh if mesh is not None else bmesh.new()
        self.update_from_object()

    def update_from_object(self):
        # Clears the bmesh, then updates it from the object mesh
        if self.obj is not None:
            self.bm.clear()
            self.bm.from_mesh(self.obj.data)
            # depsgraph = bpy.context.evaluated_depsgraph_get()
            # obj_eval = self.obj.evaluated_get(depsgraph)
            # self.bm.from_object(obj_eval, depsgraph)
        self.bm.verts.ensure_lookup_table()
        self.bm.edges.ensure_lookup_table()
        self.bm.faces.ensure_lookup_table()

    def update_object(self):
        # Updates the object mesh from the bmesh
        if self.obj is not None:
            self.bm.to_mesh(self.obj.data)
            self.obj.data.update()

    @property
    def vertices(self): return self.bm.verts
    @property
    def verts(self): return self.bm.verts
    @property 
    def edges(self): return self.bm.edges
    @property 
    def faces(self): return self.bm.faces
    @property 
    def data(self): return self.obj.data

    def activate(self):
        self.prev_active = bpy.context.view_layer.objects.active
        bpy.context.view_layer.objects.active = self.obj
    
    def deactivate(self):
        if not hasattr(self, 'prev_active'): return
        bpy.context.view_layer.objects.active = self.prev_active
        del self.prev_active

    def get_vertex_positions(self):
        return np.array([v.co[:] for v in self.verts])

    def set_vertex_positions(self, new_verts):
        # Sets the vertex positions in the bmesh from a list of new vertex coordinates
        for v, new_v in zip(self.verts, new_verts):
            v.co = mu.Vector(new_v)
        self.update_object()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.bm.free()


class ShapeKeyMesh(BMeshObject):
    def __init__(self, obj=None, mesh=None):
        super().__init__(obj=obj, mesh=mesh)

    @property 
    def shape_keys(self): 
        return self.obj.data.shape_keys

    def clear_shape_keys(self):
        # Clears all shape keys, and creates a default "Basis" shape key
        if self.shape_keys:
            bpy.context.view_layer.objects.active = self.obj
            for i in range(len(self.shape_keys.key_blocks) - 1, 0, -1):
                self.obj.active_shape_key_index = i
                bpy.ops.object.shape_key_remove()

    @property
    def keys(self):
        return list(self.shape_keys.key_blocks.keys())

    def create_shape_key(self, key="Basis"):
        # Creates a new shape key with the given name
        self.activate()
        bpy.ops.object.shape_key_add(from_mix=False)
        self.obj.active_shape_key.name = key
        self.shape_keys.key_blocks.update()
    
    def get_key(self, key, new_ok=False):
        # Returns the index and name of a shape key, creating a new one if `new_ok` is True
        mesh_keys = self.shape_keys.key_blocks.keys()
        if isinstance(key, str) and key not in mesh_keys:
            if new_ok is True:
                self.create_shape_key(key)
                mesh_keys = self.shape_keys.key_blocks.keys()
            else: 
                print(f"[!] ShapeKey '{key}' queried but not found")
        idx = key if isinstance(key, int) else mesh_keys.index(key)
        key = key if isinstance(key, str) else mesh_keys[key]
        return idx, key

    def set_shape_key(self, key=0, value=1, new_ok=False, visualize=False):
        # Sets the active shape key to the one specified by key (name or index), and sets its value to 1
        self.activate()
        for block in self.shape_keys.key_blocks:
            block.value = 0
        idx, key = self.get_key(key, new_ok=new_ok)
        self.shape_keys.animation_data_clear()
        self.shape_keys.key_blocks[key].value = value
        self.obj.active_shape_key_index = idx

    def get_vertex_positions(self, key=0, revert=False):
        self.set_shape_key(key)
        out = np.array([self.obj.active_shape_key.data[v.index].co for v in self.vertices])
        if revert: self.set_shape_key(0)
        return out

    def update_shape_key(self, key, new_verts, offset=False, revert=False):
        if key == 'Basis': return
        # Updates the shape key specified by `key` with new vertex positions
        self.set_shape_key(key, new_ok=True)
        # self.set_vertex_positions(new_verts)
        for v in self.vertices:
            self.obj.active_shape_key.data[v.index].co = mu.Vector(new_verts[v.index]) + offset * v.co 
        if revert: self.set_shape_key(0)


class SpectralMesh(BMeshObject):
    def __init__(self, obj=None, mesh=None):
        super().__init__(obj=obj, mesh=mesh)

    def get_spectral_transforms(self, kdims):
        self.eigvs = get_gl_eigvs(self, kdims, struct='verts')
        def fwd(x): return self.eigvs.transpose() @ x
        def inv(x): return self.eigvs @ x
        return fwd, inv

    # def connected_components(self):
    #     # Computes the connected components of the mesh using a breadth-first search (BFS)
    #     visited = set()
    #     components = []
    #     vert_lists = []
    #     for v in self.bm.verts:
    #         if v not in visited:
    #             # Found an unvisited vertex, start a new connected component
    #             component_bm = bmesh.new()
    #             component_bm.verts.new(v.co)
    #             # Queue for BFS traversal
    #             queue = deque([v])
    #             visited.add(v)
    #             vert_list = [v.index]
    #             component_bm.verts.ensure_lookup_table()
    #             vert_mapping = {v.index: component_bm.verts[0]}
    #             edge_set = set()
    #             while queue:
    #                 vertex = queue.popleft()
    #                 for edge in vertex.link_edges:
    #                     neighbor = edge.other_vert(vertex)
    #                     if neighbor not in visited:
    #                         visited.add(neighbor)
    #                         new_neighbor = component_bm.verts.new(neighbor.co)
    #                         vert_list += [neighbor.index]
    #                         vert_mapping[neighbor.index] = new_neighbor
    #                         queue += [neighbor]
    #                     # If both vertices of the edge are in the component, add the edge to component_bm
    #                     if (vertex.index, neighbor.index) not in edge_set and (neighbor.index, vertex.index) not in edge_set:
    #                         component_bm.edges.new((vert_mapping[vertex.index], vert_mapping[neighbor.index]))
    #                         edge_set.add((vertex.index, neighbor.index))
    #             components += [BMeshObject(mesh=component_bm)]
    #             vert_lists += [vert_list]
    #     return components, vert_lists

    # def create_connected_mesh(self):
    #     # Creates a connected mesh by connecting disconnected components using the closest pair of vertices
    #     components, vert_lists = self.connected_components()
    #     if len(components) <= 1: return
    #     self._added_edges = []
    #     for i in range(len(components) - 1):
    #         obj1, obj2 = components[i], components[i + 1]
    #         vtl1, vtl2 = vert_lists[i], vert_lists[i + 1]
    #         # Find the closest pair of vertices between two components
    #         min_dist = float('inf')
    #         min_pair = (None, None)
    #         for v1 in obj1.bm.verts:
    #             for v2 in obj2.bm.verts:
    #                 dist = (v1.co - v2.co).length_squared
    #                 if dist < min_dist:
    #                     min_dist = dist
    #                     min_pair = (self.bm.verts[vtl1[v1.index]], self.bm.verts[vtl2[v2.index]])
    #         # Link the closest pair of vertices with an edge
    #         new_edge = self.bm.edges.new(min_pair)
    #         self._added_edges += [new_edge]
    #     self.bm.edges.ensure_lookup_table()
    #     self.update_object()

    # def revert_connected_mesh(self):
    #     # Reverts the connected mesh to its original state by removing the added edges
    #     if hasattr(self, '_added_edges'):
    #         for edge in self._added_edges:
    #             self.bm.edges.remove(edge)
    #         self.bm.edges.ensure_lookup_table()
    #         del self._added_edges
    #     self.update_object()


class ColorizableMesh(BMeshObject):
    def __init__(self, obj=None, mesh=None):
        super().__init__(obj=obj, mesh=mesh)

    @staticmethod
    def scale01(arr): 
        return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))

    def color_vertices(self, diff, scale=False):
        # Colors the vertices of the mesh based on the difference array `diff`
        vert_colors = self.obj.data.vertex_colors
        while(vert_colors):
            vert_colors.remove(self.obj.data.vertex_colors[0])
        color_layer = vert_colors.new()
        colors = diff if not scale else ColorizableMesh.scale01(diff)
        
        ## Loop over the vertices and assign colors based on the difference array
        for i, l in enumerate(self.obj.data.loops):
            ## Set the color for each vertex
            v_idx = l.vertex_index
            color = [0.5 + colors[v_idx][j] * 0.5 for j in range(3)]
            color_layer.data[i].color = color + [1.] ## rgb + a

    def partition_mesh(self, centroids, num_patches=None, get_labels=True, get_centroids=True):
        # Partitions the mesh into patches using k-means clustering, based on the given centroids
        try:
            import numpy as np
            import scipy
            from scipy.cluster.vq import vq, kmeans2, whiten
        except: 
            raise Exception("partition_mesh requires scipy")
        
        if num_patches is None: 
            num_patches = 50 #len(mesh.faces)
        k_centroids, knn_labels = kmeans2(centroids, num_patches, 10)

        patches = [[] for _ in range(num_patches)]
        for i, f in enumerate(self.obj.data.vertices):
            patch_idx = knn_labels[i]
            patches[patch_idx].append(i)
            
        knn_centroids = np.take(k_centroids*np.random.uniform(size=k_centroids.shape), knn_labels, axis=0)
        
        return [patches] \
            + get_labels * [knn_labels] \
            + get_centroids * [knn_centroids]
    
class BMesh(ShapeKeyMesh, SpectralMesh, ColorizableMesh):
    pass