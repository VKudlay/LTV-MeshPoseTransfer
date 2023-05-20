import trimesh
import trimesh.viewer
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.cluster
import scipy.optimize
import sklearn.cluster
import ctypes
import copy
import os
import sys
from sksparse.cholmod import cholesky
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, vstack, hstack

# Get the current DPI scaling value
if sys.platform == "win32":
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
global num_patches
num_patches = 25
global eps

eps = 0.0000001


# From https://stackoverflow.com/questions/54616049/converting-a-rotation-matrix-to-euler-angles-and-back-special-case
def rot2eul(R):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1]/np.cos(beta), R[2, 2]/np.cos(beta))
    gamma = np.arctan2(R[1, 0]/np.cos(beta), R[0, 0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def eul2rot(theta):

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])
                   * np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R


def interpolate(coord_1, coord_2, step_length, patch_adj, patch_map, patches, rest):
    '''
    Given two poses in patch LRI coordinates, interpolate between the two poses
    at step_length increments. Files are saved in obj format
    '''
    num_steps = int(1.0 / step_length)
    for step in range(num_steps):
        w_1 = step_length*step
        w_2 = 1 - w_1
        coord = interpolation([coord_1, coord_2], [w_1, w_2])
        vertices = decode(coord, patch_adj, patch_map,
                          patches, copy.deepcopy(rest))
        out_mesh = copy.deepcopy(rest)
        out_mesh.vertices = vertices
        out_mesh.export("out_" + str(w_1) + "_" + str(w_2) + ".obj")


def encode_meshes(rest, meshes, patches):
    coords = []
    for ind, mesh in enumerate(meshes):
        coord_i, G_matrices, patch_adj, patches, patch_map = encode(
            rest, meshes[ind], patches)
        coords.append(coord_i)
    return coords


def load_meshes(projind,path1="face1", path2="face2",n_poses=10):
    '''
    Load each mesh and store in an array
    '''
    global num_patches
    meshes = []
    rest = as_mesh(trimesh.load_mesh(os.path.join(path1,f"{path1}-reference.obj")))
    rest.merge_vertices(merge_tex=True, merge_norm=True)
    for i in range(1,n_poses+1):
        pose = as_mesh(trimesh.load_mesh(os.path.join(path1,f"{path1}-{i:02}.obj")))
        pose.merge_vertices(merge_tex=True, merge_norm=True)
        meshes.append(pose)
    Ds = example_deformations(rest, meshes)
    Rs = np.zeros((len(meshes), rest.faces.shape[0], 9))
    for ind, Di in enumerate(Ds):
        Ri = [scipy.linalg.polar(D)[0].flatten() for D in Di]
        Rs[ind] = Ri
    R_new = np.reshape(Rs, (Rs.shape[1], Rs.shape[0]*9))
    patches = sklearn.cluster.KMeans(n_clusters=num_patches).fit(R_new).labels_
    num_patches = max(patches) + 1
    meshes = [meshes[projind]]+meshes[:projind]+meshes[projind+1:]
    coords = encode_meshes(rest, meshes, patches)
    weights = projection(coords[1: len(coords)], coords[0])
    print(weights)
    meshes_2 = []
    rest = as_mesh(trimesh.load_mesh(os.path.join(path2,f"{path2}-reference.obj")))
    rest.merge_vertices(merge_tex=True, merge_norm=True)
    for i in range(1,n_poses+1):
        pose = as_mesh(trimesh.load_mesh(os.path.join(path2,f"{path2}-{i:02}.obj")))
        pose.merge_vertices(merge_tex=True, merge_norm=True)
        meshes_2.append(pose)
    Ds = example_deformations(rest, meshes_2)
    Rs = np.zeros((len(meshes_2), rest.faces.shape[0], 9))
    for ind, Di in enumerate(Ds):
        Ri = [scipy.linalg.polar(D)[0].flatten() for D in Di]
        Rs[ind] = Ri
    R_new = np.reshape(Rs, (Rs.shape[1], Rs.shape[0]*9))
    patches = sklearn.cluster.KMeans(n_clusters=num_patches).fit(R_new).labels_
    num_patches = max(patches) + 1
    coords = encode_meshes(rest, meshes_2, patches)
    coord = interpolation(coords[1: len(coords)], weights)
    patch_adj, patch_map = patch_adjacency(meshes_2[0], patches)
    vertices = decode(coord, patch_adj, patch_map, patches, rest)
    rest.vertices = vertices
    return rest


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def run_interpolation():
    '''
    Input meshes and interpolate between them!
    '''
    global num_patches
    meshes = []
    
    rest = as_mesh(trimesh.load_mesh("lion/lion-reference.obj"))
    rest.merge_vertices(merge_tex=True, merge_norm=True)
    print(len(rest.vertices))
    pose = as_mesh(trimesh.load_mesh("lion/lion-01.obj"))
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    print(len(pose.vertices))
    pose = as_mesh(trimesh.load_mesh("lion/lion-02.obj"))
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    print(len(pose.vertices))
    """
    rest = trimesh.load_mesh("lion/lion-reference.obj")
    rest.merge_vertices(merge_tex=True, merge_norm=True)
    pose = trimesh.load_mesh("lion/lion-01.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    pose = trimesh.load_mesh("lion/lion-02.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    pose = trimesh.load_mesh("lion/lion-03.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    pose = trimesh.load_mesh("lion/lion-04.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    """
    
    Ds = example_deformations(rest, meshes)
    Rs = np.zeros((len(meshes), rest.faces.shape[0], 9))
    for ind, Di in enumerate(Ds):
        Ri = [scipy.linalg.polar(D)[0].flatten() for D in Di]
        Rs[ind] = Ri
    R_new = np.reshape(Rs, (Rs.shape[1], Rs.shape[0]*9))
    patches = sklearn.cluster.KMeans(n_clusters=num_patches).fit(R_new).labels_
    num_patches = max(patches) + 1
    coord_1, G_matrices, patch_adj, patches, patch_map = encode(
        rest, meshes[0], patches)
    coord_2, G_matrices, patch_adj, patches, patch_map = encode(
        rest, meshes[1], patches)
    interpolate(coord_1, coord_2, 0.05, patch_adj, patch_map, patches, rest)


def interpolation(poses, weights):
    '''
    Given p example poses and a set of weights (which sum to 1),
    compute affine combination of the poses
    '''
    sum = np.array([np.zeros_like(arr) for arr in poses[0]],dtype=object)
    for pose, weight in zip(poses, weights):
        sum += np.array([arr*weight for arr in pose],dtype=object)
    return sum


def flatten(arr):
    flat = []
    for i in arr:
        i = np.array(i).flatten()
        flat.append(i)
    flat = np.concatenate(flat)
    return flat


def projection(poses, pose_q):
    '''
    Given p example poses and some other pose, compute weights
    '''
    pose_vecs = []
    for i, pose in enumerate(poses):
        flat = flatten(pose)
        pose_vecs.append(flat)
    pose_q_vec = flatten(pose_q)
    x_1 = pose_vecs[0]
    left = np.zeros((x_1.shape[0], len(pose_vecs) - 1))
    for i in range(len(pose_vecs) - 1):
        x_next = pose_vecs[i + 1]
        left[:, i] = x_next - x_1
    left_inv = np.linalg.pinv(left)
    right = np.transpose(pose_q_vec - x_1)
    weights = left_inv@right
    w_1 = 1 - np.sum(weights)
    weights = np.insert(weights, 0, w_1, axis=0)
    return weights


def patch_adjacency(mesh, patches):
    '''
    Finds which patches in the mesh are adjacent to each other
    '''
    num_patches = max(patches) + 1
    face_adj = mesh.face_adjacency
    patch_adj = set()
    for f_adj in face_adj:
        patch_1 = patches[f_adj[0]]
        patch_2 = patches[f_adj[1]]
        if(patch_1 != patch_2):
            patch_adj.add((patch_1, patch_2))
            patch_adj.add((patch_2, patch_1))
    alladjs = [[] for _ in range(num_patches)]
    for i, j in patch_adj:
        alladjs[i].append(j)
    empties = [i for i in range(num_patches) if not alladjs[i]]
    patch_adj = [list(tup) for tup in patch_adj]
    old_vals = copy.deepcopy(patch_adj)
    patch_map = {}
    for empty in empties:
        for i in range(len(patch_adj)):
            if empty < old_vals[i][0]:
                patch_adj[i][0] -= 1
            if empty < old_vals[i][1]:
                patch_adj[i][1] -= 1
    for i in range(len(patch_adj)):
        patch_map[old_vals[i][0]] = patch_adj[i][0]
        patch_map[old_vals[i][1]] = patch_adj[i][1]
    patch_adj = {tuple(arr) for arr in patch_adj}
    return patch_adj, patch_map


def deformation_gradient(rest_pose: trimesh.Trimesh, current_pose: trimesh.Trimesh):
    '''
    Given a resting pose and a current pose, computes the deformation gradient
    for each face and stores them in an array
    '''
    rest_faces = rest_pose.faces
    current_faces = current_pose.faces
    rest_vertices = rest_pose.vertices
    current_vertices = current_pose.vertices
    v1 = current_vertices[rest_faces[:, 0]]
    v2 = current_vertices[rest_faces[:, 1]]
    v3 = current_vertices[rest_faces[:, 2]]
    n = np.cross((v2 - v1), (v3 - v1))
    n /= np.sqrt(np.linalg.norm(n, axis=1)[:, np.newaxis])
    v1_p = rest_vertices[rest_faces[:, 0]]
    v2_p = rest_vertices[rest_faces[:, 1]]
    v3_p = rest_vertices[rest_faces[:, 2]]
    n_p = np.cross((v2_p - v1_p), (v3_p - v1_p))
    n_p /= np.sqrt(np.linalg.norm(n_p, axis=1)[:, np.newaxis])
    D_left = np.column_stack((v2 - v1, v3 - v1, n))
    D_right = np.column_stack((v2_p - v1_p, v3_p - v1_p, n_p))
    D_map = np.zeros((rest_faces.shape[0], 3, 3))
    for i in range(rest_faces.shape[0]):
        D_left_i = D_left[i]
        D_right_i = D_right[i]
        D_left_i = np.reshape(D_left_i, (3, 3)).T
        D_right_i = np.reshape(D_right_i, (3, 3)).T
        D = D_left_i@np.linalg.inv(D_right_i)
        D_map[i] = D
    return D_map


def decode(coord, patch_adj, patch_map, patches, rest):
    '''
    Given a mesh in patch LRI coordinates, restores the mesh vertices
    '''
    coord[0] = coord[0]*10
    for ind, G_inv_G in enumerate(coord[3]):
        i = ind % num_patches
        j = int(ind / num_patches)
        G_inv_G = G_inv_G / ((np.count_nonzero(patches == i) *
                              np.count_nonzero(patches == j))**0.25)
        coord[3][ind] = G_inv_G
    Gs = construct_G(coord[3], patch_adj, patches)
    Ds = reconstruct_deformation(
        Gs, coord[0], coord[4], patch_map, patches)
    v_prime = solve_vertices(Ds, rest)
    v_out = transform_vertices(
        v_prime, rest, rest, coord[2], coord[1])
    return v_out


def encode(rest_pose, current_pose, patches):
    '''
    Encodes current_pose into patch LRI coordinates
    '''
    num_patches = max(patches) + 1
    patch_adj, patch_map = patch_adjacency(rest_pose, patches)
    Q_matrices = np.zeros((len(patches), 3))
    S_matrices = np.zeros((len(patches), 3, 3))
    Q_bar = np.zeros(3)
    v_bar = np.zeros(3)
    for vert in current_pose.vertices:
        v_bar += vert
    v_bar = v_bar / rest_pose.vertices.shape[0]
    sum_q_i = np.zeros((num_patches, 3))
    num_in_patch = np.zeros(num_patches)
    D_map = deformation_gradient(rest_pose, current_pose)
    # D matrices: 0.70, -0.58, 0.07, -0.11, 0.519, 0.06, 0.0027, 0.81
    for i in range(len(patches)):
        D = D_map[i]
        Q, S = scipy.linalg.polar(D)
        Q = rot2eul(Q)
        sum_q_i[patches[i]] += Q
        num_in_patch[patches[i]] += 1
        Q_bar += Q
        Q_matrices[i] = Q
        S_matrices[i] = S
    Q_bar = Q_bar / rest_pose.faces.shape[0]
    G_matrices = np.zeros((num_patches, 3, 3))
    for i in range(num_patches):
        G_matrices[i] = eul2rot(sum_q_i[i] / num_in_patch[i])
    G_adj = np.zeros((num_patches, num_patches, 3))
    for ind, adj in enumerate(patch_adj):
        G_inv_G = np.linalg.inv(G_matrices[adj[1]])@G_matrices[adj[0]]
        G_inv_G = rot2eul(G_inv_G)
        G_inv_G = ((np.count_nonzero(patches == adj[0]) *
                   np.count_nonzero(patches == adj[1]))**0.25)*G_inv_G
        G_adj[adj[0]][adj[1]] = G_inv_G
    G_inv_Q = np.zeros((len(patches), 3))
    G_inv = np.linalg.inv(G_matrices)
    for ind, Q in enumerate(Q_matrices):
        patch = patches[ind]
        G = G_matrices[patch]
        G_inv_Q[ind] = rot2eul(G_inv[patch]@eul2rot(Q))
    # for i in range(len(G_adj)):
    #     for j in range(len(G_adj[i])):
    #         G_adj[i][j] = G_adj[i][j] / ((np.count_nonzero(patches == i) *
    #                                       np.count_nonzero(patches == j))**0.25)
    coords = []
    coords.append(0.1*S_matrices)
    coords.append(eps*v_bar)
    coords.append(eps*Q_bar)
    coords.append(G_adj)
    coords.append(G_inv_Q)
    return coords, G_matrices, patch_adj, patches, patch_map


def construct_G(G_ij, patch_adj, patches):
    '''
    Step 1 of reconstruction. Given G_ij's in the coordinate vector,
    solves a linear system to return G_i's.
    '''
    num_patches = max(patches) + 1
    G_new = np.zeros((num_patches, num_patches, 3, 3))
    for i in range(G_ij.shape[0]):
        for j in range(G_ij.shape[1]):
            G_new[i][j] = eul2rot(G_ij[i][j])
    G_ij = G_new
    b = np.zeros(((num_patches-1)*9))
    A = np.zeros((num_patches*9, num_patches*9))
    for i, j in patch_adj:
        A[i*9+0, j*9+0] += -2*G_ij[i][j][0, 0]
        A[i*9+0, j*9+1] += -2*G_ij[i][j][0, 1]
        A[i*9+0, j*9+2] += -2*G_ij[i][j][0, 2]
        A[i*9+1, j*9+0] += -2*G_ij[i][j][1, 0]
        A[i*9+1, j*9+1] += -2*G_ij[i][j][1, 1]
        A[i*9+1, j*9+2] += -2*G_ij[i][j][1, 2]
        A[i*9+2, j*9+0] += -2*G_ij[i][j][2, 0]
        A[i*9+2, j*9+1] += -2*G_ij[i][j][2, 1]
        A[i*9+2, j*9+2] += -2*G_ij[i][j][2, 2]
        A[i*9+3, j*9+3] += -2*G_ij[i][j][0, 0]
        A[i*9+3, j*9+4] += -2*G_ij[i][j][0, 1]
        A[i*9+3, j*9+5] += -2*G_ij[i][j][0, 2]
        A[i*9+4, j*9+3] += -2*G_ij[i][j][1, 0]
        A[i*9+4, j*9+4] += -2*G_ij[i][j][1, 1]
        A[i*9+4, j*9+5] += -2*G_ij[i][j][1, 2]
        A[i*9+5, j*9+3] += -2*G_ij[i][j][2, 0]
        A[i*9+5, j*9+4] += -2*G_ij[i][j][2, 1]
        A[i*9+5, j*9+5] += -2*G_ij[i][j][2, 2]
        A[i*9+6, j*9+6] += -2*G_ij[i][j][0, 0]
        A[i*9+6, j*9+7] += -2*G_ij[i][j][0, 1]
        A[i*9+6, j*9+8] += -2*G_ij[i][j][0, 2]
        A[i*9+7, j*9+6] += -2*G_ij[i][j][1, 0]
        A[i*9+7, j*9+7] += -2*G_ij[i][j][1, 1]
        A[i*9+7, j*9+8] += -2*G_ij[i][j][1, 2]
        A[i*9+8, j*9+6] += -2*G_ij[i][j][2, 0]
        A[i*9+8, j*9+7] += -2*G_ij[i][j][2, 1]
        A[i*9+8, j*9+8] += -2*G_ij[i][j][2, 2]
        A[j*9+0, i*9+0] += -2*G_ij[i][j][0, 0]
        A[j*9+0, i*9+1] += -2*G_ij[i][j][1, 0]
        A[j*9+0, i*9+2] += -2*G_ij[i][j][2, 0]
        A[j*9+1, i*9+0] += -2*G_ij[i][j][0, 1]
        A[j*9+1, i*9+1] += -2*G_ij[i][j][1, 1]
        A[j*9+1, i*9+2] += -2*G_ij[i][j][2, 1]
        A[j*9+2, i*9+0] += -2*G_ij[i][j][0, 2]
        A[j*9+2, i*9+1] += -2*G_ij[i][j][1, 2]
        A[j*9+2, i*9+2] += -2*G_ij[i][j][2, 2]
        A[j*9+3, i*9+3] += -2*G_ij[i][j][0, 0]
        A[j*9+3, i*9+4] += -2*G_ij[i][j][1, 0]
        A[j*9+3, i*9+5] += -2*G_ij[i][j][2, 0]
        A[j*9+4, i*9+3] += -2*G_ij[i][j][0, 1]
        A[j*9+4, i*9+4] += -2*G_ij[i][j][1, 1]
        A[j*9+4, i*9+5] += -2*G_ij[i][j][2, 1]
        A[j*9+5, i*9+3] += -2*G_ij[i][j][0, 2]
        A[j*9+5, i*9+4] += -2*G_ij[i][j][1, 2]
        A[j*9+5, i*9+5] += -2*G_ij[i][j][2, 2]
        A[j*9+6, i*9+6] += -2*G_ij[i][j][0, 0]
        A[j*9+6, i*9+7] += -2*G_ij[i][j][1, 0]
        A[j*9+6, i*9+8] += -2*G_ij[i][j][2, 0]
        A[j*9+7, i*9+6] += -2*G_ij[i][j][0, 1]
        A[j*9+7, i*9+7] += -2*G_ij[i][j][1, 1]
        A[j*9+7, i*9+8] += -2*G_ij[i][j][2, 1]
        A[j*9+8, i*9+6] += -2*G_ij[i][j][0, 2]
        A[j*9+8, i*9+7] += -2*G_ij[i][j][1, 2]
        A[j*9+8, i*9+8] += -2*G_ij[i][j][2, 2]
        A[j*9+0, j*9+0] += 2*G_ij[i][j][0, 0]**2 + 2 * \
            G_ij[i][j][1, 0]**2 + 2*G_ij[i][j][2, 0]**2
        A[j*9+0, j*9+1] += 2*G_ij[i][j][0, 0]*G_ij[i][j][0, 1] + 2 * \
            G_ij[i][j][1, 0]*G_ij[i][j][1, 1] + \
            2*G_ij[i][j][2, 0]*G_ij[i][j][2, 1]
        A[j*9+0, j*9+2] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 0] + 2 * \
            G_ij[i][j][0, 0]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 0]*G_ij[i][j][1, 2]
        A[j*9+1, j*9+0] += 2*G_ij[i][j][0, 0]*G_ij[i][j][0, 1] + 2 * \
            G_ij[i][j][1, 0]*G_ij[i][j][1, 1] + \
            2*G_ij[i][j][2, 0]*G_ij[i][j][2, 1]
        A[j*9+1, j*9+1] += 2*G_ij[i][j][0, 1]**2 + 2 * \
            G_ij[i][j][1, 1]**2 + 2*G_ij[i][j][2, 1]**2
        A[j*9+1, j*9+2] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 1] + 2 * \
            G_ij[i][j][0, 1]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 1]*G_ij[i][j][1, 2]
        A[j*9+2, j*9+0] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 0] + 2 * \
            G_ij[i][j][0, 0]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 0]*G_ij[i][j][1, 2]
        A[j*9+2, j*9+1] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 1] + 2 * \
            G_ij[i][j][0, 1]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 1]*G_ij[i][j][1, 2]
        A[j*9+2, j*9+2] += 2*G_ij[i][j][2, 2]**2 + 2 * \
            G_ij[i][j][0, 2]**2 + 2*G_ij[i][j][1, 2]**2
        A[j*9+3, j*9+3] += 2*G_ij[i][j][0, 0]**2 + 2 * \
            G_ij[i][j][1, 0]**2 + 2*G_ij[i][j][2, 0]**2
        A[j*9+3, j*9+4] += 2*G_ij[i][j][0, 0]*G_ij[i][j][0, 1] + 2 * \
            G_ij[i][j][1, 0]*G_ij[i][j][1, 1] + \
            2*G_ij[i][j][2, 0]*G_ij[i][j][2, 1]
        A[j*9+3, j*9+5] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 0] + 2 * \
            G_ij[i][j][0, 0]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 0]*G_ij[i][j][1, 2]
        A[j*9+4, j*9+3] += 2*G_ij[i][j][0, 0]*G_ij[i][j][0, 1] + 2 * \
            G_ij[i][j][1, 0]*G_ij[i][j][1, 1] + \
            2*G_ij[i][j][2, 0]*G_ij[i][j][2, 1]
        A[j*9+4, j*9+4] += 2*G_ij[i][j][0, 1]**2 + 2 * \
            G_ij[i][j][1, 1]**2 + 2*G_ij[i][j][2, 1]**2
        A[j*9+4, j*9+5] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 1] + 2 * \
            G_ij[i][j][0, 1]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 1]*G_ij[i][j][1, 2]
        A[j*9+5, j*9+3] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 0] + 2 * \
            G_ij[i][j][0, 0]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 0]*G_ij[i][j][1, 2]
        A[j*9+5, j*9+4] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 1] + 2 * \
            G_ij[i][j][0, 1]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 1]*G_ij[i][j][1, 2]
        A[j*9+5, j*9+5] += 2*G_ij[i][j][2, 2]**2 + 2 * \
            G_ij[i][j][0, 2]**2 + 2*G_ij[i][j][1, 2]**2
        A[j*9+6, j*9+6] += 2*G_ij[i][j][0, 0]**2 + 2 * \
            G_ij[i][j][1, 0]**2 + 2*G_ij[i][j][2, 0]**2
        A[j*9+6, j*9+7] += 2*G_ij[i][j][0, 0]*G_ij[i][j][0, 1] + 2 * \
            G_ij[i][j][1, 0]*G_ij[i][j][1, 1] + \
            2*G_ij[i][j][2, 0]*G_ij[i][j][2, 1]
        A[j*9+6, j*9+8] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 0] + 2 * \
            G_ij[i][j][0, 0]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 0]*G_ij[i][j][1, 2]
        A[j*9+7, j*9+6] += 2*G_ij[i][j][0, 0]*G_ij[i][j][0, 1] + 2 * \
            G_ij[i][j][1, 0]*G_ij[i][j][1, 1] + \
            2*G_ij[i][j][2, 0]*G_ij[i][j][2, 1]
        A[j*9+7, j*9+7] += 2*G_ij[i][j][0, 1]**2 + 2 * \
            G_ij[i][j][1, 1]**2 + 2*G_ij[i][j][2, 1]**2
        A[j*9+7, j*9+8] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 1] + 2 * \
            G_ij[i][j][0, 1]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 1]*G_ij[i][j][1, 2]
        A[j*9+8, j*9+6] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 0] + 2 * \
            G_ij[i][j][0, 0]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 0]*G_ij[i][j][1, 2]
        A[j*9+8, j*9+7] += 2*G_ij[i][j][2, 2]*G_ij[i][j][2, 1] + 2 * \
            G_ij[i][j][0, 1]*G_ij[i][j][0, 2] + \
            2*G_ij[i][j][1, 1]*G_ij[i][j][1, 2]
        A[j*9+8, j*9+8] += 2*G_ij[i][j][2, 2]**2 + 2 * \
            G_ij[i][j][0, 2]**2 + 2*G_ij[i][j][1, 2]**2
        for k in range(9):
            A[i*9+k, i*9+k] += 2
    b -= A[0, 9:]+A[4, 9:]+A[8, 9:]
    A = np.delete(np.delete(A, np.arange(9), axis=0), np.arange(9), axis=1)
    c, low = scipy.linalg.cho_factor(A)
    G = scipy.linalg.cho_solve((c, low), b.flatten()).reshape((-1, 3, 3))
    arrs = [np.identity(3)]+[scipy.linalg.polar(arr)[0] for arr in G]
    return arrs


def reconstruct_deformation(Gs, S_matrices, G_inv_Qs, patch_map, patches):
    '''
    Step 2 of reconstruction. Given G_i's, reconstructs the deformation
    gradient between faces in the mesh.
    '''
    num_faces = S_matrices.shape[0]
    deformations = np.zeros((num_faces, 3, 3))
    for face in range(num_faces):
        patch_ind = patch_map[patches[face]]
        Sf = S_matrices[face]
        G_inv_Q = eul2rot(G_inv_Qs[face])
        Qf = Gs[patch_ind]@G_inv_Q
        # Q_matrices: 0.93, -0.36, 0.014, 0.36, 0.93, 0.0343
        D = Qf@Sf
        deformations[face] = D
    return deformations


def solve_vertices(face_deformations, rest: trimesh.Trimesh):
    '''
    Step 3 of reconstruction. Solves for mesh vertices given deformation
    gradients.
    '''
    F = np.zeros((3, 3*rest.faces.shape[0]))
    for face_ind, face in enumerate(rest.faces):
        x_i = rest.vertices[face[0]]
        x_j = rest.vertices[face[1]]
        x_k = rest.vertices[face[2]]
        F[:, face_ind*3] = face_deformations[face_ind]@(x_j - x_i)
        F[:, face_ind*3 + 1] = face_deformations[face_ind]@(x_i - x_k)
        F[:, face_ind*3 + 2] = face_deformations[face_ind]@(x_k - x_j)
    
    num_faces = rest.faces.shape[0]
    num_vertices = rest.vertices.shape[0]

    # Initialize a LIL matrix which allows for efficient row operations
    G = lil_matrix((3*num_faces, num_vertices), dtype=np.float64)

    # Iterate over faces to fill the matrix
    for face_ind, face in enumerate(rest.faces):
        x_i, x_j, x_k = face
        G[face_ind*3, x_i] = -1
        G[face_ind*3, x_j] = 1
        G[face_ind*3 + 1, x_i] = 1
        G[face_ind*3 + 1, x_k] = -1
        G[face_ind*3 + 2, x_j] = -1
        G[face_ind*3 + 2, x_k] = 1

    # Convert LIL matrix to CSC format
    G = G.tocsc()


    Gt_G = G.transpose().dot(G)
    Gt_Ft = G.transpose()@F.transpose()
    remove_ind = np.random.randint(0, rest.vertices.shape[0])
    Gt_G = vstack((Gt_G[:remove_ind, :], Gt_G[remove_ind+1:, :]))
    Gt_G = hstack((Gt_G[:, :remove_ind], Gt_G[:, remove_ind+1:]))
    Gt_Ft = vstack((Gt_Ft[:remove_ind, :], Gt_Ft[remove_ind+1:, :]))
    factor = cholesky(Gt_G)
    x = factor(Gt_Ft).toarray()
    x = np.insert(x, remove_ind, [0, 0, 0], axis=0)
    return x


def transform_vertices(current_vertices, rest_pose, current_pose, Q_bar, v_bar):
    '''
    Step 4 of decoder algorithm. Restored vertices from step 3
    have arbitrary global translation and rotation -- rotates and
    translates vertices back to orientation of input data.
    '''
    Q_bar = eul2rot(Q_bar)
    current_pose.vertices = current_vertices
    D_map = deformation_gradient(rest_pose, current_pose)
    Q_bar_prime = np.zeros((3, 3))
    for i in range(rest_pose.faces.shape[0]):
        D = D_map[i]
        Q, S = scipy.linalg.polar(D)
        Q_bar_prime += Q
    Q_bar_prime = Q_bar_prime / rest_pose.faces.shape[0]
    Q_bar_prime_inv = scipy.linalg.inv(Q_bar_prime)
    v_bar_prime = np.mean(current_vertices, axis=0)
    Q_bar_Q_bar_inv = Q_bar@Q_bar_prime_inv

    right = np.transpose(current_vertices - v_bar_prime)
    current_vertices = np.transpose(Q_bar_Q_bar_inv@right)
    current_vertices = current_vertices + v_bar
    return current_vertices


def example_deformations(rest, poses):
    '''
    Computes deformation gradients between a rest pose and
    several input poses. Used to learn similar deformation gradients
    for creating patches.
    '''
    Ds = np.zeros((len(poses), rest.faces.shape[0], 3, 3))
    for ind, pose in enumerate(poses):
        Di = deformation_gradient(rest, poses[ind])
        Di = Di.reshape(rest.faces.shape[0], 3, 3)
        Ds[ind] = Di
    return Ds

for i in range(10):
    out = load_meshes(i)
    out.export(f"mesh_{i}.obj")
#run_interpolation()
def make_animation():
    global num_patches
    meshes = []
    
    rest = trimesh.load_mesh("/Users/lucafonstad/Downloads/final-project 3/head-poses/head-reference.obj")
    rest.merge_vertices(merge_tex=True, merge_norm=True)
    print(len(rest.vertices))
    pose = trimesh.load_mesh("/Users/lucafonstad/Downloads/final-project 3/head-poses/head-04-grin.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    print(len(pose.vertices))
    pose = trimesh.load_mesh("/Users/lucafonstad/Downloads/final-project 3/head-poses/head-02-cry.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    print(len(pose.vertices))
    """
    rest = trimesh.load_mesh("lion/lion-reference.obj")
    rest.merge_vertices(merge_tex=True, merge_norm=True)
    pose = trimesh.load_mesh("lion/lion-01.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    pose = trimesh.load_mesh("lion/lion-02.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    pose = trimesh.load_mesh("lion/lion-03.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    pose = trimesh.load_mesh("lion/lion-04.obj")
    pose.merge_vertices(merge_tex=True, merge_norm=True)
    meshes.append(pose)
    """
    
    Ds = example_deformations(rest, meshes)
    Rs = np.zeros((len(meshes), rest.faces.shape[0], 9))
    for ind, Di in enumerate(Ds):
        Ri = [scipy.linalg.polar(D)[0].flatten() for D in Di]
        Rs[ind] = Ri
    R_new = np.reshape(Rs, (Rs.shape[1], Rs.shape[0]*9))
    patches = sklearn.cluster.KMeans(n_clusters=num_patches).fit(R_new).labels_
    num_patches = max(patches) + 1
    coord_1, G_matrices, patch_adj, patches, patch_map = encode(
        rest, meshes[0], patches)
    coord_2, G_matrices, patch_adj, patches, patch_map = encode(
        rest, meshes[1], patches)
    coord = interpolation([coord_1, coord_2], [0.5, 0.5])
    vertices = decode(coord, patch_adj, patch_map,
                          patches, copy.deepcopy(rest))
    out_mesh = copy.deepcopy(rest)
    out_mesh.vertices = vertices
    print(len(out_mesh.faces[0]))

