import math
import numpy as np
import scipy.linalg
import scipy.cluster
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from mathutils import Vector

# Code from https://github.com/kugelrund/mesh_segmentation
# Changed mesh segmentation to work with trimesh library


def _face_center(mesh, face):
    """Computes the coordinates of the center of the given face"""
    center = Vector()
    for vert in face.verts:
        center += mesh.vertices[vert.index].co
    return center/len(face.verts)


def _geodesic_distance(mesh, face1, face2, edge):
    """Computes the geodesic distance over the given edge between
    the two adjacent faces face1 and face2"""
    edge_center = (mesh.vertices[edge[0]].co + mesh.vertices[edge[1]].co)/2
    return np.linalg.norm(edge_center - _face_center(mesh, face1)) + \
        np.linalg.norm(edge_center - _face_center(mesh, face2))


def _angular_distance(mesh, face1, face2, face1_ind, face2_ind):
    """Computes the angular distance of the given adjacent faces"""
    norm_1 = mesh.faces[face1_ind].normal
    norm_2 = mesh.faces[face2_ind].normal
    proj = np.dot(norm_1, norm_2)
    angle = np.arccos(proj)
    angular_distance = (1 - math.cos(angle))
    if(np.dot(_face_center(mesh, face2), _face_center(mesh, face1)) < 0):
        angular_distance *= eta
    return angular_distance

def _create_face_distance_matrix(mesh, delta):
    """Creates the matrix of the angular and geodesic distances
    between all adjacent faces. The i,j-th entry of the returned
    matrices contains the distance between the i-th and j-th face.
    """
    # helping vectors to create sparse matrix later on
    row_indices = []
    col_indices = []
    Gval = []  # values for matrix of angular distances
    Aval = []  # values for matrix of geodesic distances
    # iterate adjacent faces and calculate distances
    parts = mesh.faces
    l = len(parts)
    for edge in mesh.edges:
        adj = edge.link_faces
        if len(adj) != 2: continue
        i, j = adj[0].index, adj[1].index
        cmp1, cmp2 = parts[i], parts[j]
        Gtemp = _geodesic_distance(mesh, cmp1, cmp2, [e.index for e in edge.verts])
        Atemp = _angular_distance(mesh, cmp1, cmp2, i, j)
        Gval += [Gtemp, Gtemp]
        Aval += [Atemp, Atemp]
        row_indices += [i, j]
        col_indices += [j, i] # add symmetric entry

    Gval = np.array(Gval)
    Gval /= np.mean(Gval)
    Aval = np.array(Aval) 
    Aval /= np.mean(Aval)
    values = delta * Gval + (1.0 - delta) * Aval
    distance_matrix = scipy.sparse.csr_matrix((values, (row_indices, col_indices)), shape=(l, l))
    return distance_matrix

def _create_vert_distance_matrix(mesh):
    # helping vectors to create sparse matrix later on
    row_indices = []
    col_indices = []
    Dval = []  # values for matrix of edge length
    # iterate adjacent faces and calculate distances
    parts = mesh.verts
    l = len(parts)
    for edge in mesh.edges:
        adj = edge.verts
        if len(adj) != 2: print("[!] Non-exclusive adjacency in dist matrix")
        i, j = adj[0].index, adj[1].index
        Dtemp = edge.calc_length()
        Dval += [Dtemp, Dtemp]
        row_indices += [i, j]
        col_indices += [j, i] # add symmetric entry

    values = np.array(Dval) / np.mean(Dval)
    distance_matrix = scipy.sparse.csr_matrix((values, (row_indices, col_indices)), shape=(l, l))
    return distance_matrix


def _create_distance_matrix(mesh, delta=0.5, struct='faces'):
    """Creates distance matrix"""
    if struct == 'faces':
        return _create_face_distance_matrix(mesh, delta), len(mesh.faces)
    elif struct == 'verts':
        return _create_vert_distance_matrix(mesh), len(mesh.verts)

def _create_affinity_matrix(mesh, struct='faces'):
    """Create the adjacency matrix of the given mesh"""

    print(f"segmentation: Creating {struct} distance matrices...")
    distance_matrix, length = _create_distance_matrix(mesh, struct=struct)

    print("segmentation: Finding shortest paths between all faces...")
    # for each non adjacent pair of faces find shortest path of adjacent faces
    W = scipy.sparse.csgraph.dijkstra(distance_matrix)
    inf_indices = np.where(np.isinf(W))
    W[inf_indices] = 0

    print("segmentation: Creating affinity matrix...")
    # change distance entries to similarities
    sigma = W.sum()/(length ** 2)
    den = 2 * (sigma ** 2)
    W = np.exp(-W/den)
    W[inf_indices] = 0
    np.fill_diagonal(W, 1)

    return W


def _initial_guess(Q, k):
    """Computes an initial guess for the cluster-centers
    Chooses indices of the observations with the least association to each
    other in a greedy manner. Q is the association matrix of the observations.
    """

    # choose the pair of indices with the lowest association to each other
    min_indices = np.unravel_index(np.argmin(Q), Q.shape)

    chosen = [min_indices[0], min_indices[1]]
    for _ in range(2, k):
        # Take the maximum of the associations to the already chosen indices for
        # every index. The index with the lowest result in that therefore is the
        # least similar to the already chosen pivots so we take it.
        # Note that we will never get an index that was already chosen because
        # an index always has the highest possible association 1.0 to itself
        new_index = np.argmin(np.max(Q[chosen, :], axis=0))
        chosen.append(new_index)

    return chosen

def get_gl_eigvs(mesh, k, ev_method='dense', struct='faces', normalized=False, verbose=False):
    '''
    Perform spectral decomposition by decomposing Graph Laplacian
    https://www.cs.cmu.edu/~epxing/Class/10701-08s/Lecture/lecture23-Spectral.pdf
    '''
    # affinity matrix
    W = _create_affinity_matrix(mesh, struct)

    if verbose: print("segmentation: Calculating graph laplacian...")

    D = W.sum(1)
    # degree matrix -> graph laplacian
    Dsqrt = np.sqrt(np.reciprocal(D))  ## RuntimeWarning: invalid value encountered in reciprocal
    L = ((W * Dsqrt).transpose() * Dsqrt).transpose()

    if verbose: print("segmentation: Calculating eigenvectors...")
    # get eigenvectors
    if ev_method == 'dense':
        _, V = scipy.linalg.eigh(L, eigvals=(L.shape[0] - k, L.shape[0] - 1))
    else:
        _, V = scipy.sparse.linalg.eigsh(L, k)
    return V

def get_spectral_coeffs(verts, V):
    return V.transpose() @ verts

def from_spectral_coeffs(coeffs, V):
    return coeffs @ V

def segment_mesh(mesh, k, coefficients, action, ev_method, kmeans_init):
    """
    Segments the given mesh into k clusters and performs the given
    action for each cluster
    """

    # set coefficients
    global delta
    global eta
    delta, eta = coefficients

    # affinity matrix
    V = get_gl_eigvs(mesh, k, ev_method, True)

    # normalize each row to unit length
    V /= np.linalg.norm(V, axis=1)[:, None]

    if kmeans_init == 'kmeans++':
        print("segmentation: Applying kmeans...")
        _, idx = scipy.cluster.vq.kmeans2(V, k, minit='++', iter=50)
    else:
        print("segmentation: Preparing kmeans...")
        # compute association matrix
        Q = V.dot(V.transpose())
        # compute initial guess for clustering
        initial_centroids = _initial_guess(Q, k)

        print("segmentation: Applying kmeans...")
        _, idx = scipy.cluster.vq.kmeans2(V, V[initial_centroids, :], iter=50)

    print("segmentation: Done clustering!")
    # perform action with the clustering result
    return idx
