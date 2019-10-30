'''
Created on 29.10.2019

@author: Salma Mozaffari
'''
import numpy as np
import os
import compas
from compas.geometry import normalize_vectors
from compas.geometry import cross_vectors
from compas.geometry import subtract_vectors
from compas.geometry import cross_vectors
from compas.geometry import length_vector
from compas.datastructures import Mesh
from compas_plotters import MeshPlotter
from random import randrange

########################## Geometry-1 ###########################

def orthonormal_vectors(vec_1, vec_2):
    """
    create a set of three orthonormal vectors 
    vec_1 & vec_2: list of three floats
    return a set of the three orthonormal vectors
    """
    vec_cros=cross_vectors(vec_1, vec_2)
    vec_list=normalize_vectors([vec_1, vec_2, vec_cros])

    return vec_list

# print (orthonormal_vectors([1.0,1.0,2.0],[3.0,3.0,2.0]))

########################## Geometry-2 ###########################

def polygon_area(pts_list):
    """
    use the cross product to compute the area of a convex, 2D polygon 
    pts_list: list of ordered polygon 3D points (i.e list of three floats with z=0.0) 
    return the polygon area
    """
    # calculate the centroid of the points
    n = len(pts_list)
    sum_coor=[sum(coor) for coor in zip(*pts_list)]
    c_pt=[coor/ n for coor in sum_coor]
    # caclulate the area
    poly_area=0.0
    for ind in range(len(pts_list)-1):
        vec_1=subtract_vectors(c_pt, pts_list[ind])
        vec_2=subtract_vectors(c_pt, pts_list[ind+1])
        poly_area+=0.5*abs(cross_vectors(vec_1, vec_2)[2])
    vec_1=vec_2
    vec_2=subtract_vectors(pts_list[0], c_pt)
    poly_area+=0.5 *length_vector(cross_vectors(vec_1, vec_2))
    
    return poly_area

# print (polygon_area([[0.0,0.0,0.0],[0.0,2.0,0.0],[2.0,2.0,0.0],[2.0,0.0,0.0]]))

########################## Geometry-3 ###########################

def cross_product(arr_1, arr_2):
    """
    compute the cross products of two arrays of vectors
    arr_1 & arr_2: list of lists (three floats)
    return a list of lists (cross product of the vectors)
    """
    cros_list=[]
    for ind in range(len(arr_1)):
        vec_1=arr_1[ind]
        vec_2=arr_2[ind]
        vec_cros=[vec_1[1]*vec_2[2]-vec_1[2]*vec_2[1], vec_1[2]*vec_2[0] - vec_1[0]*vec_2[2], vec_1[0]*vec_2[1] -vec_1[1]*vec_2[0]]
        cros_list.append(vec_cros)
    
    return cros_list

def cross_product_numpy(arr_1, arr_2):
    """
    define a function for computing the cross products of two arrays of vectors
    arr_1 & arr_2: list of lists (three floats)
    returns an numpy array of cross product of the vectors
    """
    arr_1=np.array(arr_1)
    arr_2=np.array(arr_2)
    arr_cros=np.cross(arr_1, arr_2)

    return arr_cros
   
# print (np.array([[1.0,0.0,3.0],[1.0,2.0,3.0],[2.0,4.0,3.0]]))
# print (cross_product_numpy([[1.0,0.0,3.0],[1.0,2.0,3.0],[2.0,4.0,3.0]], [[1.0,1.0,3.0],[1.0,0.0,3.0],[1.0,2.0,3.0]]))
# print (cross_product([[1.0,0.0,3.0],[1.0,2.0,3.0],[2.0,4.0,3.0]], [[1.0,1.0,3.0],[1.0,0.0,3.0],[1.0,2.0,3.0]]))

########################## Data Structures-1 ###########################

def traverse_boundary_to_boundary(mesh):
    """
    traverse the mesh from boundary to boundary in a "straight" line and visulize the results
    mesh: mesh data structure
    return a list of ordered vertex keys and plot the solution
    """
    bound_keys=mesh.vertices_on_boundary()
    # randomly pick a boundary key
    ind=randrange(len(bound_keys))  
    key=bound_keys[ind]
    pick_keys=[key]

    print(pick_keys)
    #non-corner keys
    if mesh.vertex_degree(key)>2: 
        f_keys=mesh.vertex_faces(key)
        f_vers=[]
        for f_key in f_keys:
            f_vers.append((mesh.face_vertices(f_key)))
        int_keys=set.intersection(*map(set,f_vers))
        next_key=list(int_keys-set(pick_keys))[0]
        pick_keys.append(next_key)        
        
        print (f_vers)
        print (int_keys)
        print (next_key)
        print (pick_keys)

        while next_key not in bound_keys:

            f_keys=mesh.vertex_faces(next_key)
            f_vers=[]
            for f_key in f_keys:
                f_vers.append((mesh.face_vertices(f_key)))
            int_keys=set.intersection(*map(set,f_vers))
            next_key=int_keys-set(pick_keys)
            print (next_key)

            print (f_vers)
            print (int_keys)
            # print (next_key)
            # print (pick_keys)
            break
            # pick_keys.append(next_key)
        

        



        # print mesh.face_vertices(fkey)

    # plotter = MeshPlotter(mesh, figsize=(8, 5))

    # plotter.draw_vertices(text='key', radius=0.01)
    # plotter.draw_edges()
    
    # plotter.show()





mesh = Mesh.from_obj(compas.get('faces.obj'))
traverse_boundary_to_boundary(mesh)
