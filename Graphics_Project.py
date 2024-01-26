from calendar import c
from math import acos, cos, sin
import numpy as np
from matplotlib import pyplot as plt

M = 512
N = 512

def interpolate_color(x1, x2, x, C1, C2):

    # We first initialize a list with 3 values
    values = np.ones(3)

    # I explain the way my interpolate_color works in the report
    # First we check if the y coordinates match, then we interpolate between x values
    if x1[1] == x2[1] :
        if int(x1[0]) == int(x2[0]) :
            return C1
        else:
            for i in range(3) :
                values[i] =abs(int(x[0])-int(x2[0]))/abs(int(x1[0])-int(x2[0]))*C1[i] +  abs(int(x[0])-int(x1[0]))/abs(int(x1[0])-int(x2[0]))*C2[i]
    # If the y values are not equal then we interpolate between y values        
    else:
        for i in range(3) :
            values[i] =abs(x[1]-x2[1])/abs(x1[1]-x2[1])*C1[i] + abs(x[1]-x1[1])/abs(x1[1]-x2[1])*C2[i]
    
    return values


def render(verts2d, faces, vcolors, depth, shade_t,normals, bcoords, bg_color, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, eye, lights):
    
    # We initialize the image array
    img = np.ones((512,512,3))
    for i in range(512):
        for j in range(512):
            img[i][j] = bg_color

    # find the mean depth for every triangle
    depthTri = np.zeros(len(faces))

    for i in range(len(faces)):
        depthTri[i] = (depth[faces[i][0]] + depth[faces[i][1]] + depth[faces[i][2]])/3

    # We create an array with all the triangles sorted according to their depth
    inds = depthTri.argsort()[::-1]
    facesSorted = faces[inds]
    bcoordsSorted = bcoords[inds]
    # then we call the shade triangle for each triangle depending on which shader we want to use
    if shade_t == "phong" :
        for face in range(len(facesSorted)) :
            img = shade_phong(verts2d[facesSorted[face]], normals[facesSorted[face]], vcolors[facesSorted[face]], bcoordsSorted[face], eye, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, img, lights)
    elif shade_t == "gouraud" :
        for face in range(len(facesSorted)) :
            img = shade_gouraud(verts2d[facesSorted[face]], normals[facesSorted[face]], vcolors[facesSorted[face]], bcoordsSorted[face], eye, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, img, lights)

    return img

def affine_transform(cp, theta, u, t):
    # first i define the array i will return
    cq = []
    # if t is not none kanoume tin metatopisi 
    if t is not None :
        cq = cp + t
    # if the angle is not none tote kanoume tin peristrofi
    if theta is not None :
        # i define the matrix 
        cq = np.zeros((len(cp),3))
        for i in range(0,len(cp)) :
            # and then i make the calculation of the rotation
            cq[i] = (1 - cos(theta))*np.dot((cp[i].dot(u)),u) + cos(theta)*cp[i] + sin(theta)*np.cross(u,cp[i])

    return cq


def system_transform(cp, R, c0): 
    # i simply make the calculation and return it
    dp = np.matmul(np.linalg.inv(R),(cp-c0))
    return dp


def rasterize(verts2d, img_h, img_w, cam_h, cam_w) :
    # i add the offset so that its centered
    for i in range(len(verts2d)) :
        verts2d[i] = verts2d[i] + cam_h/2
    # i define the array to save the new values
    verts_rast = np.zeros((len(verts2d),2))
    # i find the position of the point with respect to the image resolution and the sensors size and we round it
    # i also substract it from the img_h because the two matrices define the image differently
    for i in range(len(verts2d)):
        verts_rast[i][0] =  img_h - int(verts2d[i][1]*img_w/cam_w)
        verts_rast[i][1] =   int(verts2d[i][0]*img_h/cam_h)
             
    # i return the matrix
    return verts_rast

def project_cam(f,c_u,c_x,c_y,c_z,verts3d):
    # i define the R matrix as the stacking of the 3 vectors
    R = np.vstack((c_x,c_y,c_z))

    # i change the coordinate system for the points
    p = np.zeros((len(verts3d), 3))
    for vert in range(len(verts3d)) :
        p[vert] = system_transform(verts3d[vert],R,c_u)

    # i define the matrices which we will return at the end
    verts2d = np.zeros((len(p),2))
    depth = np.zeros((len(p)))

    # i calculate the projection and pass the z value as the depth
    for i in range(len(p)) :
        z = p[i][2]
        depth[i] = z
        verts2d[i][0] = f*p[i][0]/z
        verts2d[i][1] = f*p[i][1]/z 

    # i return the value
    return verts2d, depth


def project_cam_lookat(f, c_org, c_lookat, c_up, verts3d):
    
    # i first calculate the vectors of the camera with the given vectors
    c_z = c_lookat - c_org
    c_z = c_z/np.linalg.norm(c_z)

    t = c_up - c_up.dot(c_z)*c_z
    c_y = t/np.linalg.norm(t)

    c_x = np.cross(c_z, c_y)

    # and then i call project_cam 

    return project_cam(f, c_org, c_x, c_y, c_z, verts3d)

# ambient light is simple enough
def ambient_light(k_a, I_a):
    return k_a*I_a

# The vectors are calculated, normalized and the final calculation is done
def diffuse_light(P, N, color, k_d, light_positions, light_intensities):
    I = np.zeros(3)
    for i in range(len(light_positions)):
        L = light_positions[i] - P
        L = L/np.linalg.norm(L)
        I += k_d*light_intensities[i]*N.dot(L)
    return I*color

# The vectors are calculated, normalized and the final calculation is done
def specular_light(P, N, color, cam_pos, k_s, n, light_positions, light_intensities):
    I = np.zeros(3)
    for i in range(len(light_positions)):
        V = cam_pos - P
        V = V/np.linalg.norm(V)
        L = light_positions[i] - P
        L = L/np.linalg.norm(L)
        I += k_s*light_intensities[i]*pow(V.dot(2*N.dot(N.dot(L))),n)
    return I*color


def calculate_normals(vertices, face_indices):

    # Two arrays are initialized 
    normalsTriang = np.zeros((len(face_indices),3))
    normals = np.zeros((len(vertices),3))

    #calculate the normal vectors for each triangle
    for i in range(len(face_indices)):
        a = vertices[face_indices[i][0]]
        b = vertices[face_indices[i][1]]
        c = vertices[face_indices[i][2]]
        normalsTriang[i] = np.cross((b-a),(c-a))
        normalsTriang[i] = normalsTriang[i]/np.linalg.norm(normalsTriang[i])
    
    # find the normal vector of a vertex averaging the normal vectors of the triangles the vertex belongs to
    for i in range(len(vertices)) :
        for j in range(len(face_indices)) :
            if i in face_indices[j] :
                normals[i] += normalsTriang[j]
        normals[i] = normals[i]/np.linalg.norm(normals[i])

    return normals

# This is the same kind of flawed function from the first exercise
# It does the same thing but calculates the color of each vertex with the functions calculating the effects of lighting    
# Param :
# lights : A vector that passes on the informantion on what types of lighting will be used
def shade_gouraud(verts_p, verts_n, verts_c, bcoords, cam_pos, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, X, lights):

    #calculate the lighting as the sum of the different types of lighting
    vcolors = np.zeros((3,3))
    for i in range(3) :
        ambient = ambient_light(k_a, I_a)
        diffuse = diffuse_light(bcoords, verts_n[i], verts_c[i], k_d, light_positions, light_intensities )
        specular = specular_light(bcoords, verts_n[i], verts_c[i], cam_pos, k_s, n, light_positions, light_intensities)
        vcolors[i] = ambient*lights[0] + diffuse*lights[1] + specular*lights[2]

    # we find the min and max for x and y values
    ymin = int(min(verts_p[0][1],verts_p[1][1],verts_p[2][1]))
    ymax = int(max(verts_p[0][1],verts_p[1][1],verts_p[2][1]))

    xmin = int(min(verts_p[0][0],verts_p[1][0],verts_p[2][0]))
    xmax = int(max(verts_p[0][0],verts_p[1][0],verts_p[2][0]))
    
    # I make an array with the edges so that i can use them more easily
    edges1 = np.vstack((verts_p[0],verts_p[1]))
    edges2 = np.vstack((verts_p[0],verts_p[2]))
    edges3 = np.vstack((verts_p[1],verts_p[2]))

    edges = np.stack((edges1,edges2,edges3))

    edgesC = np.array([[0,1],[0,2],[1,2]])
    
    # I make an array and calculate the angle for each edge so we can use it later
    angs = np.zeros(3)

    for i in range(3):
        if edges[i][0][0] - edges[i][1][0] == 0 :
            angs[i] = float('inf')
        else :
            angs[i] = (edges[i][0][1] - edges[i][1][1])/(edges[i][0][0] - edges[i][1][0])
    

    # I then find the min and max for each edge as I arranged them above

    ymins = np.array([min(verts_p[0][1],verts_p[1][1]),min(verts_p[0][1],verts_p[2][1]),min(verts_p[1][1],verts_p[2][1])])
    ymaxs = np.array([max(verts_p[0][1],verts_p[1][1]),max(verts_p[0][1],verts_p[2][1]),max(verts_p[1][1],verts_p[2][1])])
    
    xmins = np.array([int(min(verts_p[0][0],verts_p[1][0])),int(min(verts_p[0][0],verts_p[2][0])),int(min(verts_p[1][0],verts_p[2][0]))])
    xmaxs = np.array([int(max(verts_p[0][0],verts_p[1][0])),int(max(verts_p[0][0],verts_p[2][0])),int(max(verts_p[1][0],verts_p[2][0]))])
    
    
    # Depending on which point is at the bottom I initialize the starting active points 
    # and pack each point with the corresponding active edge it belongs to        
    if verts_p[0][1] == ymin :
        active_point_edge1 = np.array([verts_p[0][0], verts_p[0][1], 0])
        active_point_edge2 = np.array([verts_p[0][0], verts_p[0][1], 1])
        
        
    elif verts_p[1][1] == ymin :
        active_point_edge1 = np.array([verts_p[1][0], verts_p[1][1], 0])
        active_point_edge2 = np.array([verts_p[1][0], verts_p[1][1], 2])
        
    else:
        active_point_edge1 = np.array([verts_p[2][0], verts_p[2][1], 1])
        active_point_edge2 = np.array([verts_p[2][0], verts_p[2][1], 2])
        
    # I stack these two points together    
    active_point_edge = np.stack((active_point_edge1,active_point_edge2))
    
    # If all the points are on the same coordinates I paint it with the corresponding color
    if (verts_p[0] == verts_p[1]).all() and (verts_p[0] == verts_p[2]).all() and (verts_p[1] == verts_p[2]).all():
        for i in range(3):
            X[int(verts_p[0][0])][int(verts_p[0][1])][i] = (vcolors[0][i] + vcolors[1][i] + vcolors[2][i])/3
    # If the points are not creating a line then I fill it           
    elif (verts_p[0] == verts_p[1]).all() == False and (verts_p[0] == verts_p[2]).all() == False and (verts_p[1] == verts_p[2]).all() == False :

        for y in range(ymin,ymax,1) :
            # If two of the points create a flat line at the bottom then we paint that line and then initialize the active points/edges accordingly
            if y == ymins[0] == ymaxs[0] == ymin :
                for x in range(xmins[0],xmaxs[0]):
                    # to find the color we interpolate between the two points
                    X[x][y] = interpolate_color(edges[0][0], edges[0][1], np.array([x,y]), vcolors[edgesC[0][0]], vcolors[edgesC[0][1]])
                    active_point_edge1 = np.array([verts_p[0][0], verts_p[0][1], 1])
                    active_point_edge2 = np.array([verts_p[1][0], verts_p[1][1], 2])
                    active_point_edge = np.stack((active_point_edge1,active_point_edge2))
            elif y == ymin == ymaxs[1] == ymins[1] :
                for x in range(xmins[1],xmaxs[1]):
                    # to find the color we interpolate between the two points
                    X[x][y] = interpolate_color(edges[1][0], edges[1][1], np.array([x,y]), vcolors[edgesC[1][0]], vcolors[edgesC[1][1]])
                    active_point_edge1 = np.array([verts_p[0][0], verts_p[0][1], 0])
                    active_point_edge2 = np.array([verts_p[2][0], verts_p[2][1], 2])
                    active_point_edge = np.stack((active_point_edge1,active_point_edge2))
            elif y == ymin == ymaxs[2] == ymins[2] :
                for x in range(xmins[2],xmaxs[2]):
                    # to find the color we interpolate between the two points
                    X[x][y] = interpolate_color(edges[2][0], edges[2][1], np.array([x,y]), vcolors[edgesC[2][0]], vcolors[edgesC[2][1]])
                    active_point_edge1 = np.array([verts_p[1][0], verts_p[1][1], 0])
                    active_point_edge2 = np.array([verts_p[2][0], verts_p[2][1], 1])
                    active_point_edge = np.stack((active_point_edge1,active_point_edge2))
            else:
                #we first sort the active points
                active_point_edge = active_point_edge[np.argsort(active_point_edge[:,0])]
                #and paint its pixel between the two active edgepoints
                for x in range(int(active_point_edge[0][0]),int(active_point_edge[1][0])) :
                    # To find the color of a random point inside the triangle first we find the color interpolating on the y axis 2 times for the active edgepoints
                    # then we interpolate between those two, which interpolates using their x value and find the color of the point in between
                    color1 = interpolate_color(edges[int(active_point_edge[0][2])][0],edges[int(active_point_edge[0][2])][1],np.array([active_point_edge[0][0],active_point_edge[0][1]]),vcolors[edgesC[int(active_point_edge[0][2])][0]],vcolors[edgesC[int(active_point_edge[0][2])][1]])
                    color2 = interpolate_color(edges[int(active_point_edge[1][2])][0],edges[int(active_point_edge[1][2])][1],np.array([active_point_edge[1][0],active_point_edge[1][1]]),vcolors[edgesC[int(active_point_edge[1][2])][0]],vcolors[edgesC[int(active_point_edge[1][2])][1]])
                    color3 = interpolate_color(active_point_edge[0],active_point_edge[1],np.array([int(x),y]),color1,color2)
                    X[int(x)][y] = color3

                # Updating the active edges according to the algorhithm
                for i in range(3) :
                    if y + 1 == ymins[i] :
                        for j in range(2) :
                            if edges[i][j][1] == y+1 :
                                newPoint = np.array([edges[i][j][0],edges[i][j][1],i])
                                active_point_edge = np.vstack((active_point_edge, newPoint))
                    # edges_to_rem is and array which holds the indexes for the edges which are to be removed, it is not needed to be so complex
                    edges_to_rem = np.array([])    
                    if y + 1 == ymaxs[i] :
                        for j in range(len(active_point_edge)):
                            if active_point_edge[j][2] == i :
                                edges_to_rem = np.append(edges_to_rem,int(j))

                    if(edges_to_rem.size != 0):
                        active_point_edge = np.delete(active_point_edge,edges_to_rem.astype(int),axis=0)        

            # Finally we add 1/angle to the points that were not just added above and increment their y value
            for i in range(len(active_point_edge)) :
                if active_point_edge[i][1] != y + 1 :
                    active_point_edge[i][1] += 1
                    angle = angs[int(active_point_edge[i][2])]
                    active_point_edge[i][0] = active_point_edge[i][0] + 1/angle
        
    return X

# This is the same kind of flawed function from the first exercise
# It does the same thing but calculates the color of each point by interpolating the vector and the underlying color and then calculating the final total color
# Param :
# lights : A vector that passes on the informantion on what types of lighting will be used
def shade_phong(verts_p, verts_n, verts_c, bcoords, cam_pos, k_a, k_s, k_d, n, light_positions, light_intensities, I_a, X, lights):
    
    # we find the min and max for x and y values
    ymin = int(min(verts_p[0][1],verts_p[1][1],verts_p[2][1]))
    ymax = int(max(verts_p[0][1],verts_p[1][1],verts_p[2][1]))

    xmin = int(min(verts_p[0][0],verts_p[1][0],verts_p[2][0]))
    xmax = int(max(verts_p[0][0],verts_p[1][0],verts_p[2][0]))
    
    # I make an array with the edges so that i can use them more easily
    edges1 = np.vstack((verts_p[0],verts_p[1]))
    edges2 = np.vstack((verts_p[0],verts_p[2]))
    edges3 = np.vstack((verts_p[1],verts_p[2]))

    edges = np.stack((edges1,edges2,edges3))

    edgesC = np.array([[0,1],[0,2],[1,2]])
    
    # I make an array and calculate the angle for each edge so we can use it later
    angs = np.zeros(3)

    for i in range(3):
        if edges[i][0][0] - edges[i][1][0] == 0 :
            angs[i] = float('inf')
        else :
            angs[i] = (edges[i][0][1] - edges[i][1][1])/(edges[i][0][0] - edges[i][1][0])
    

    # I then find the min and max for each edge as I arranged them above

    ymins = np.array([min(verts_p[0][1],verts_p[1][1]),min(verts_p[0][1],verts_p[2][1]),min(verts_p[1][1],verts_p[2][1])])
    ymaxs = np.array([max(verts_p[0][1],verts_p[1][1]),max(verts_p[0][1],verts_p[2][1]),max(verts_p[1][1],verts_p[2][1])])
    
    xmins = np.array([int(min(verts_p[0][0],verts_p[1][0])),int(min(verts_p[0][0],verts_p[2][0])),int(min(verts_p[1][0],verts_p[2][0]))])
    xmaxs = np.array([int(max(verts_p[0][0],verts_p[1][0])),int(max(verts_p[0][0],verts_p[2][0])),int(max(verts_p[1][0],verts_p[2][0]))])
    
    
    # Depending on which point is at the bottom I initialize the starting active points 
    # and pack each point with the corresponding active edge it belongs to        
    if verts_p[0][1] == ymin :
        active_point_edge1 = np.array([verts_p[0][0], verts_p[0][1], 0])
        active_point_edge2 = np.array([verts_p[0][0], verts_p[0][1], 1])
        
        
    elif verts_p[1][1] == ymin :
        active_point_edge1 = np.array([verts_p[1][0], verts_p[1][1], 0])
        active_point_edge2 = np.array([verts_p[1][0], verts_p[1][1], 2])
        
    else:
        active_point_edge1 = np.array([verts_p[2][0], verts_p[2][1], 1])
        active_point_edge2 = np.array([verts_p[2][0], verts_p[2][1], 2])
        
    # I stack these two points together    
    active_point_edge = np.stack((active_point_edge1,active_point_edge2))
    
    # If all the points are on the same coordinates I paint it with the corresponding color
    if (verts_p[0] == verts_p[1]).all() and (verts_p[0] == verts_p[2]).all() and (verts_p[1] == verts_p[2]).all():
        for i in range(3):
            color = (verts_c[0][i] + verts_c[1][i] + verts_c[2][i])/3
            # find the average of the normal vectors 
            Nmean = (verts_n[0][i] + verts_n[1][i] + verts_n[2][i])/3
            finalColor = ambient_light(k_a, I_a)*lights[0] + diffuse_light(bcoords, Nmean[i], color[i], k_d, light_positions, light_intensities )*lights[1] + specular_light(bcoords, Nmean[i], color[i], cam_pos, k_s, n, light_positions, light_intensities)*lights[2]
            X[int(verts_p[0][0])][int(verts_p[0][1])][i] = finalColor

    # If the points are not creating a line then I fill it           
    elif (verts_p[0] == verts_p[1]).all() == False and (verts_p[0] == verts_p[2]).all() == False and (verts_p[1] == verts_p[2]).all() == False :

        for y in range(ymin,ymax,1) :
            # If two of the points create a flat line at the bottom then we paint that line and then initialize the active points/edges accordingly
            if y == ymins[0] == ymaxs[0] == ymin :
                for x in range(xmins[0],xmaxs[0]):
                    # to find the color we interpolate between the two points
                    color = interpolate_color(edges[0][0], edges[0][1], np.array([x,y]), verts_c[edgesC[0][0]], verts_c[edgesC[0][1]])
                    # to find the normal vector we interpolate between the normal vectors
                    N = interpolate_color(edges[0][0], edges[0][1], np.array([x,y]), verts_n[edgesC[0][0]], verts_n[edgesC[0][1]])
                    # the final color is then the result using this normal vector
                    finalColor = ambient_light(k_a, I_a)*lights[0] + diffuse_light(bcoords, N, color, k_d, light_positions, light_intensities )*lights[1] + specular_light(bcoords, N, color, cam_pos, k_s, n, light_positions, light_intensities)*lights[2]
                    X[x][y] = finalColor
                    active_point_edge1 = np.array([verts_p[0][0], verts_p[0][1], 1])
                    active_point_edge2 = np.array([verts_p[1][0], verts_p[1][1], 2])
                    active_point_edge = np.stack((active_point_edge1,active_point_edge2))

            elif y == ymin == ymaxs[1] == ymins[1] :
                for x in range(xmins[1],xmaxs[1]):
                    # to find the color we interpolate between the two points
                    color = interpolate_color(edges[1][0], edges[1][1], np.array([x,y]), verts_c[edgesC[1][0]], verts_c[edgesC[1][1]])
                    # to find the normal vector we interpolate between the normal vectors
                    N = interpolate_color(edges[1][0], edges[1][1], np.array([x,y]), verts_n[edgesC[1][0]], verts_n[edgesC[1][1]])
                    # the final color is then the result using this normal vector
                    finalColor = ambient_light(k_a, I_a)*lights[0] + diffuse_light(bcoords, N, color, k_d, light_positions, light_intensities )*lights[1] + specular_light(bcoords, N, color, cam_pos, k_s, n, light_positions, light_intensities)*lights[2]
                    X[x][y] = finalColor
                    active_point_edge1 = np.array([verts_p[0][0], verts_p[0][1], 0])
                    active_point_edge2 = np.array([verts_p[2][0], verts_p[2][1], 2])
                    active_point_edge = np.stack((active_point_edge1,active_point_edge2))


            elif y == ymin == ymaxs[2] == ymins[2] :
                for x in range(xmins[2],xmaxs[2]):
                    # to find the color we interpolate between the two points
                    color = interpolate_color(edges[2][0], edges[2][1], np.array([x,y]), verts_c[edgesC[2][0]], verts_c[edgesC[2][1]])
                    # to find the normal vector we interpolate between the normal vectors
                    N = interpolate_color(edges[2][0], edges[2][1], np.array([x,y]), verts_n[edgesC[2][0]], verts_n[edgesC[2][1]])
                    # the final color is then the result using this normal vector
                    finalColor = ambient_light(k_a, I_a)*lights[0] + diffuse_light(bcoords, N, color, k_d, light_positions, light_intensities )*lights[1] + specular_light(bcoords, N, color, cam_pos, k_s, n, light_positions, light_intensities)*lights[2]
                    X[x][y] = finalColor
                    active_point_edge1 = np.array([verts_p[1][0], verts_p[1][1], 0])
                    active_point_edge2 = np.array([verts_p[2][0], verts_p[2][1], 1])
                    active_point_edge = np.stack((active_point_edge1,active_point_edge2))

            else:
                #we first sort the active points
                active_point_edge = active_point_edge[np.argsort(active_point_edge[:,0])]
                #and paint its pixel between the two active edgepoints
                for x in range(int(active_point_edge[0][0]),int(active_point_edge[1][0])) :
                    # To find the color of a random point inside the triangle first we find the color interpolating on the y axis 2 times for the active edgepoints
                    # then we interpolate between those two, which interpolates using their x value and find the color of the point in between
                    color1 = interpolate_color(edges[int(active_point_edge[0][2])][0],edges[int(active_point_edge[0][2])][1],np.array([active_point_edge[0][0],active_point_edge[0][1]]),verts_c[edgesC[int(active_point_edge[0][2])][0]],verts_c[edgesC[int(active_point_edge[0][2])][1]])
                    color2 = interpolate_color(edges[int(active_point_edge[1][2])][0],edges[int(active_point_edge[1][2])][1],np.array([active_point_edge[1][0],active_point_edge[1][1]]),verts_c[edgesC[int(active_point_edge[1][2])][0]],verts_c[edgesC[int(active_point_edge[1][2])][1]])
                    color3 = interpolate_color(active_point_edge[0],active_point_edge[1],np.array([int(x),y]),color1,color2)

                    # To find the normal vector of a random point inside the triangle first we find the color interpolating on the y axis 2 times for the active edgepoints
                    # then we interpolate between those two, which interpolates using their x value and find the color of the point in between
                    N1 = interpolate_color(edges[int(active_point_edge[0][2])][0],edges[int(active_point_edge[0][2])][1],np.array([active_point_edge[0][0],active_point_edge[0][1]]),verts_n[edgesC[int(active_point_edge[0][2])][0]],verts_n[edgesC[int(active_point_edge[0][2])][1]])
                    N2 = interpolate_color(edges[int(active_point_edge[1][2])][0],edges[int(active_point_edge[1][2])][1],np.array([active_point_edge[1][0],active_point_edge[1][1]]),verts_n[edgesC[int(active_point_edge[1][2])][0]],verts_n[edgesC[int(active_point_edge[1][2])][1]])
                    N3 = interpolate_color(active_point_edge[0],active_point_edge[1],np.array([int(x),y]),N1,N2)
                    # the final color is then the result using this normal vector
                    finalColor = ambient_light(k_a, I_a)*lights[0] + diffuse_light(bcoords, N3, color3, k_d, light_positions, light_intensities )*lights[1] + specular_light(bcoords, N3, color3, cam_pos, k_s, n, light_positions, light_intensities)*lights[2]

                    X[int(x)][y] = finalColor

                # Updating the active edges according to the algorhithm
                for i in range(3) :
                    if y + 1 == ymins[i] :
                        for j in range(2) :
                            if edges[i][j][1] == y+1 :
                                newPoint = np.array([edges[i][j][0],edges[i][j][1],i])
                                active_point_edge = np.vstack((active_point_edge, newPoint))
                    # edges_to_rem is and array which holds the indexes for the edges which are to be removed, it is not needed to be so complex
                    edges_to_rem = np.array([])    
                    if y + 1 == ymaxs[i] :
                        for j in range(len(active_point_edge)):
                            if active_point_edge[j][2] == i :
                                edges_to_rem = np.append(edges_to_rem,int(j))

                    if(edges_to_rem.size != 0):
                        active_point_edge = np.delete(active_point_edge,edges_to_rem.astype(int),axis=0)        

            # Finally we add 1/angle to the points that were not just added above and increment their y value
            for i in range(len(active_point_edge)) :
                if active_point_edge[i][1] != y + 1 :
                    active_point_edge[i][1] += 1
                    angle = angs[int(active_point_edge[i][2])]
                    active_point_edge[i][0] = active_point_edge[i][0] + 1/angle
        
    return X

def render_object(shader, focal, eye, lookat, up, bg_color, M, N, H, W, verts, vert_colors, face_indices, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, lights):
    # first we find the normal vectors
    normals = calculate_normals(verts, face_indices)
    # then we calculate the center of mass of each triangle to use when calculating the light
    bcoords = np.zeros((len(face_indices),3))
    for i in range(len(face_indices)):
        for j in range(3):
            bcoords[i][0] += verts[face_indices[i][j]][0]  
            bcoords[i][1] += verts[face_indices[i][j]][1]
            bcoords[i][2] += verts[face_indices[i][j]][2]
        bcoords[i] = bcoords[i]/3
    # i first find the projected triangles and their depth    
    verts2d, depth = project_cam_lookat(focal, eye, lookat, up, verts)
    # then i call rasterize to get exact values of coordinates to map points to pixels
    verts_rast = rasterize(verts2d, M, N, H, W)
    # and finally i render the image as we did in the first project
    return render(verts_rast, face_indices, vert_colors, depth, shader, normals, bcoords, bg_color, k_a, k_d, k_s, n, light_positions, light_intensities, I_a, eye, lights) # ypothetw oti prepei edw na allaksw ta orismata ? kapou bazw to background color


data = np.load('h3.npy', allow_pickle=True)

#load all the data
verts = data[()]['verts']
vertex_colors = data[()]['vertex_colors']
face_indices = data[()]['face_indices']
cam_eye = data[()]['cam_eye']
cam_up = data[()]['cam_up']
cam_lookat = data[()]['cam_lookat']
ka = data[()]['ka']
kd = data[()]['kd']
ks = data[()]['ks']
n = data[()]['n']
light_positions = data[()]['light_positions']
light_intensities = data[()]['light_intensities']
Ia = data[()]['Ia']
M = data[()]['M']
N = data[()]['N']
W = data[()]['W']
H = data[()]['H']
bg_color = data[()]['bg_color']
depth = data[()]['depth']


#calculate and save all the images

lights = np.array([1,0,0])

img = render_object('gouraud', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('gouraud_ambient.png')

lights = np.array([0,1,0])

img = render_object('gouraud', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('gouraud_diffused.png')

lights = np.array([0,0,1])

img = render_object('gouraud', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('gouraud_specular.png')

lights = np.array([1,1,1])

img = render_object('gouraud', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('gouraud_all.png')


lights = np.array([1,0,0])

img = render_object('phong', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('phong_ambient.png')

lights = np.array([0,1,0])

img = render_object('phong', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('phong_diffused.png')

lights = np.array([0,0,1])

img = render_object('phong', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('phong_specular.png')

lights = np.array([1,1,1])

img = render_object('phong', 70, cam_eye, cam_lookat, cam_up, bg_color, M, N, H, W, verts, vertex_colors, face_indices, ka, kd, ks, n, light_positions, light_intensities, Ia, lights)

plt.imshow(img,interpolation='nearest')

plt.savefig('phong_all.png')

