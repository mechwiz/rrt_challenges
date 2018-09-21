import numpy as np
import matplotlib.pyplot as mp
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.misc import imread
import cv2

def rand_conf():
    qrand = (100-1)*np.random.random(2)
    return [int(qrand[0]),int(qrand[1])]

def nearest_vertex(qrand,G):
    if np.shape(G)[0] == 1:
        qnear = G[0]
    else:
        dist = []
        xa = qrand[0]
        ya = qrand[1]
        for i in range(np.shape(G)[0]):
            xb = G[i][0]
            yb = G[i][1]
            dist.insert(i,np.sqrt((xa-xb)**2+(ya-yb)**2))
        min_index = dist.index(np.min(dist))
        qnear = G[min_index]
    return qnear

def new_config(qnear,qrand,dq):
    xa = qrand[0]
    ya = qrand[1]
    xb = qnear[0]
    yb = qnear[1]
    qnew = [xb + (xa-xb)*dq,yb + (ya-yb)*dq]
    return qnew

def add_vertex(G,qnew):
    G.append(qnew)
    return G

def add_edge(E,qnear,qnew):
    E.append([qnear,qnew])
    return E

def check_edge(C,qnear, qnew):
    check = 0
    for i in range(np.shape(C)[0]):
        ax,ay = qnear
        bx,by = qnew
        r,cx,cy = C[i]
        ax -= cx
        ay -= cy
        bx -= cx
        by -= cy
        c = ax**2 + ay**2 - r**2
        b = 2*(ax*(bx - ax) + ay*(by - ay))
        a = (bx - ax)**2 + (by - ay)**2
        disc = b**2 - 4*a*c
        if disc > 0:
            sqrtdisc = np.sqrt(disc)
            t1 = (-b + sqrtdisc)/(2*a);
            t2 = (-b - sqrtdisc)/(2*a);
            #print t1,t2
            if(0 < t1 and t1 < 1 or 0 < t2 and t2 < 1):
                #print 'checked'
                check = 1
                break
    if check == 1:
        return True
    else:
        return False

def simple_rrt_gen(iternum,dq):
    G = []
    E = []
    qinit = [50,50]
    G.append(qinit)
    for i in range(iternum):
        qrand = rand_conf()
        qnear = nearest_vertex(qrand,G)
        qnew = new_config(qnear,qrand,dq)
        G = add_vertex(G,qnew)
        E = add_edge(E,qnear,qnew)

    verts = []
    codes = []

    for i in range(np.shape(E)[0]):
        verts.append(E[i][0])
        verts.append(E[i][1])
        codes.append(Path.MOVETO)
        codes.append(Path.LINETO)
    fig = mp.figure()
    path = Path(verts, codes)
    patch = patches.PathPatch(path)
    ax = fig.add_subplot(111)
    ax.add_patch(patch)
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    mp.show()

def circle_rrt_gen(dq):
    G = []
    E = []
    C=[]
    C = generate_circles(20,8,3)

    qinit = get_qinit(C)
    qgoal = get_qgoal(C,qinit)
    G.append(qinit)

    if check_edge(C,qinit,qgoal) == False:
        G = add_vertex(G,qgoal)
        E = add_edge(E,qinit,qgoal)
    else:

        while True:
            qrand = rand_conf()
            qnear = nearest_vertex(qrand,G)
            qnew = new_config(qnear,qrand,dq)
            check = check_edge(C,qnear,qnew)

            if check == False:
                G = add_vertex(G,qnew)
                E = add_edge(E,qnear,qnew)
                check_goal = check_edge(C,qnew,qgoal)
                if check_goal == False:
                    G = add_vertex(G,qgoal)
                    E = add_edge(E,qnew,qgoal)
                    break


    verts = []
    codes = []

    for i in range(np.shape(E)[0]):
        verts.append(E[i][0])
        verts.append(E[i][1])
        codes.append(Path.MOVETO)
        codes.append(Path.LINETO)

    verts_h = []
    codes_h = []
    verts_h.append(verts[-1])
    verts_h.append(verts[-2])
    codes_h.append(Path.MOVETO)
    codes_h.append(Path.LINETO)

    check = 0
    while check == 0:
        for sublist in E:
            if sublist[1] == verts_h[-1]:
                verts_h.append(sublist[1])
                verts_h.append(sublist[0])
                codes_h.append(Path.MOVETO)
                codes_h.append(Path.LINETO)
            if verts_h[-1] == qinit:
                check = 1
                break

    fig = mp.figure()
    path = Path(verts, codes)
    patch = patches.PathPatch(path)
    path_h = Path(verts_h,codes_h)
    patch_h = patches.PathPatch(path_h,color='blue',lw=2)
    ax = fig.add_subplot(111)
    ax.add_patch(patch)
    ax.add_patch(patch_h)
    fcirc = lambda x: patches.Circle((x[1],x[2]), radius=x[0], fill=True, alpha=1, fc='k', ec='k')
    circs = [fcirc(x) for x in C]
    for c in circs:
        ax.add_patch(c)
    mp.plot([qinit[0]],[qinit[1]],marker='o',markersize=10,color='red')
    mp.plot([qgoal[0]],[qgoal[1]],marker='o',markersize=10,color='blue')
    ax.set_xlim([0,100])
    ax.set_ylim([0,100])
    mp.show()


SIZE = 100


def get_qinit(C):
    while 1:
        check = True
        qinit = rand_conf()
        for i in range(np.shape(C)[0]):
            ax,ay = qinit
            r,cx,cy = C[i]
            d = np.sqrt((ax-cx)**2 +(ay-cy)**2)
            if d < r:
                check = False
                break
        if check == True:
            break
    return qinit

def get_qgoal(C,qinit):
    while 1:
        check = True
        qgoal = rand_conf()
        if qgoal == qinit:
            check = False
        else:
            for i in range(np.shape(C)[0]):
                ax,ay = qgoal
                r,cx,cy = C[i]
                d = np.sqrt((ax-cx)**2 +(ay-cy)**2)
                if d < r:
                    check = False
                    break
        if check == True:
            break
    return qgoal



def generate_circles(num, mean, std):
    """
    This function generates /num/ random circles with a radius mean defined by
    /mean/ and a standard deviation of /std/.

    The circles are stored in a num x 3 sized array. The first column is the
    circle radii and the second two columns are the circle x and y locations.
    """
    circles = np.zeros((num,3))
    # generate circle locations using a uniform distribution:
    circles[:,1:] = np.random.uniform(mean, SIZE-mean, size=(num,2))
    # generate radii using a normal distribution:
    circles[:,0] = np.random.normal(mean, std, size=(num,))
    return circles

def points (p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    dx = abs(x1-x0)
    dy = abs(y1-y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1


    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx-dy

    point_list = []
    while True:
        point_list.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break

        e2 = 2*err
        if e2 > -dy:
            # overshot in the y direction
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            # overshot in the x direction
            err = err + dx
            y0 = y0 + sy

    return point_list

def check_edge_n(c,qnear, qnew):
    check = False
    dx = [0,-1,-1,-1,0,1,1,1]
    dy = [1,1,0,-1,-1,-1,0,1]


    line_pnts = points(qnear,qnew)

    for i in range(len(line_pnts)):
        x,y = line_pnts[i]
        #print c[x][y]
        if c[y,x] > 0:
            check = True
            break
        # for j in range(8):
        #     #print y+dx[j],x+dy[j]
        #     if (y+dx[j]) > 0 and (y+dx[j]) < np.shape(c)[0] and (x+dy[j]) > 0 and (x+dy[j]) < np.shape(c)[1]:
        #         if c[y+dx[j],x+dy[j]] > 0:
        #             check = True
        #             break
        # if check == True:
        #     break

    return check

def rand_int(s,qnear):
    qrand = [0,0]
    r = s/2
    s = s-1
    qrand[0] = np.random.random_integers(max(qnear[0]-r,0),min(qnear[0]+r,s))
    qrand[1] = np.random.random_integers(max(qnear[1]-r,0),min(qnear[1]+r,s))
    return [qrand[0],qrand[1]]

def n_rrt_gen(dq):
    FNAME = "imgs/N_map.png"
    world = imread(FNAME,mode='L')
    nonz = np.nonzero(world)
    wc = (np.ones(np.shape(world),dtype=np.uint8))*255
    wc[nonz] = 0
    world = wc
    world = np.flipud(world)
    worldo = world.copy()
    #kernel = np.ones((3,3), np.uint8)
    #world = cv2.dilate(world,kernel,iterations=4)

    G = []
    E = []

    qinit = [40,40]
    qgoal = [60,60]
    qnear = np.copy(qinit).tolist()

    #Xmax = world.shape[0]
    #Ymax = world.shape[1]
    #fig = mp.figure()
    #ax = fig.add_subplot(111)
    #ax.imshow(world,cmap=mp.cm.binary,interpolation='nearest', origin='lower',extent=[0,Xmax,0,Ymax])
    #mp.plot([qinit[0]],[qinit[1]],marker='o',markersize=10,color='red')
    #mp.plot([qgoal[0]],[qgoal[1]],marker='o',markersize=10,color='blue')
    #ax.set_xlim([0,world.shape[0]])
    #ax.set_ylim([0,world.shape[1]])
    #mp.show()

    G.append(qinit)

    if check_edge_n(world,qinit,qgoal) == False:
        G = add_vertex(G,qgoal)
        E = add_edge(E,qinit,qgoal)
    else:

        while True:
            qrand = rand_int(world.shape[0],qnear)
            qnear = nearest_vertex(qrand,G)
            qnew = new_config(qnear,qrand,dq)
            check = check_edge_n(world,qnear,qnew)
            if check == False:
                G = add_vertex(G,qnew)
                E = add_edge(E,qnear,qnew)

                check_goal = check_edge_n(world,qnew,qgoal)
                if check_goal == False:
                    G = add_vertex(G,qgoal)
                    E = add_edge(E,qnew,qgoal)
                    break

    verts = []
    codes = []

    for i in range(np.shape(E)[0]):
        verts.append(E[i][0])
        verts.append(E[i][1])
        codes.append(Path.MOVETO)
        codes.append(Path.LINETO)

    verts_h = []
    codes_h = []
    verts_h.append(verts[-1])
    verts_h.append(verts[-2])
    codes_h.append(Path.MOVETO)
    codes_h.append(Path.LINETO)

    check = 0
    while check == 0:
        for sublist in E:
            if sublist[1] == verts_h[-1]:
                verts_h.append(sublist[1])
                verts_h.append(sublist[0])
                codes_h.append(Path.MOVETO)
                codes_h.append(Path.LINETO)
            if verts_h[-1] == qinit:
                check = 1
                break

    Xmax = worldo.shape[0]
    Ymax = worldo.shape[1]
    fig = mp.figure()
    path = Path(verts, codes)
    patch = patches.PathPatch(path)
    path_h = Path(verts_h,codes_h)
    patch_h = patches.PathPatch(path_h,color='blue',lw=2)
    ax = fig.add_subplot(111)
    ax.add_patch(patch)
    ax.add_patch(patch_h)
    ax.imshow(worldo,cmap=mp.cm.binary,interpolation='nearest', origin='lower',extent=[0,Xmax,0,Ymax])
    mp.plot([qinit[0]],[qinit[1]],marker='o',markersize=10,color='red')
    mp.plot([qgoal[0]],[qgoal[1]],marker='o',markersize=10,color='blue')
    ax.set_xlim([0,worldo.shape[0]])
    ax.set_ylim([0,worldo.shape[1]])
    mp.show()

if __name__ == '__main__':
    simple_rrt_gen(100,1)
    circle_rrt_gen(1)
    n_rrt_gen(1)
