import random
random.seed(1)
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
import matplotlib as mpl
import scipy.io as sio
import sklearn as skl
from sklearn.cluster import KMeans
from sklearn import metrics
import time


bundle=1                #0 for no bundling, 1 for bundling
chi=1                   #0 for no chirality on the wall, 1 for helix formation
total_step = 120        # Total simulation step: each step is 0.03 days
cell_no = 100          # Total cell number in forward direction
cell_no_b = cell_no     # Total cell number in backward direction
rad = 180.0             # Inner radius of microcolumn
z_fwd = 0               # Initial seeding position in z direction for forward growth
z_bw = 2000             # Initial seeding position in z direction for backward growth (length of the tube)
  
FibR=70                 #Interaction radius for bundle formation

# Define all the parameters
D=0.4
delT=1
K1=4*D
K2=(4*math.pi*D)**(3/2)
k0 = 5
k3 = 0.001
k5 = 1

s1 = 0.01              # Sensitivity to concentration gradients from forward direction
s_d1 = 0.05            # Sensitivity to direction perturbation term in forward direction
                      
v0 = 20                 # Base growth rate
v0_grad = 0.008         # Growth rate based on gradient strength
tau = 150.0             # Time constant for growth rate
err1_theta =  math.pi   # Direction perturbation term in theta
v_bi = 1                # Chemical effect of bidirectional growth: 1 at beginning

tau_b = 20              # Time constant for branching rate

P = 0                   # Branching probability at infinite: 0 means no branching
B1 = 0.1                # Branching condition 1: Branching may happen if probability value greater than B1
B2 = 0.15               # Branching condition 2: Branching may happen if random uniform term greater than B2


def whosMy(*args):
  sequentialTypes = [dict, list, tuple] 
  for var in args:
    t=type(var)
    if t== np.ndarray:  
      print(type(var),var.dtype, var.shape)
    elif t in sequentialTypes: 
      print(type(var), len(var))
    else:
      print(type(var))



# Define the initial positions
def initial_posXYZ(cell_no, rad, Z):
    # 0    1    2     3      4      5       6       7        8              9          10
    # x    y    z   dir_x  dir_y  dir_z  cell_no  step  previous_node  current_node  branch_no
    pos0XYZ = elevenCol
    for i in range(cell_no):
        rr = 0.999*rad* random.uniform(0.0, 1.0)
        thr= 2*math.pi * random.uniform(0.0, 1.0)
        zr0= rad * random.uniform(-1.0, 1.0) + Z
        pos_initial = np.array([[rr*np.cos(thr), rr*np.sin(thr),zr0, 0, 0, 1, i, 0, 0, 0, 1]])

        pos0XYZ = np.vstack((pos0XYZ, pos_initial))
        reset_nodes = elevenCol
        if np.array_equal(reset_nodes, pos0XYZ[0]):
            pos0XYZ = np.delete(pos0XYZ, 0, 0)
    return pos0XYZ


def initial_nodes(cell_no):
    count_nodes = np.array([0])
    for i in range(cell_no):
        initial_node = np.array([0])
        count_nodes = np.vstack((count_nodes, initial_node))
        if i == 0:
            reset_nodes = np.array([0])
            if np.array_equal(reset_nodes, count_nodes[0]):
                count_nodes = np.delete(count_nodes, 0, 0)
    return count_nodes

def initial_CM(pos0,FibRad):
       pos_current=pos0
       current_length = len(pos0)
       CM=np.zeros((current_length,3))
       lst_n=np.arange(0,current_length)
       nn=0 
       n_cl=0
       IndN=[]
       while nn < len(lst_n): 
            nn=0
            n=lst_n[nn]
            dis1 = np.squeeze(np.sqrt((pos_current[lst_n,0]-pos_current[n,0])**2+(pos_current[lst_n,1]-pos_current[n,1])**2+(pos_current[lst_n,2]-pos_current[n,2])**2))
                                           
            Rindx0=np.squeeze(np.asarray(np.where((dis1>=0) & (dis1<=FibRad))),axis=0)
            
            Rindx=lst_n[Rindx0]
            LRindx=len(Rindx)
            if LRindx>1:
               cmx= np.sum(pos_current[Rindx,0])/LRindx
               cmy= np.sum(pos_current[Rindx,1])/LRindx
               cmz= np.sum(pos_current[Rindx,2])/LRindx
               
               CM[Rindx,0]=cmx
               CM[Rindx,1]=cmy
               CM[Rindx,2]=cmz
               
               n_cl+=1
               #if Rindx in lst_n:
               lst_n=np.setdiff1d(lst_n,Rindx)
                           
            if n in lst_n:
               lst_n=np.setdiff1d(lst_n,n)
                   
       return CM, n_cl
   
# Calculate the concentration gradients
def gradientXYZ(pos_previous, pos_current):
    # Calculate the gradients
    gradXYZ = threeCol
    
    posx = pos_previous[:, 0] - pos_current[0]
    posy = pos_previous[:, 1] - pos_current[1]
    posz = pos_previous[:, 2] - pos_current[2]
    gr0=(-2/(K1*K2*delT**(5/2)))*posx*np.exp(-(posx**2)/(K1*delT))
    gr1=(-2/(K1*K2*delT**(5/2)))*posy*np.exp(-(posy**2)/(K1*delT))
    gr2=(-2/(K1*K2*delT**(5/2)))*posz*np.exp(-(posz**2)/(K1*delT))
    gradXYZ[0]=np.sum(gr0)
    gradXYZ[1]=np.sum(gr1)
    gradXYZ[2]=np.sum(gr2)
    
    return gradXYZ

def bundle_labelsCM(pos_current,FibRad,ListCM):
       current_length = len(pos_current)
       dirFiber=np.zeros((current_length,8))
       lst_n=np.arange(0,current_length)
       
       nn=0       
       
       while nn < len(lst_CM): 
            nn=0
            
            n=lst_n[nn]
            
            dis1 = np.squeeze(np.sqrt((pos_current[lst_n,0]-pos_current[n,0])**2+(pos_current[lst_n,1]-pos_current[n,1])**2+(pos_current[lst_n,2]-pos_current[n,2])**2))
                                       
            Rindx0=np.squeeze(np.asarray(np.where((dis1>=0) & (dis1<=FibRad))),axis=0)
            
            Rindx=lst_n[Rindx0]
            
            LRindx=len(Rindx)
            if LRindx>1:
               cmx= np.sum(pos_current[Rindx,0])/LRindx+0.1*random.gauss(0,FibRad/4)
               cmy= np.sum(pos_current[Rindx,1])/LRindx+0.1*random.gauss(0,FibRad/4)
               cmz= np.sum(pos_current[Rindx,2])/LRindx
               tvx=cmx-pos_current[Rindx,0]
               tvy=cmy-pos_current[Rindx,1]
               tvz=cmz-pos_current[Rindx,2]
               norm=np.sqrt(tvx**2+tvy**2+tvz**2)
               
               dirFiber[Rindx,0]=tvx/norm
               dirFiber[Rindx,1]=tvy/norm
               dirFiber[Rindx,2]=tvz/norm
               dirFiber[Rindx,3]=cmx
               dirFiber[Rindx,4]=cmy
               dirFiber[Rindx,5]=cmz
               dirFiber[Rindx,6]=rad-np.sqrt(cmx**2+cmy**2)
               dett=(1./16.)*math.pi*abs(random.uniform(0.0, 1.0))
               dirFiber[Rindx,7]=dett
               CCM=np.vstack([CCM,[cmx,cmy,cmz]])
                        
               #if Rindx in lst_n:
               lst_n=np.setdiff1d(lst_n,Rindx)
               
            
            if n in lst_n:
               lst_n=np.setdiff1d(lst_n,n)
                   
       return dirFiber,CCM 


def bundle_labels(pos_current,FibRad,CCM):
       current_length = len(pos_current)
       dirFiber=np.zeros((current_length,8))
       lst_n=np.arange(0,current_length)
       nn=0       
       
       while nn < len(lst_n): 
            nn=0
           
            n=lst_n[nn]
            
            dis1 = np.squeeze(np.sqrt((pos_current[lst_n,0]-pos_current[n,0])**2+(pos_current[lst_n,1]-pos_current[n,1])**2+(pos_current[lst_n,2]-pos_current[n,2])**2))
                                       
            Rindx0=np.squeeze(np.asarray(np.where((dis1>=0) & (dis1<=FibRad))),axis=0)
            
            Rindx=lst_n[Rindx0]
            
            LRindx=len(Rindx)
            if LRindx>1:
               cmx= np.sum(pos_current[Rindx,0])/LRindx+0.1*random.gauss(0,FibRad/4)
               cmy= np.sum(pos_current[Rindx,1])/LRindx+0.1*random.gauss(0,FibRad/4)
               cmz= np.sum(pos_current[Rindx,2])/LRindx
               tvx=cmx-pos_current[Rindx,0]
               tvy=cmy-pos_current[Rindx,1]
               tvz=cmz-pos_current[Rindx,2]
               norm=np.sqrt(tvx**2+tvy**2+tvz**2)
               
               dirFiber[Rindx,0]=tvx/norm
               dirFiber[Rindx,1]=tvy/norm
               dirFiber[Rindx,2]=tvz/norm
               dirFiber[Rindx,3]=cmx
               dirFiber[Rindx,4]=cmy
               dirFiber[Rindx,5]=cmz
               dirFiber[Rindx,6]=rad-np.sqrt(cmx**2+cmy**2)
               dett=(1./16.)*math.pi*abs(random.uniform(0.0, 1.0))
               dirFiber[Rindx,7]=dett
               CCM=np.vstack([CCM,[cmx,cmy,cmz]])
               
               lst_n=np.setdiff1d(lst_n,Rindx)
               
            if n in lst_n:
               lst_n=np.setdiff1d(lst_n,n)
            
       
       return dirFiber,CCM 

# Define 4 different ways of handling fibers on the inner surface of the uTenn, depending on chi and bund
def c0b0(pos_current,i,FRG,pos2,l):
    #print('Chi=0, Bund=0')
    pos2[0] = pos_current[i][0]
    pos2[1] = pos_current[i][1]
    return pos2
def c0b1(pos_current,i,FRG,pos2,l):
    #print('Chi=0, Bund=1')
    if np.sum(FRG[i,0:3])==0:            
       pos2[0] = pos_current[i][0]
       pos2[1] = pos_current[i][1]
    else:
       rb=min(FRG[i,6],1.0)
       rx=random.uniform(-rb,rb)
       ry=random.uniform(-np.sqrt(rb**2-rx**2),np.sqrt(rb**2-rx**2))
       
       pos2[0] =FRG[i,3]+rx  
       pos2[1] =FRG[i,4]+ry
    return pos2

def c1b0(pos_current,i,FRG,pos2,l):
    #print('Chi=1, Bund=0')
    t0=np.arctan2(pos_current[i][1],pos_current[i][0]) 
    dett=(1./8.)*math.pi*abs(random.uniform(0.0, 1.0))
    #print('OrigAngle='+str(t0))
    #print('dAngle='+str(dett))
    pos2[0] = rad*np.cos(t0+dett)
    pos2[1] = rad*np.sin(t0+dett)
    pos2[2] = pos_current[i][2]+l
    return pos2

def c1b1(pos_current,i,FRG,pos2,l):
    #print('Chi=1, Bund=1') 
    if np.sum(FRG[i,0:3])==0:            
       t0=np.arctan2(pos2[1],pos2[0]) 
       dett=(1./16.)*math.pi*abs(random.uniform(0.0, 1.0))
#       print('OrigAngle='+str(t0))
#       print('dAngle='+str(dett))
       pos2[0] = rad*np.cos(t0+dett)
       pos2[1] = rad*np.sin(t0+dett)
       pos2[2] = pos_current[i][2]+l
    else:
       rb=min(FRG[i,6],1.5)
       rx=random.uniform(-rb,rb)
       ry=random.uniform(-np.sqrt(rb**2-rx**2),np.sqrt(rb**2-rx**2))
       t1=np.arctan2(FRG[i,4],FRG[i,3])
       dett=FRG[i,7]
       pos2[0] = rad*np.cos(t1+dett)+rx
       pos2[1] = rad*np.sin(t1+dett)+ry
       pos2[2] = pos_current[i][2]+l
    return pos2
                  

#Find the new position in forward direction
def generate_nextXYZ(step_no, pos_current, pos_previous, v_bi, bundle,CCM):
    pos_new_full = elevenCol
    pos_new_xyz_full = threeCol
    pos_new_rtz_full = threeCol
    i = 0
    
    current_length = len(pos_current)
    
    if step_no<=5:
       FibRad=FibR
    else:
       FibRad=FibR
       
    if bundle>0:
       FRG,CCM=bundle_labels(pos_current,FibRad,CCM)
       #print('FRG='+str(FRG))
    else:
       FRG=0
    
    
    while i < current_length:
        
        step = v0 * random.uniform(0.8, 1.2)
        dir1 = np.array(pos_current[i][3:6])
        
        grad = gradientXYZ(pos_previous[:, 0:3], pos_current[i][0:3])
        
        sf1=9./(1.+np.exp(-pos_current[i][2]+z_bw))
        errR=random.uniform(0.0, 0.01+sf1)
        errTh=2*math.pi*random.uniform(0.0,1.0)
        #errZ=random.uniform(-1.0,1.0)
        errZ=random.uniform(0.0,0.01)
        err1= np.array([errR*np.cos(errTh), errR*np.sin(errTh), errZ])

        if bundle>0:
           dirF=np.array([FRG[i,0],FRG[i,1],FRG[i,2]])
        else:
           dirF=threeCol
        if pos_current[i][2]>=z_bw:
           dirF=threeCol
           

        strength = math.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
        if strength > 0:#0.001:
            grad1 = grad / strength
        else:
            grad1 = threeCol
            grad = threeCol
        
        sf2=4./(1.+np.exp(-pos_current[i][2]+z_bw))+1
        sf3=9./(1.+np.exp(-pos_current[i][2]+z_bw))+1
             
        dir2 = dir1 + sf2*s1 * grad1 + sf2*s_d1 * err1
#        if pos_current[i][2]>=z_bw:
#               dir2 = dir1 + s1 * grad1 + s_d1 * err1
#        else:
#               dir2 = dir1 + s1 * grad1 + s_d1 * err1
        
        l1 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)
        dir2 = dir2 / l1
        l = (v_bi * (v0_grad * strength + step)) * 2 ** (-step_no / tau)
        
        pos2 = pos_current[i][0:3] +dir2* l+ dirF*(7*s_d1)*bundle*l
        ldirF=(15*s_d1)*l*math.sqrt(dirF[0] ** 2 + dirF[1] ** 2 + dirF[2] ** 2)
        ldir2=l*math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)

        cell_number = pos_current[i][6]
        node_counter = initial_node[int(cell_number)]

        # make uTENN grow out after reaching the end

        if pos2[2] <= z_bw:
            #if abs(pos2[0]) <= rad:
            if math.sqrt(pos2[0]**2+pos2[1]**2) <= rad:
                pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
                initial_node[int(cell_number)] = node_counter + 1
                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
            else:
                pos2=options[case](pos_current,i,FRG,pos2,l)

                pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
                initial_node[int(cell_number)] = node_counter + 1
                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
        else:
            pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
            initial_node[int(cell_number)] = node_counter + 1
            pos_new_full = np.vstack((pos_new_full, pos2_full))
            reset_nodes = elevenCol
            if np.array_equal(reset_nodes, pos_new_full[0]):
                pos_new_full = np.delete(pos_new_full, 0, 0)
        
        i += 1

    return pos_new_full, CCM





#def plotter(pos_all, pos_all_b):
def plotter(pos_all):
    # 0    1    2     3      4      5       6       7        8              9          10
    # x    y    z   dir_x  dir_y  dir_z  cell_no  step  previous_node  current_node  branch_no
    
    fig1 = plt.figure(figsize=(30, 30))
    ax3d = fig1.add_subplot(111, projection='3d')
    
    x = pos_all[:, 0] 
    y = pos_all[:, 1] 
    z = pos_all[:, 2]
    
    ax3d.plot(x, y, z, 'r.',markersize=6)
    
    # Cylinder
    x_center=0
    y_center=0
    z_0c=0
    radius=rad
    height=z_bw
    resolution=100
    color='b'
    xc = np.linspace(x_center-radius, x_center+radius, resolution)
    zc = np.linspace(z_0c, z_0c+height, resolution)
    X, Z = np.meshgrid(xc, zc)
    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center 

    ax3d.plot_surface(X, Y, Z, linewidth=0, alpha=0.2, color=color)
    ax3d.plot_surface(X, (2*y_center-Y), Z, linewidth=0, alpha=0.2,color=color)
    
    x_init = pos0[:, 0] 
    y_init = pos0[:, 1] 
    z_init = pos0[:, 2]
    ax3d.plot(x_init, y_init, z_init, 'b.',markersize=12)
# if you want to plot initial CM    
#    x_initCM = CCM[:, 0] 
#    y_initCM = CCM[:, 1] 
#    z_initCM = CCM[:, 2]
#    ax3d.plot(x_initCM, y_initCM, z_initCM, 'ko',markersize=0)
  
    ax3d.set_aspect(3)
    #ax3d.set_xlim([-rad, rad])
    #ax3d.set_ylim([-rad, rad])
    #ax3d.set_zlim([-rad, z_bw+300])
    ax3d.set_xlabel('x, $ \mu m$', fontsize=15, labelpad=20)
    ax3d.set_ylabel('y, $ \mu m$', fontsize=15, labelpad=20)
    ax3d.set_zlabel('z, $ \mu m$', fontsize=15, labelpad=30) 
    
    #ax3d.set_zticks(np.arange(0, z_bw, 100))
    #ax3d.set_xticks(np.arange(-rad, rad, 50))
    #ax3d.set_yticks(np.arange(-rad, rad, 50))
    #ax3d.set_title('%.2f Days' % (step_no * 0.03), fontsize=30)
    #plt.xticks(rotation='vertical')
    #plt.yticks(rotation='vertical')
    ax3d.view_init(elev=4, azim=315)
    #plt.savefig('UnidirectionalMicroTENN' , dpi=500, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig('Step_%d.png' % (step_no + 1), dpi=400, bbox_inches='tight', pad_inches=1)
    plt.show()

#def pos_saver(pos_all, pos_all_b):
def pos_saver(pos_all):
    name_file_fwd = 'UGM_B_'+str(bundle)+'_Ch_'+str(chi)+'_Br_'+str(P)+'_L_' + str(z_bw) +'_Rad_' + str(int(rad)) + '_NCells_' + str(cell_no)
    name_file_fwd_txt = name_file_fwd + '.txt'
    name_file_fwd_mat = name_file_fwd + '.mat'
    
    np.savetxt(name_file_fwd_txt, pos_all)
    sio.savemat(name_file_fwd_mat, {'pos_all': pos_all})
    print('Saved as: '+ name_file_fwd_txt)
    print('Saved as: '+ name_file_fwd_mat)

print('CellNo='+str(cell_no))
print('uTennZ='+str(z_bw))
print('FibRad='+str(FibR))
print('bundle='+str(bundle))
print('Chi='+str(chi))
#Create zeros
elevenCol = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
threeCol = np.array([0, 0, 0])
CCM=threeCol
z_growth = threeCol
branch_step = 0
branch_step_b = 0

if chi==0 and bundle==0:
   case=0

if chi==0 and bundle>0:
   case=1

if chi==1 and bundle==0:
   case=2

if chi==1 and bundle>0:
   case=3

print('Case='+str(case)+' :chi='+str(chi)+', bundle='+str(bundle))

options={0:c0b0,1:c0b1,2:c1b0,3:c1b1,}


starttime = time.clock()



#initialization
pos0 = initial_posXYZ(cell_no, rad, z_fwd)
cent0,n_cl = initial_CM(pos0, rad)
print('No_cl='+str(n_cl))

initial_node = initial_nodes(cell_no)

# From step 0 to step 1, pos_current = pos_previous
step_no = 1
pos_1,CCM = generate_nextXYZ(step_no, pos0, pos0, v_bi, bundle, CCM)
# After step 1, pos_current different from pos_previous
pos_all = np.vstack((pos0, pos_1))
pos_current = pos_1
pos_previous = pos0

#loop over all the rest of the steps
for step_no in range(2, total_step):
    print('Step_No='+str(step_no))
    original_length = len(pos_current)
    a = 0
    while a < len(pos_current):
        if pos_current[a][2]<=z_bw:
           p=0
        else: 
           p = P*(1 - math.exp(-( pos_current[a][2]-z_bw) / (tau_b))) 
        
        if (p >= B1 and random.uniform(0, 1) > B2):
            insert_array = np.hstack((pos_current[a][0:10], 2))
            pos_current = np.insert(pos_current, a + 1, insert_array, 0)
            a += 1
        a += 1
    if len(pos_current) > original_length:
        branch_step = step_no
    else:
        branch_step = branch_step

    pos_new, CCM = generate_nextXYZ(step_no, pos_current, pos_previous, v_bi, bundle, CCM)
    
    pos_all = np.vstack((pos_all, pos_new))

    for m in range(len(pos_new)):
        pos_new[m] = np.hstack((pos_new[m][0:10], 1))

    pos_previous = pos_current
    
    pos_current = pos_new

 
    #if v_bi == 1:
        # if abs(z_max[1] - z_max[2]) < 200:
        #if step_no > 100:
            #v_bi = 4
            #tau = 60


    # if step_no < 400:
        # if (step_no + 1) % 5 == 0:
        #    plotter(pos_all,pos_all_b)
        #    pos_saver(pos_all,pos_all_b)


# Print running time
endtime = time.clock()
print('Loop time = ' + str(endtime - starttime) + 's')


pos_saver(pos_all)

plotter(pos_all)
