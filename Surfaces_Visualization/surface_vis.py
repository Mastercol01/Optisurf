import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, art3d

def quickplot2D(x_axis, y_axis, title,
                 xlabel, ylabel, labels=None,
                 markers='-o', fontsize=12, 
                 figsize=(12,7), xlim=None, ylim=None):
        
        try:
            num_y_axis=len(y_axis)
        except:
            num_y_axis=1
            y_axis=[y_axis]
            labels=[labels]
            markers=[markers]
            
        lenmark=len(markers)
        if (lenmark<num_y_axis):
            complist=[ markers[lenmark-1] for _ in range(num_y_axis-lenmark) ]
            markers.extend(complist)
            
        lenlabs=len(labels)
        if (lenlabs<num_y_axis):
            complist=[ labels[lenlabs] for _ in range(num_y_axis-lenlabs) ]
            labels.extend(complist)

        plt.figure(figsize=figsize) 
        
        for i in range(num_y_axis):
            plt.plot(x_axis, y_axis[i], markers[i], label=labels[i])
            
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.xticks(rotation=60)
        plt.xlim(xlim)
        plt.ylim(ylim)
        if (labels!=None):
            plt.legend()
        plt.show()
        



def quickplot3D(x_axis, y_axis, z_vals, 
                xlabel='X', ylabel='Y', zlabel='Z',
                xlim=None, ylim=None, zlim=None,
                axis_view=(25, 30)):

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_axis, y_axis)
    
    ax1.plot_surface(X, Y, z_vals)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_zlabel(zlabel)
    
    elevation, azimuth=axis_view
    
    ax1.view_init(elevation, azimuth)
    
    plt.show()
    


def vismesh1(vertices, faces, xlabel='X', 
             ylabel='Y', zlabel='Z', colors=None, 
             axis_view=(25, 30)):

    v=vertices
    f=faces
    C=colors
    
    min_x, max_x=v[:,0].min(),v[:,0].max()
    min_y, max_y=v[:,1].min(),v[:,1].max()
    min_z, max_z=v[:,2].min(),v[:,2].max()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    
    if(colors!=None): 
            norm = plt.Normalize(C.min(), C.max())
            colors = plt.cm.viridis(norm(C))
            pc = art3d.Poly3DCollection(v[f], facecolors=colors, edgecolor="black")
            ax.add_collection(pc)
    else:
#        pc = art3d.Poly3DCollection(v[f],edgecolor="black")
        pc = art3d.Poly3DCollection(v[f])
        ax.add_collection(pc)
        
    buffer=0.5    
    ax.set_xlim(min_x-buffer,max_x+buffer)
    ax.set_ylim(min_y-buffer,max_y+buffer)
    ax.set_zlim(min_z-buffer,max_z+buffer)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    elevation, azimuth=axis_view
    ax.view_init(elevation, azimuth)
    
    plt.show()
    
