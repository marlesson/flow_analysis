import math
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)

    r = float(max(0, (ratio - 1)))
    b = float(max(0, (1 - ratio)))
    g = 1 - b - r

    return r, g, b

def rad2rgb(rad):
    #color = floatRgb(rad, -math.pi, math.pi)
    #rgb(-math.pi, math.pi, rad)
    color = rgb(-math.pi, math.pi, rad)
    #colot = [0.9604165758394343, 0.039583424160565706, 0.0]
    return color

def floatRgb(mag, cmin, cmax):
    """
    Return a tuple of floats between 0 and 1 for the red, green and
    blue amplitudes.
    """
    try:
      # normalize to [0,1]
      x = float(mag-cmin)/float(cmax-cmin)
    except:
      # cmax = cmin
      x = 0.5
    blue = min((max((4*(0.75-x), 0.)), 1.))
    red  = min((max((4*(x-0.25), 0.)), 1.))
    green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
    return [int(red), int(green), int(blue)]
    
def cart2pol(x, y):
    """Convert from Cartesian to polar coordinates.

    Example
    -------
    >>> theta, radius = pol2cart(x, y)
    """
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return theta, radius

def compass(angles, radii, arrowprops=dict(color='darkorange', linewidth=2)):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    """

    #angles, radii = cart2pol(u, v)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    #kw = 
    #if arrowprops:
    #    kw.update(arrowprops)
    [ax.annotate("", xy=(-angle, radius), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", linewidth=2,  color=rad2rgb(angle))) for
     angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax

   
def plot_arrows(df, shape):
    plt.clf()
    
    #Plt
    plt.figure(figsize=(15,10))

    plt.xlim((0,shape[0]))
    plt.ylim((0,shape[1]))

    ax = plt.gca()
    ax.invert_yaxis()

    plt.quiver(df.x1.values, df.y1.values, 
               df.x.values, df.y.values,  
               angles='xy', scale_units='xy', color=[rad2rgb(r) for r in df.angle.values])
    return plt


def filter_vec(df, x, y, d):
    return df[(df.x1 >= x-d) & (df.x1 <= x+d) & (df.y1 >= y-d) & (df.y1 <= y+d)]

def next_pos(shape, x, y, d):
    x_max, y_max = shape
    
    x = x + d*2 + 1
    
    if x > x_max:
        x = int(d*2-1)
        y = y + d*2 + 1

    if y >= y_max:
        return [False, x, y]
    
    return [True, x, y]

def block_partition(df, shape, d):
    x_max, y_max = shape
    res = True
    x = int(d*2-1)
    y = int(d*2-1)
    new_vecs = []

    while res:
        df_filter = filter_vec(df, x, y, d)
        if len(df_filter) > 0:
            row   = df_filter.agg('sum')
            angle = math.atan2(row.y, row.x)
            size  = math.sqrt(row.x**2+row.y**2)#/len(df_filter)
            x1    = int(size*math.cos(angle))
            y1    = int(size*math.sin(angle))

            new_vecs.append([x, y, x1, y1, angle, size])

        res, x, y = next_pos(shape, x, y, d)
    return pd.DataFrame(new_vecs, columns=['x1', 'y1', 'x', 'y', 'angle', 'size'])
