import numpy as np
import cv2
import time
from scipy import signal

# def draw_flow(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
#     fx, fy = flow[y,x].T
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     cv2.polylines(vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (x2, y2) in lines:      
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis

def optical_flow(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25

    w   = window_size/2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'

    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode)+\
         signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)

    # within window window_size * window_size
    
    for i in range(int(w), int(I1g.shape[0]-w)):
        for j in range(int(w), int(I1g.shape[1]-w)):
            
            Ix = fx[int(i-w):int(i+w+1), int(j-w):int(j+w+1)].flatten()
            Iy = fy[int(i-w):int(i+w+1), int(j-w):int(j+w+1)].flatten()
            It = ft[int(i-w):int(i+w+1), int(j-w):int(j+w+1)].flatten()
            #b = ... # get b here
            #A = ... # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = ... # get velocity here
            u[i,j]=nu[0]
            v[i,j]=nu[1]

    return (u,v)

if __name__ == '__main__':
    import sys
    print(sys.argv[1])
    try: fn = sys.argv[1]
    except: fn = 0

    cam = cv2.VideoCapture(fn)
    
    ret, prev   = cam.read()
   
    prevgray    = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cur_glitch  = prev.copy()
    
    while True:
        ret, img = cam.read()
        vis      = img.copy()
        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optical Flow
        optical_flow(prevgray, gray, 2)
        cv2.imshow('frame',gray)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()