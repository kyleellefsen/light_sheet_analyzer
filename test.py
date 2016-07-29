import numpy as np

def inside_or_outside(x,y,z):
    radius = 40
    origin = [100, 100, 100]
    xr = x-origin[0]
    yr = y-origin[1]
    zr = z-origin[2]
    d=np.sqrt(xr**2+yr**2+zr**2)
    inside_or_ouside_image = d < radius
    inside_or_ouside_image = inside_or_ouside_image * (1+(z/50))  # make the sphere brightest on top
    return inside_or_ouside_image

""" Create circle in original (x,y,z) coordinate system. """
size = 500
A = []
y_idx, x_idx = np.indices((size, size))
for i in np.arange(size):
    print(i)
    z_idx = np.ones((size,size))*i
    A.append(inside_or_outside(x_idx, y_idx, z_idx))
A = np.array(A)
A += 1
Window(A.swapaxes(1,2), name='Original')  # x and y axes are flipped from their matrix representations when displaying


### RECREATE THE MOVIE FROM THE OBJECTIVE'S PERSPECTIVE

size = 500
A = []
zp_idx, x_idx = np.indices((size, size))
zp_idx = np.flipud(zp_idx)
zp_idx = zp_idx / np.sqrt(2)
for c in np.arange(80,250):
    print(c)
    yp_idx = np.flipud(zp_idx) - c
    A.append(inside_or_outside(x_idx, yp_idx, zp_idx))
A = np.array(A)
A = A.swapaxes(1,2)
A += 1
Window(A, name='Perspective from light sheet') # x and y axes are flipped from their matrix representations when displaying



### PUT IT BACK TOGETHER AGAIN



mt, mx, my = A.shape
nSteps = mt  #remove this
shift_factor = 1 #remove this
mv = int(np.floor(mt / nSteps))  # number of volumes
A = A[:mv * nSteps]
B = np.reshape(A, (mv, nSteps, mx, my))
B = B.swapaxes(1,3)  # the direction we step is going to be the new y axis, whereas the old y axis will eventually become the z axis
B = np.repeat(B,shift_factor,axis=3)   # We need to stretch the y axis pixels (which were the step size) so that one new y pixel is the same as a pixel in the x direction. Hopefully before this transformation, the step size (ums) is an integer multiple of the x pixel size (um).
# Now our matrix is in terms of (mv, mz, mx, my).
mv,mz,mx,my = B.shape

C = np.zeros((mv, np.ceil(mz/np.sqrt(2)), mx, my), dtype=B.dtype)
for v in np.arange(mv):
    for x in np.arange(mx):
        C[v,:,x,:] = zoom(B[v,:,x,:], (1/np.sqrt(2), 1), order=0)  # squash the z axis pixel size by sqrt(2)
mv,mz,mx,my = C.shape

newy = my+mz # because we will be shifting each x-y plane in the y direction by one pixel, the resulting size will be my plus the number of x-y planes (mz)
D = np.zeros((mv, mz, mx, newy), dtype=A.dtype)
shifted=0
for z in np.arange(mz):
    minus_z = mz-z
    shifted = minus_z
    D[:,z,:,shifted:shifted+my] = C[:,z,:,:]
#D=D[:,::-1,:,:] # (mv, mz, mx, my)

Window(np.squeeze(D))
