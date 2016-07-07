
def inside_or_outside(x,y,z):
    radius = 40
    origin = [100, 100, 100]
    x = x-origin[0]
    y = y-origin[1]
    z = z-origin[2]
    d=np.sqrt(x**2+y**2+z**2)
    return d<radius



size = 500
A = []
for i in np.arange(size):
    print(i)
    x_idx = np.ones((size,size))*i
    y_idx, z_idx = np.indices((size, size))
    A.append(inside_or_outside(x_idx, y_idx, z_idx))
A = np.array(A)
Window(A)



size = 500
A = []
for c in np.arange(0,size*3,3):
    print(c)
    z_idx, x_idx = np.indices((size, size))
    z_idx = np.flipud(z_idx)
    z_idx = z_idx / np.sqrt(2)
    y_idx = c - z_idx
    A.append(inside_or_outside(x_idx, y_idx, z_idx))
A = np.array(A)
Window(A)