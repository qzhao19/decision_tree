import numpy as np

def sort(x, y, start, end, reverse=True):
    x_y = [(x[i], y[i]) for i in range(len(x))]
    
    if not reverse:
        x_y[start:end] = sorted(x_y[start:end], key=lambda z: z[0])
    else:
        x_y[start:end] = sorted(x_y[start:end], key=lambda z: z[0], reverse=True)
    
    x[:] = [z[0] for z in x_y]
    y[:] = [z[1] for z in x_y]

    return np.array(x), np.array(y)