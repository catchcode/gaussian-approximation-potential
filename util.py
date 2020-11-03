import sys
import math
import numpy as np
import matplotlib.pyplot as plt

angstrom = 1.0
bohr = 0.529177210903 * angstrom

electron_volt = 1.0
hartree = 27.211386245988 * electron_volt
rydberg = hartree/2

def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def gaussian(x, mu, sigma):
    '''Gaussian normal distribution functions'''
    return np.exp(-0.5*((x-mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)

def xyz2r(xyz):
    if np.array(xyz).shape == (3,):
        x, y, z = xyz
    elif np.array(xyz).shape[-1] == 3:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    else:
        raise ValueError("Cannot identify the structure of xyz: ", xyz)
    return np.sqrt(x**2 + y**2 + z**2) 

def xyz2spherical(xyz):
    '''Convert Cartesian coordinates to spherical polar coordinates.
    The latitude theta is defined from the Z-axis down.
    The angles are in radian unit.
    '''
    r_theta_phi = np.zeros(xyz.shape, dtype=np.float64)
    if len(xyz.shape) == 1:
        x, y, z = xyz
    else:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    xy = x**2 + y**2
    r_theta_phi[:, 0] = np.sqrt(xy + z**2)          # r
    r_theta_phi[:, 1] = np.arctan2(np.sqrt(xy), z)  # theta
    r_theta_phi[:, 2] = np.arctan2(y, x)    # phi
    return r_theta_phi

def cutoff_function(r, r_cut=5.0, dr=1.0):
    '''A smooth cutoff function with compact support
    r_cut: cutoff radius
    dr: smoothness
    '''
    if np.array(r).shape == (): #
        if r > r_cut:
            value = 0.0
        elif r > r_cut - dr:
            value = (1.0 + math.cos(math.pi*(r - (r_cut - dr))/dr))/2
        elif r >= 0:
            value = 1.0
        else:
            raise ValueError(f"Unrecognized value of r: {r}")
        return value
    elif np.array(r).shape[0] >= 1:
        return np.array([cutoff_function(r_i, r_cut, dr) for r_i in r])
    else:
        raise TypeError(f"Unidentified type of r: {r}")

def grad_r(xyz):
    '''Gradient of r = sqrt(x**2 + y**2 + z**2)'''
    if np.array(xyz).shape == (3,):
        #return xyz / np.linalg.norm(xyz)
        return xyz / xyz2r(xyz) 
    elif np.array(xyz).shape[-1] == 3:
        #return np.array([xyz_i / xyz2r(xyz_i) for xyz_i in xyz])
        return np.array([grad_r(xyz_i) for xyz_i in xyz])
    else:
        raise TypeError(f"Unidentified type of xyz: {xyz}") 

def grad_cutoff_function(xyz, r_cut=5.0, dr=1.0):
    '''Gradient of the cutoff function
    '''
    if np.array(xyz).shape == (3,):
        #r = np.linalg.norm(xyz)
        r = xyz2r(xyz)
        if r > r_cut:
            r_deriv = 0.0
        elif r > r_cut - dr:
            r_deriv = (- math.pi/dr * math.sin(math.pi*(r - (r_cut - dr))/dr))/2
        elif r >= 0:
            r_deriv = 0.0
        else:
            raise ValueError
        return r_deriv * grad_r(xyz)
    elif np.array(xyz).shape[-1] == 3:
        return np.array([grad_cutoff_function(xyz_i, r_cut, dr) for xyz_i in xyz])
    else:
        raise TypeError(f"Unidentified type of xyz: {xyz}") 

def plot_matrix(matrix):
    '''Visualize a matrix with heatmap
    '''
    plt.matshow(matrix)
    plt.colorbar()
    plt.show()


def print_matrix(name, matrix):
    '''
    Print a matrix
    '''
    import pandas as pd
    print(name, pd.DataFrame(matrix), end='\n\n')


def benchmark_notes():
    """
    #1   4.0 msec per integration
    #def radial_basis(n, r, r_cut=R_CUT, n_max=N_MAX, sigma_atom=SIGMA_ATOM):
    #    return norm.pdf(r, loc=r_cut*(n/n_max), scale=sigma_atom)

    #2   0.72-0.81 msec per integration
    #def gaussian(x, mu, sigma):
    #    return np.exp(-0.5*((x-mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)
    #
    #def radial_basis(n, r, r_cut=R_CUT, n_max=N_MAX, sigma_atom=SIGMA_ATOM):
    #    return gaussian(r, r_cut*(n/n_max), sigma_atom)

    #3   0.69-0.72 msec
    #def radial_basis(n, r, r_cut=R_CUT, n_max=N_MAX, sigma_atom=SIGMA_ATOM):
    #    return np.exp(-0.5*((r-r_cut*(n/n_max))/sigma_atom)**2) / (np.sqrt(2*np.pi)*sigma_atom)

    #4   0.66-0.69 msec
    #def radial_basis(n, r):
    #    return np.exp(-0.5*((r-R_CUT*(n/N_MAX))/SIGMA_ATOM)**2) / (np.sqrt(2*np.pi)*SIGMA_ATOM)

    #5   0.55-0.62 msec
    #@lru_cache(maxsize=8192)
    # def radial_basis(r, n):
    #    '''Radial basis functions'''
    #    return np.exp(-0.5*((r-R_CUT*(n/N_MAX))/SIGMA_ATOM)**2)
    """

def dotprod(array1, array2):
    '''Evalute the dot product of two arrays of the same shape.'''
    arr1, arr2 = array1, array2
    if not ( isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray) ):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
    if arr1.shape != arr2.shape:
        raise ValueError("shape mismatch of two vectors/arrays: \
            {} is not the same shape as {}".format(arr1.shape, arr2.shape))
    return np.sum(arr1 * arr2)  

def two_Re_dotprod(array1, array2):
    '''Used when applying chain rule to evaluate the derivative of real function 
    f(z, z^*) = f^*(z, z^*) with arguments of z and its conjugate. 
    Explicitly, df/dx = ∂f/∂z * dz/dx + ∂f/∂z^* * dz^*/dx = 2 Re ∂f/∂z * dz/dx 
    = 2 Re dotprod(∂f/∂z, dz/dx) = two_Re_dotprod(∂f/∂z, dz/dx)
    '''
    return 2*dotprod(array1, array2).real

def eval_rel_error(approx, target):
    '''Evalute the relative error of approximated value/vector w.r.t. target/true one '''
    return np.linalg.norm(approx-target)/np.linalg.norm(target)

