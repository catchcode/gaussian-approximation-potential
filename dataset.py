import math
from timeit import default_timer as timer
import numpy as np
from scipy.special import sph_harm
from scipy.linalg import cho_solve, cho_factor
import ase.io
import ase.neighborlist
import pandas as pd
from util import *
from descriptor import soap
from harmonics import SphericalHarmonicYCartesian, gradSphericalHarmonicYCartesian

pd.options.display.float_format = '{:,.3f}'.format
np.set_printoptions(precision=3, suppress=True)
plot = False
timing = False
profiler = False
precision = 'double'

# Upper case E implies that molecular energy E consists of atomice energies e's
# Lower case e/f refers to the atomic energy/force
# In a similar spirit, molecular variables R, Q, and so on below denote collective sets of atomic variables r, q, etc.
calc_e_from_E   = True  # Basic mode of computation
f_data_given    = False
calc_e_from_Ef  = calc_e_from_E and f_data_given # calculate e from E & f data
calc_f_from_rep = False # calculate f from the precomputed representative descriptors and alpha
calc_ef_from_E  = calc_e_from_E and calc_f_from_rep
calc_ef_from_Ef = calc_e_from_Ef and calc_f_from_rep
calc_ef_from_Ef_experimental = False    # same as calc_ef_from_Ef but with an experimental scheme 
                                        # extending a sparse cov matrix with force terms  

if precision == 'single':
    float_dtype = np.float32
    complex_dtype = np.complex64
elif precision == 'double':
    float_dtype = np.float64
    complex_dtype = np.complex128
else:
    float_dtype = np.float
    complex_dtype = np.complex


class dataset:
    '''dataset management class'''

    def __init__(self, xyzfilename):
        '''Load dataset given as an xyz format file

        input:
            xyzfilename: xyz format file
        output:
            ase.atoms class which has lattice vectors, atomic positions,
            forces, energy, stress tensor
        '''

        self.filename = xyzfilename
        self.checksum = None
        self.xyzs = None               # (Structure, Energy) = SE
        self.ndata = None              # number of SE data points
        self.total_energy_list = None
        self.total_force_list = None   # atomic forces
        self.natoms_list = None        # number of atoms
        self.y = None                  # total energy vector (total energy list???)
        self.chemical_symbol_to_energy = {
            'H': -16.705003113816887,
            'C': -258.9727666548056,
            'N': -393.6962697601746,
            'O': -578.760929138896,
            'Ni': -11667.,
            }

        # Import hashlib library (md5 method is part of it)
        import hashlib

        # Open, close, read file and calculate MD5 on its contents
        with open(self.filename, 'r') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            self.checksum = hashlib.md5(data.encode('utf-8')).hexdigest()

        self.xyzs = ase.io.read(xyzfilename, index=':')
        #self.xyzs = self.xyzs[:11]
        self.ndata = len(self.xyzs) # D if calc_e_from_Ef == False
        print('number of molecular energies =', self.ndata)

        try:
            self.total_energy_list = np.array([xyz.get_total_energy() for xyz in self.xyzs], dtype=float_dtype)
            if calc_e_from_Ef or calc_f_from_rep:
                self.total_force_list = np.vstack([np.array(xyz.get_forces(), dtype=float_dtype) for xyz in self.xyzs])
        except RuntimeError:
            self.total_energy_list = np.array([0 for xyz in self.xyzs], dtype=float_dtype)
            if calc_e_from_Ef or calc_f_from_rep:
                self.total_force_list = np.vstack([np.zeros((len(xyz),3), dtype=float_dtype) for xyz in self.xyzs])
        except Exception as e:
            # reraise for any other exception
            print(e)
            raise
        self.natoms_list = np.array([len(xyz) for xyz in self.xyzs], dtype=np.int32)
        # y, total energy list (offset by atomic energy)
        chemical_symbols_list = [xyz.get_chemical_symbols() for xyz in self.xyzs]
        self.y0 = [sum([self.chemical_symbol_to_energy[symbol] for symbol in symbols]) for symbols in chemical_symbols_list]
        self.y = self.total_energy_list - self.y0
        if calc_e_from_Ef:
            #self.y = np.append(self.y, -self.total_force_list.flat)   
            # numpy.ravel() is faster than numpy.ndarray.flatten(); the former is equivalent to numpy.reshape(-1)
            # numpy.ndarray.flat is an iterator version of the above; superfast
            # in this case, however, there is no need to do this
            self.y = np.append(self.y, -self.total_force_list)  # the negative sign '-' must be included

