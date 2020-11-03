import math
from timeit import default_timer as timer
import numpy as np
from scipy.special import sph_harm
from scipy.linalg import cho_solve, cho_factor
import ase.io
import ase.neighborlist
import pandas as pd
from util import *
from harmonics import SphericalHarmonicYCartesian, gradSphericalHarmonicYCartesian

# sys.tracebacklimit = 0
pd.options.display.float_format = '{:,.3f}'.format
np.set_printoptions(precision=3, suppress=True)
plot = False
timing = False
profiler = False
precision = 'double'

class soap:
    '''GAP Model'''

    def __init__(self, 
                lmax=6, nmax=6,         # 
                rcut=5.0, dr=1.0,
                sigma_r=0.5, 
                sigma_w=1.0,
                sigma_nu_energy=0.0001, sigma_nu_force=0.01, 
                zeta=4, 
                basis_tolerence=10, 
                calc_force = False,
                ):
        self.lmax = lmax
        self.nmax = nmax
        self.sigma_r = sigma_r
        self.alpha_atom = 0.5 / sigma_r**2
        self.basis_tolerence = basis_tolerence
        self.basis_rcut = rcut #+ sigma * math.sqrt(2 * basis_tolerence * math.log(10))
        self.rcut = rcut
        self.dr = dr
        self.zeta = zeta
        self.sigma_nu_energy = sigma_nu_energy
        self.sigma_nu_force = sigma_nu_force
        self.sigma_jitter = 1E-8    # Regularization parameter for the sparse covariance matrix 

        # exp_iota[l, n]: Gaussian \times Modified spherical Bessel function of the first kind
        #                 at z = 2 * alpha * r_n * r_ij
        self.exp_iota = np.empty((lmax, nmax), dtype=float_dtype)
        self.Yc = np.empty((lmax, 2*lmax+1), dtype=complex_dtype)
        self.r_n = [self.basis_rcut * n / nmax for n in range(nmax)]

        # gradient of exp_iota
        if calc_e_from_Ef or calc_f_from_rep:
            self.grad_Yc = np.empty((3, lmax, 2*lmax+1), dtype=complex_dtype)
            self.grad_exp_iota = np.empty((3, lmax, nmax), dtype=float_dtype)

        # Evaluate basis function at ri
        # $phi_j(ri) = e^(-alpha*(ri-rj)**2)$
        self.C = np.array(
            [[math.exp(-self.alpha_atom * (self.r_n[i] - self.r_n[j])**2)
              for j in range(nmax)]
             for i in range(nmax)], dtype=float_dtype)

        # Overlap matrix between two Gaussian radial basis functions
        # $\int_{0}^{\infty} dr r^2 phi(r-ri) * phi(r-rj)$
        integral_prefactor = 1 / math.sqrt(2**7 *self.alpha_atom**5)
        self.S = np.array(
            [[self.overlap_integral_between_radial_basis(n, m)
              for m in range(nmax)]
             for n in range(nmax)], dtype=float_dtype) * integral_prefactor

        # Basis set transformation matrix
        # B^T and T^T will be used in the coefficient() function
        # as x@B and x@T (left-multiplication) to calculate
        # radial coefficients in orthogonal basis space.

        # B: Orthonormal basis -> Gaussian basis
        # B: g(r) -> phi(r)
        # B^T: Gaussian basis representation -> Orthonormal basis representation
        # B^T: c' -> c
        self.B = np.linalg.cholesky(self.S)

        # T^T: Radial function, R(r) -> Orthonormal basis representation, c (???)
        # T^T: R(r) -> c
        self.T = cho_solve(cho_factor(self.C), self.B)

        self.k_IM = None
        self.alpha = None
        self.ndata = None
        self.y = None
        self.natoms_all = None
        self.natoms_max = None
        self.rep_atoms = None
        self.qall = None
        if calc_e_from_Ef or calc_f_from_rep:
            self.call = None
            #self.gradqall = None
        self.atom_energy = None

    # atomic weights
    def symbol_to_weight(self, symbol):
        if symbol == 'H':
            weight = 0.2
        elif symbol == 'C':
            weight = 0.5
        elif symbol == 'N':
            weight = 0.7
        elif symbol == 'O':
            weight = 0.9
        elif symbol == 'Ni':
            weight = 1.0
        else:
            weight = 1.0
        return weight

    def __str__(self):
        return 'lmax=%d, nmax=%d, sigma=%f, rcut=%f, zeta=%d, sigma_nu_energy=%f' % (
            self.lmax, self.nmax, self.sigma, self.rcut,
            self.zeta, self.sigma_nu_energy)

    def export_descriptors_as_csv(self, filename="descriptors.csv"):
        '''
        Export flattened descriptors as a csv format file.

        How to import a csv file:

        import pandas as pd
        df = pd.read_csv("./descriptors.csv")
        '''
        natoms, nmax, nmax, lmax = self.qall.shape
        a = self.qall.reshape(natoms, nmax * nmax * lmax)
        pd.DataFrame(a).to_csv(filename, index=False)
        print('descriptors are exported as descriptors.csv')

    def export_descriptors_as_npy(self, filename="descriptors.npy"):
        '''
        Export flattened descriptors as numpy binary format.

        You can import this descriptor matrix as follows:

        import numpy as np
        a = np.load('descriptors.npy')

        Here is a 30x343 matrix.
        Each row vector of the matrix represents a precomputed atomic descriptors for silicon bulk.
        Each descriptor vector is represented by 343 real numbers.
        This dataset has some redundancy.
        Using k-means clustering, find all redundant data.
        More specifically, find a list of indexes of non-redundant rows.

        a.shape
        (30, 343)
        from util import plot_matrix
        plot_matrix(a)
        '''
        natoms, nmax, nmax, lmax = self.qall.shape
        a = self.qall.reshape(natoms, nmax * nmax * lmax)
        np.save(filename, a)
        print('descriptors are exported as descriptors.npy')

    def export_descriptors_as_json(self, filename="descriptors.json"):
        '''Export flattened descriptors as JSON'''
        import codecs
        import json
        natoms, nmax, nmax, lmax = self.qall.shape
        a = self.qall.reshape(natoms, nmax * nmax * lmax)
        b = a.tolist()    # convert np.array to list
        json.dump(
            b, codecs.open(filename, 'w', encoding='utf-8'),
            separators=(',', ':'), sort_keys=True, indent=4)
        print('descriptors are exported as descriptors.json')

        # In order to "unjsonify" the array use:

        # obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
        # b_new = json.loads(obj_text)
        # a_new = np.array(b_new)

    def fit(self, train_dataset: dataset):
        '''Solve GAP equation'''

        # Below three objects are about dataset
        # number of data
        #self.dataset = train_dataset
        self.ndata = train_dataset.ndata

        # list of the number of atoms
        self.natoms_list = [len(xyz) for xyz in train_dataset.xyzs]

        # list of chemical symbols in train dataset
        self.symbol_list = [xyz.get_chemical_symbols() for xyz in train_dataset.xyzs]

        # total number of atoms
        self.natoms_all = sum(self.natoms_list)
        self.natoms_max = max(self.natoms_list)
        
        self.atom_weights_list = np.zeros((self.ndata, self.natoms_max))
        for i, symbols_data in enumerate(self.symbol_list):
            for j, symbol in enumerate(symbols_data):
                self.atom_weights_list[i, j] = self.symbol_to_weight(symbol)
        # print(self.atom_weights)

        # total energy vector y
        self.y = train_dataset.y
        self.y0 = train_dataset.y0

        
        # the whole expansion coefficients
        if calc_e_from_Ef or calc_f_from_rep:
            self.call = np.zeros((0, self.nmax, self.lmax, 2*self.lmax+1), dtype=complex_dtype)
            #self.gradqall = np.array([])

        # descriptors q
        self.qall = np.zeros((0, self.nmax, self.nmax, self.lmax), dtype=complex_dtype)

        # dataset -> descriptors
        # fe.xyz: 1541.83 seconds for 25272 integrations, 61.01 milli-seconds per integration
        # now it takes 2.43 seconds (635x faster).
        try:
            self.qall = np.load('q.npy')
            #self.gradqall = np.load('gradq.npy')
        except IOError:
            for i, xyz in enumerate(train_dataset.xyzs):
                if calc_e_from_Ef or calc_f_from_rep:
                    c = self.coefficients(xyz, self.atom_weights_list[i])
                    #print(self.atom_weights_list[i],end='\n')
                    q = self.c2q(c)
                    #gradq = self.R_grad_Q(xyz, c, q, self.atom_weights_list[i])    
                    self.call = np.vstack((self.call, c))
                    self.qall = np.vstack((self.qall, q))
                    # dimension of gradq differs case by case according to number of entries of xyz
                    #self.gradqall = np.append((self.gradqall, gradq))
                else:
                    self.qall = np.vstack((self.qall, self.descriptors(xyz, self.atom_weights_list[i])))
                
        except Exception as e:
            # reraise for any other exceptions
            print(e)
            raise

        # TODO: Create an interface function export() for user
        # self.export_descriptors_as_csv()
        # self.export_descriptors_as_npy()
        # self.export_descriptors_as_json()

        # TODO: Sparsification
        # Get the list of M representative atomic environments
        # using k-means clustering or CUR decomposition.
        # Without sparcification, K becomes singular, for example, for a dimer.
        # The environments for both atoms in the dimer are exactly the same due to symmetry.
        # Until sparsification is implemented, we manually select one atom in each dimer.

        N = self.natoms_all
        all_atoms = range(N)
        if self.rep_atoms is None:
            self.rep_atoms = all_atoms
        M = len(self.rep_atoms)

        # K_MM, K_MN: covariance matrix
        # M: number of representative local atomic environments
        # N: number of all local atomic environments

        # K_MN, K_NM, K_MM
        # M: number of representative atomic env
        K_MM = np.zeros((M, M), dtype=float_dtype)
        for i in range(M):
            atom_i = self.rep_atoms[i]
            for j in range(M):
                atom_j = self.rep_atoms[j]
                K_MM[i,j] = self.covariance(self.qall[atom_i], self.qall[atom_j])
        if plot:
            plot_matrix(K_MM)
        
        # Regularization of sparse covariance matrix K_MM
        # see Bartok and Csanyi, Int. J. Quant. Chem. (2015)
        for i in range(M):
            K_MM[i,i] += self.sigma_jitter

        # L_ND: linear (differential operator) matrix connecting y with y'
        # y = L.T*y’ where y’ is a vector of atomic energies $\epsilon$
        # N: number of all local atomic environments
        # D: number of input data components (energies, forces and stresses)

        if calc_e_from_Ef:
            D = self.ndata + 3*self.natoms_all
        else:
            D = self.ndata
        
        KL_MD = np.zeros((M, D), dtype=float_dtype)

        for i in range(M):
            atom_i = self.rep_atoms[i]
            k_start = 0
            for j in range(self.ndata): # self.ndata = number of E's
                cov_ei_Ej = 0
                k_end = k_start + self.natoms_list[j]   # self.natoms_list[j] = number of e_jk's in E_j
                for k in range(k_start, k_end):
                    cov_ei_ek = self.covariance(self.qall[atom_i], self.qall[k])
                    cov_ei_Ej += cov_ei_ek  # E_j = e_j1 + ... + e_jk + ... + e_j(k_end)
                KL_MD[i,j] = cov_ei_Ej
                k_start = k_end

        if calc_e_from_Ef:
            e_idx_start = 0
            f_idx_start = self.ndata
            for j in range(self.ndata): # E_j
                xyz = train_dataset.xyzs[j] # j-th molecule
                weights = np.array([self.symbol_to_weight(symbol) for symbol in xyz.get_chemical_symbols()])
                e_idx_end = e_idx_start + self.natoms_list[j]
                f_idx_end = f_idx_start + 3*self.natoms_list[j]
                # collective set of c/q's of xyz's in a molecule
                c_mol = self.call[e_idx_start:e_idx_end]
                q_mol = self.qall[e_idx_start:e_idx_end]
                #gradq = self.R_grad_Q(xyz, c_mol, q_mol, self.atom_weights_list[j])
                gradq = self.R_grad_Q(xyz, c_mol, q_mol, weights)
                for i in range(M):
                    atom_i = self.rep_atoms[i]
                    KL_MD[i, f_idx_start:f_idx_end] = self.R_grad_COV_q(self.qall[atom_i], q_mol, gradq).flat
                e_idx_start, f_idx_start = e_idx_end, f_idx_end        
        
        if plot:
            plot_matrix(KL_MD)

        # Lambda_DD: regularization matrix
        if calc_e_from_Ef:
            diag_list = np.concatenate([self.sigma_nu_energy**2 * np.ones(self.ndata),
                                        self.sigma_nu_force**2 * np.ones(3*self.natoms_all)])
            #Lambda_DD = np.diag(diag_list)
            Lambda_DD_I = np.diag(1/diag_list)
        else:
            #Lambda_DD = np.identity(self.ndata, dtype=float_dtype) * self.sigma_nu_energy**2
            Lambda_DD_I = np.identity(self.ndata, dtype=float_dtype) / self.sigma_nu_energy**2
        #Lambda_DD_I = np.linalg.inv(A)

        # alpha: coefficient vector
        # $ \alpha_M =
        #   [K_{MM} + K_{MN} L_{ND} \Lambda^{-1}_{DD} L^{T}_{DN} K_{NM}]^{-1}
        #   K_{MN} L_{ND} \Lambda^{-1}_{DD} y_{D} $
        #self.alpha = np.linalg.pinv(
        #    K_MM + K_MN @ L @ Lambda_DD_I @ LT @ K_MN.T) @ K_MN @ L @ Lambda_DD_I @ self.y
        #F = K_MM + K_MN @ L @ Lambda_DD_I @ LT @ K_MN.T

        F = K_MM + KL_MD @ Lambda_DD_I @ KL_MD.T
        self.alpha = np.linalg.pinv(F) @ KL_MD @ Lambda_DD_I @ self.y

        if plot:
            plot_matrix(F)
            print('condition number ={0: .1e}'.format(np.linalg.cond(F)))

        # print('K_MM', pd.DataFrame(K_MM), end='\n\n')
        # print('K_MN', pd.DataFrame(K_MN), end='\n\n')
        # print('L', pd.DataFrame(L), end='\n\n')
        # print('L.T', pd.DataFrame(LT), end='\n\n')
        # print('K_MN * L_ND', pd.DataFrame(K_MN @ L), end='\n\n')
        #if calc_e_from_Ef:
        #    print('KL_MD', pd.DataFrame(KL_MD[:,self.ndata:]), end='\n\n')
        # print('Lambda_DD_I', pd.DataFrame(Lambda_DD_I), end='\n\n')
        # print('K_MN.T', pd.DataFrame(K_MN.T), end='\n\n')
        # print('F', pd.DataFrame(F), end='\n\n')
        # print('F@FI', pd.DataFrame(F@FI), end='\n\n')

    # computation process: 
    # xyz_i = displacements of environment of an atom in a molecule
    # xyz = xyz_i's of atoms in a molecule
    # xyzs = list of xyz/molecule instances
    # c = ci's of atoms in a molecule; q = qi's of atoms in a molecule
    # epsilon = component energy of an atom
    # 
    # total_energies[
    #   xyzs -> molecule_energy[
    #               xyz -> descriptors[coefficients[xyz->(ci[xyz_i])->c] -> c2q[c->q]] 
    #               -> atomic_energies[q->epsilons]
    #           ]
    #    -> E(sum of epsilons)'s
    # ]
    def ci(self, xyz_ij_list, rel_weights):
        '''Calculate the expansion coefficients (c^{i}_{nlm}) for atom i in a molecule'''
        # initialization: don't use np.empty instead of np.zeros
        _ci = np.zeros((self.nmax, self.lmax, 2*self.lmax+1), dtype=complex_dtype)
        # central atom
        _ci[:,0,0] = self.B[0,:] / np.sqrt(4*np.pi) #* 4*np.pi
        # neighboring atoms
        if np.array(xyz_ij_list).shape == (3,): # in case xyz_ij_list has only one vector, i.e. xyz_ij_list = (x,y,z)
            xyz_ij_list = [xyz_ij_list]
        if np.array(rel_weights).shape == ():
            rel_weights = [rel_weights]
        for i, xyz_ij in enumerate(xyz_ij_list):
            r_ij = xyz2r(xyz_ij)
            self.calc_exp_iota(r_ij)
            ci_ln = rel_weights[i] * self.exp_iota @ self.T #* 4*np.pi
            for l in range(self.lmax):
                for m in range(-l, l+1):
                    # self.Yc[l, m+l] = sph_harm(m, l, r[ij, 2], r[ij, 1]).conjugate()
                    # the following is slightly better in terms of accuracy of computed energy
                    self.Yc[l, m+l] = SphericalHarmonicYCartesian(l, m, xyz_ij).conjugate()
            for n in range(self.nmax):
                for l in range(self.lmax):
                    for m in range(-l, l+1):
                        _ci[n, l, m+l] += ci_ln[l, n] * self.Yc[l, m+l]
        return _ci

    def coefficients(self, xyz, weights):
        '''Calculate the expansion coefficients (c^{i}_{nlm}) for all atoms in a molecule'''
        n_atoms = len(xyz)
        c = np.zeros((n_atoms, self.nmax, self.lmax, 2*self.lmax+1), dtype=complex_dtype)
        center_id_list, neighbor_id_list, xyz_ij_list = ase.neighborlist.neighbor_list(
            'ijD', xyz, self.rcut)

        for i in range(n_atoms): # ID of atom i
            atom_i_idx = list(np.where(center_id_list == i)[0])
            atom_j_id_list = neighbor_id_list[atom_i_idx]
            rel_weight = 2 * weights[atom_j_id_list] / (weights[atom_j_id_list] + weights[i])
            c[i] = self.ci(xyz_ij_list[atom_i_idx], rel_weight)

        # avg_num_of_neighbors = len(atom_index)/atom_count
        # print('avg num of neighbors =', int(round(avg_num_of_neighbors)))

        return c

    def grad_ci(self, xyz_ij_list, rel_weights, pos='center'):   
        '''Calculate the gradient of (c^{i}_{nlm}) of atom i w.r.t. the position of atom j in a molecule.
        Beware that j can be different from i'''
        #center_idx, neighbor_idx, displacements = ase.neighborlist.neighbor_list('ijD', xyzs, self.rcut)
        #if pos == 'center':
        #    xyz_ij_list = displacements[center_idx == i]  # neighbor atoms around atom_i (= atom_k's)
        #else: #pos == 'neighbor'
        #    xyz_ij_list = displacements[neighbor_idx == k]    # only atom_k (!= atom_i)
        grad_c = np.zeros((3, self.nmax, self.lmax, 2*self.lmax+1), dtype=complex_dtype)
        if np.array(xyz_ij_list).shape == (3,): # in case xyz_ij_list is just a 3-vector, i.e., xyz_ij_list = [x, y, z]
            xyz_ij_list = [xyz_ij_list]
        if np.array(rel_weights).shape == (): # in case rel_weights is just a number
            rel_weights = [rel_weights]
        for i, xyz_ij in enumerate(xyz_ij_list):
            r_ij = xyz2r(xyz_ij)
            self.calc_exp_iota(r_ij)
            ci_ln = rel_weights[i] * self.exp_iota @ self.T #* 4*np.pi
            cutoff = cutoff_function(r_ij, self.rcut, self.dr)
            self.calc_grad_exp_iota(r_ij, xyz_ij)
            grad_ci_kln = rel_weights[i] * self.grad_exp_iota @ self.T
            grad_cutoff = grad_cutoff_function(xyz_ij, self.rcut, self.dr)
            for k in range(3):  # x, y, z
                grad_ci_kln[k] = grad_ci_kln[k] * cutoff + ci_ln * grad_cutoff[k]
            for l in range(self.lmax):
                for m in range(-l, l+1):
                    self.Yc[l,m+l] = SphericalHarmonicYCartesian(l, m, xyz_ij).conjugate()
                    self.grad_Yc[:,l,m+l] = gradSphericalHarmonicYCartesian(l, m, xyz_ij).conjugate()
            for n in range(self.nmax):
                for l in range(self.lmax):
                    for m in range(-l, l+1):
                        for k in range(3):
                            grad_c[k,n,l,m+l] += grad_ci_kln[k,l,n] * self.Yc[l,m+l] +\
                                                    ci_ln[l,n] * cutoff * self.grad_Yc[k,l,m+l]

        if pos == 'center':
            return -grad_c # ∂r_ij/∂r_i = -1, r_ij = r_j - r_i
        elif pos == 'neighbor':
            return grad_c   # ∂r_ik/∂r_k = +1
        else:
            raise ValueError("Wrong position configuration!")

    def overlap_integral_between_radial_basis(self, i, j):
        '''Calculate overlap integral between radial basis functions. [0, Inf)'''
        r_i = self.r_n[i]
        r_j = self.r_n[j]
        r_i_r_j = r_i + r_j
        a = self.alpha_atom
        return math.sqrt(2) * a**1.5 * r_i_r_j * math.exp(-a * (r_i**2 + r_j**2)) \
               + math.sqrt(math.pi) * a * (1 + a * r_i_r_j**2) \
                 * math.exp(-0.5 * a * (r_i - r_j)**2) \
                 * (1 + math.erf(math.sqrt(a/2) * r_i_r_j))

    def kernel(self, qi, qj):
        '''Kernel
        Rotationally invariant similarity measure
        between two local atomic environment i and j
        not yet normalized and can be a complex value
        '''
        #return np.sum(qi * qj)
        return dotprod(qi, qj)

    def covariance(self, qi, qj):
        '''Normalized SOAP kernel'''
        norm_qi = self.norm_q(qi)
        norm_qj = self.norm_q(qj)
        normalized_kernel = self.kernel(qi, qj) / (norm_qi*norm_qj)
        return np.abs(normalized_kernel)**self.zeta

    def q_grad_cov(self, qi, qj):
        '''Calculate the gradient of covariance w.r.t. "normalized qj" only (and not qi nor qj.conjugate())'''
        norm_qi = self.norm_q(qi)
        norm_qj = self.norm_q(qj)
        normalized_kernel = self.kernel(qi, qj) / (norm_qi*norm_qj)
        prefactor = 0.5 * self.zeta * self.covariance(qi, qj) / np.abs(normalized_kernel)**2
        return prefactor * normalized_kernel.conjugate() * qi / norm_qi

    def norm_q(self, qi):
        '''Evaluate the norm of qi not the one of the whole array qall'''
        return np.sqrt(np.abs(self.kernel(qi.conjugate(), qi)))
        # Compared to the previouse non-conjugated dot product, conjugated one leads to slightly better accuracy!

    def r_grad_q(self, ci, grad_ci):
        ''' Calculate the gradient of unnormalized qi w.r.t. r = [x, y, z]'''
        c = ci
        nmax, lmax, mmax = c.shape
        x_grad_c, y_grad_c, z_grad_c = grad_ci
        x_grad_q, y_grad_q, z_grad_q = np.zeros((3,nmax,nmax,lmax), dtype=complex_dtype)
        for n1 in range(nmax):
            for n2 in range(nmax):
                for l in range(lmax):
                    x_grad_q[n1,n2,l] = dotprod(x_grad_c[n1,l,:].conjugate(), c[n2,l,:]) + dotprod(c[n1,l,:].conjugate(), x_grad_c[n2,l,:])
                    y_grad_q[n1,n2,l] = dotprod(y_grad_c[n1,l,:].conjugate(), c[n2,l,:]) + dotprod(c[n1,l,:].conjugate(), y_grad_c[n2,l,:])
                    z_grad_q[n1,n2,l] = dotprod(z_grad_c[n1,l,:].conjugate(), c[n2,l,:]) + dotprod(c[n1,l,:].conjugate(), z_grad_c[n2,l,:])
        return np.stack([x_grad_q, y_grad_q, z_grad_q])

    def r_grad_norm_q(self, qi, grad_qi):
        ''' Calculate the gradient of norm of qi w.r.t. r = [x, y, z].
        grad_qi is the gradient of unnormalized qi w.r.t. r, which is computed by r_grad_q
        '''
        if grad_qi.shape != (3, self.nmax, self.nmax, self.lmax):
            raise TypeError(grad_qi.shape)

        norm_q = self.norm_q(qi)
        q_c = qi.conjugate()
        x_grad_q, y_grad_q, z_grad_q = grad_qi
        x_grad_norm_q = dotprod(q_c, x_grad_q).real / norm_q
        y_grad_norm_q = dotprod(q_c, y_grad_q).real / norm_q
        z_grad_norm_q = dotprod(q_c, z_grad_q).real / norm_q
        return np.stack([x_grad_norm_q, y_grad_norm_q, z_grad_norm_q])

    def r_grad_normalized_q(self, qi, grad_qi):
        ''' Calculate the gradient of normalized qi w.r.t r = [x, y, z].
        grad_qi is the gradient of unnormalized qi w.r.t r, which is computed by r_grad_q
        '''
        if grad_qi.shape != (3, self.nmax, self.nmax, self.lmax):
            raise TypeError(grad_qi.shape)

        norm_q = self.norm_q(qi)
        normalized_q = qi / norm_q
        x_grad_q, y_grad_q, z_grad_q = grad_qi
        x_grad_norm_q, y_grad_norm_q, z_grad_norm_q = self.r_grad_norm_q(qi, grad_qi)
        x_grad_normalized_q = - x_grad_norm_q * normalized_q + x_grad_q
        y_grad_normalized_q = - y_grad_norm_q * normalized_q + y_grad_q
        z_grad_normalized_q = - z_grad_norm_q * normalized_q + z_grad_q
        return np.stack([x_grad_normalized_q, y_grad_normalized_q, z_grad_normalized_q]) / norm_q

    def r_grad_cov(self, qi, qj, grad_qj):
        '''Calculate the gradient of covariance w.r.t. r = [x, y, z]'''
        if grad_qj.shape != (3, self.nmax, self.nmax, self.lmax):
            raise TypeError(grad_qi.shape)
        q_grad_cov = self.q_grad_cov(qi, qj)
        x_grad_normalized_q, y_grad_normalized_q, z_grad_normalized_q = self.r_grad_normalized_q(qj, grad_qj)
        x_grad_covariance = two_Re_dotprod(q_grad_cov, x_grad_normalized_q)
        y_grad_covariance = two_Re_dotprod(q_grad_cov, y_grad_normalized_q)
        z_grad_covariance = two_Re_dotprod(q_grad_cov, z_grad_normalized_q)
        return np.stack([x_grad_covariance, y_grad_covariance, z_grad_covariance])

    def r_grad_cov2(self, qi, qj, grad_qj):  # numerically checked: equivalent to r_grad_cov
        '''Calculate the gradient of covariance w.r.t. r = [x, y, z]
        in different way'''
        prefactor = self.zeta * self.covariance(qi, qj)
        norm_sq_qj = self.norm_q(qj)**2
        dotprod_c = self.kernel(qi, qj).conjugate()
        abs_dotprod_sq = np.abs(dotprod_c)**2
        qj_c = qj.conjugate()
        x_grad_qj, y_grad_qj, z_grad_qj = grad_qj
        x_grad_covariance = -dotprod(qj_c, x_grad_qj).real/norm_sq_qj + (dotprod_c*dotprod(qi, x_grad_qj)).real/abs_dotprod_sq
        y_grad_covariance = -dotprod(qj_c, y_grad_qj).real/norm_sq_qj + (dotprod_c*dotprod(qi, y_grad_qj)).real/abs_dotprod_sq
        z_grad_covariance = -dotprod(qj_c, z_grad_qj).real/norm_sq_qj + (dotprod_c*dotprod(qi, z_grad_qj)).real/abs_dotprod_sq
        return prefactor * np.stack([x_grad_covariance, y_grad_covariance, z_grad_covariance])

    def qq_hess_cov(self, qi, grad_qi, qj, grad_qj):
        '''Calculate the hessian of covariance w.r.t. "normalized" qi & qj only (and not qi nor qj.conjugate())'''
        pass

    def rr_hess_cov(self, qi, grad_qi, qj, grad_qj):
        '''Calculate the hessian of covariance w.r.t. ri & rj.'''
        pass

    def c2q(self, c):
        '''Calculate (unnormalized) descriptors from expansion coefficients'''
        if np.array(c).shape == (self.nmax, self.lmax, 2*self.lmax+1):  # for single input
            q = np.zeros((self.nmax, self.nmax, self.lmax), dtype=complex_dtype)
            for n1 in range(self.nmax):
                    for n2 in range(self.nmax):
                        for l in range(self.lmax):
                            q[n1, n2, l] = dotprod(c[n1, l, :].conjugate(), c[n2, l, :])
        else:   # for collective input # change it to recursively? 
            natoms = len(c)
            q = np.zeros((natoms, self.nmax, self.nmax, self.lmax), dtype=complex_dtype)
            for i in range(natoms):
                for n1 in range(self.nmax):
                    for n2 in range(self.nmax):
                        for l in range(self.lmax):
                            q[i, n1, n2, l] = dotprod(c[i, n1, l, :].conjugate(), c[i, n2, l, :])
        #q = q*(4*np.pi)**2
        return q

    def atomic_energies(self, q):
        '''For q_i's, find atomic energies, epsilon_i's'''
        # $\epsilon(\mathrm{\hat{q}}_i)$: atomic energy
        # $ \epsilon(\hat{\mathrm{q}})
        #  = \sum_j^M \alpha_j K(\hat{\mathrm{q}}_j, \hat{\mathrm{q}})
        #  = \mathrm{k}(\hat{\mathrm{q}})^{T} \mathbf{\alpha} $
        # k_IM = [
        #     [self.covariance(self.qall, q[i], q[j]) for i, _ in enumerate(self.qall)]
        #     for j, _ in enumerate(q)]

        num_of_atoms = len(q)
        num_of_rep_atoms = len(self.rep_atoms)
        self.k_IM = np.empty((num_of_atoms, num_of_rep_atoms))
        for i in range(num_of_atoms):
            for j, rep_atom in enumerate(self.rep_atoms):
                self.k_IM[i][j] = self.covariance(q[i], self.qall[rep_atom])
                #self.k_IM[i][j] = self.covariance(self.qall[rep_atom], q[i])   # test of symmetry of covariance: checked

        epsilons = self.k_IM @ self.alpha

        # print('k_IM = ', '\n', pd.DataFrame(self.k_IM), end='\n\n'); exit()
        if plot:
            plot_matrix(self.k_IM)
        # print('epsilons = ', epsilons)

        return epsilons

    def molecule_energy(self, xyz, weights):
        '''Evaluate GAP total energy'''
        # E: total energy
        # $ E = \sum_i^M \epsilon(\hat{\mathrm{q}}_i) $
        q = self.descriptors(xyz, weights)
        epsilons = self.atomic_energies(q)
        total_energy = np.sum(np.array(epsilons))
        return total_energy
    
    #@staticmethod
    def total_energies(self, data):
        '''Evaluate GAP total energies.
        Argument 'data' can be another dataset different from the train dataset'''
        if isinstance(data, dataset):
            xyzs = data.xyzs
        elif isinstance(data, ase.Atoms):
            xyzs = data

        ndata = len(xyzs)
        symbol_list = [xyz.get_chemical_symbols() for xyz in xyzs]
        # list of computed energies in data
        energies = np.zeros((ndata), dtype=float_dtype)

        for i, xyz in enumerate(xyzs):
            atom_test_weights = np.array([self.symbol_to_weight(symbol) for symbol in symbol_list[i]])
            energies[i] = self.molecule_energy(xyz, atom_test_weights)

        energies += data.y0

        return energies

    def R_grad_COV_xyz(self, qext, xyz, weights):
        '''Evaluate in a molecule xyz 
        [r1_grad_cov, ..., rl_grad_cov, ..., rk_grad_cov]
        where rl_grad_cov = [dcov/dx_l, dcov/dy_l, dcov/dz_l]
        and cov = cov(q, q1) + ... + cov(q, qk) 
        '''
        n_e = len(xyz)
        n_pos = n_e
        center_id_list, neighbor_id_list, rel_xyz_list = ase.neighborlist.neighbor_list('ijD', xyz, self.rcut)
        grad_COV = np.zeros((n_pos, 3), dtype=float_dtype) 
        c = self.coefficients(xyz, weights)
        q = self.c2q(c)
        for i in range(n_pos):    # r_i
            for j in range(n_e):    # e_j
                atom_j_idx = list(np.where(center_id_list == j)[0])
                if i == j:
                    # gradient of e_i w.r.t. r_i
                    xyz_jl_list = rel_xyz_list[atom_j_idx] # center = j, neighbor = l
                    atom_l_id_list = neighbor_id_list[atom_j_idx]
                    rel_weight = 2 * weights[atom_l_id_list] / (weights[atom_l_id_list] + weights[j])
                    grad_cj = self.grad_ci(xyz_jl_list, rel_weight, pos='center')
                else:   # i != j
                    # gradient of e_j w.r.t. r_i when j!=i
                    atom_l_id_list = neighbor_id_list[atom_j_idx]
                    atom_ji_idx = list(np.where(atom_l_id_list == i)[0])
                    if atom_ji_idx == []:
                        #print(f'The atom {i} is not a neighbor to {j}')
                        #grad_cj = np.zeros((3, *c[j].shape))
                        grad_qj = np.zeros((3, *q[j].shape))
                    else:
                        xyz_ji = rel_xyz_list[atom_ji_idx] # center = j, neighbor = (i only)
                        rel_weight = 2 * weights[i] / (weights[i] + weights[j])
                        grad_cj = self.grad_ci(xyz_ji, rel_weight, pos='neighbor')
                        grad_qj = self.r_grad_q(c[j], grad_cj)
                grad_COV[i] += self.r_grad_cov2(qext, q[j], grad_qj)
        return grad_COV

    def R_grad_Q(self, xyz, c, q, weights):
        '''Evaluate in a molecule xyz
        [[r1_grad_q1, r1_grad_q2, ...], [ri_grad_q1,..., ri_grad_qj, ...], ... ]
        where ri_grad_qj = [dqj/dx_i, dqj/dy_i, dqj/dz_i]
        '''
        n_e = len(xyz)
        n_pos = n_e
        if q.shape[1:] == (self.nmax, self.nmax, self.lmax):
            single_q_shape = q.shape[1:]
        elif q.shape == (self.nmax, self.nmax, self.lmax):
            single_q_shape = q.shape
        else:
            raise TypeError(q.shape)

        center_id_list, neighbor_id_list, rel_xyz_list = ase.neighborlist.neighbor_list('ijD', xyz, self.rcut)
        grad_Q = np.zeros((n_pos, n_e, 3, *single_q_shape), dtype=complex_dtype) 
        for i in range(n_pos):    # r_i
            for j in range(n_e):    # e_j
                atom_j_idx = list(np.where(center_id_list == j)[0])
                if i == j:
                    # gradient of e_i w.r.t. r_i
                    xyz_jl_list = rel_xyz_list[atom_j_idx] # center = j, neighbor = l
                    atom_l_id_list = neighbor_id_list[atom_j_idx]
                    rel_weight = 2 * weights[atom_l_id_list] / (weights[atom_l_id_list] + weights[j])
                    grad_cj = self.grad_ci(xyz_jl_list, rel_weight, pos='center')
                else:
                    # gradient of e_j w.r.t. r_i when j!=i
                    atom_l_id_list = neighbor_id_list[atom_j_idx]
                    atom_ji_idx = list(np.where(atom_l_id_list == i)[0])
                    if atom_ji_idx == []:
                        #print(f'The atom {i} is not a neighbor to {j}')
                        grad_cj = np.zeros((3, *c[j].shape))
                    else:
                        xyz_ji = rel_xyz_list[atom_ji_idx] # center = j, neighbor = (i only)
                        rel_weight = 2 * weights[i] / (weights[i] + weights[j])
                        grad_cj = self.grad_ci(xyz_ji, rel_weight, pos='neighbor')
                grad_Q[i, j] = self.r_grad_q(c[j], grad_cj)
        return grad_Q
    
    def R_grad_COV_q(self, qext, q, gradq):
        '''Evaluate in a molecule 
        [r1_grad_cov, ..., rl_grad_cov, ..., rk_grad_cov]
        where rl_grad_cov = [dcov/dx_l, dcov/dy_l, dcov/dz_l]
        and cov = cov(qext, q1) + ... + cov(qext, qk) 
        '''
        if q.shape[1:] == (self.nmax, self.nmax, self.lmax):
            n_e = q.shape[0]
        elif q.shape == (self.nmax, self.nmax, self.lmax):
            n_e = 1
            q = [q]; gradq = [gradq]
        else:
            raise TypeError(q.shape)
        n_pos = n_e
        grad_COV = np.zeros((n_pos, 3), dtype=float_dtype) 
        for i in range(n_pos):    # r_i
            for j in range(n_e):    # e_j
                grad_COV[i] += self.r_grad_cov2(qext, q[j], gradq[i,j])
        return grad_COV
    
    def atomic_forces_from_e(self, xyz, weights):
        '''For q_i, find atomic energies, epsilon'''
        n_e = len(xyz)
        n_pos = n_e
        n_rep_atoms = len(self.rep_atoms)
        center_id_list, neighbor_id_list, rel_xyz_list = ase.neighborlist.neighbor_list('ijD', xyz, self.rcut)
        grad_k_IM = np.empty((3*n_pos*n_e, n_rep_atoms), dtype=float_dtype)
        c = self.coefficients(xyz, weights)
        q = self.c2q(c)
        for i in range(n_pos):    # r_i
            for j in range(n_e):    # e_j
                atom_j_idx = list(np.where(center_id_list == j)[0])
                if i == j:
                    # gradient of e_i w.r.t. r_i
                    xyz_jl_list = rel_xyz_list[atom_j_idx] # center = j, neighbor = l
                    atom_l_id_list = neighbor_id_list[atom_j_idx]
                    rel_weight = 2 * weights[atom_l_id_list] / (weights[atom_l_id_list] + weights[j])
                    grad_cj = self.grad_ci(xyz_jl_list, rel_weight, pos='center')
                else:
                    # gradient of e_j w.r.t. r_i when j!=i
                    atom_l_id_list = neighbor_id_list[atom_j_idx]
                    atom_ji_idx = list(np.where(atom_l_id_list == i)[0])
                    xyz_ji = rel_xyz_list[atom_ji_idx] # center = j, neighbor = i only
                    rel_weight = 2 * weights[i] / (weights[i] + weights[j])
                    grad_cj = self.grad_ci(xyz_ji, rel_weight, pos='neighbor')
                grad_qj = self.r_grad_q(c[j], grad_cj)
                idx = 3*(i*n_e + j)   # starting row index
                #print(idx, end=',')
                for k, rep_atom in enumerate(self.rep_atoms):
                    grad_k_IM[idx:idx+3, k] = self.r_grad_cov2(self.qall[rep_atom], q[j], grad_qj)   # beware of the order of the arguments
                    
        forces = - grad_k_IM @ self.alpha   # the negative sign '-' must be included
        #print('grad_k_IM = ', '\n', pd.DataFrame(grad_k_IM), end='\n\n'); exit()
        #print('forces = ', '\n', pd.DataFrame(forces.reshape((n_pos*n_e, 3))), end='\n\n'); exit()
        return forces

    def molecule_forces_from_e(self, xyz, weights):
        '''Evaluate GAP molecular forces'''
        # F_i : force felt by atom i in a molecule
        # $ \bold{F}_i = \sum_k \del_{\bold{r}_i} \epsilon(\hat{\mathrm{q}}_k) $
        n_atoms = len(xyz)
        n_e = n_atoms
        forces = self.atomic_forces_from_e(xyz, weights)
        molecule_forces = np.empty((n_atoms, 3), dtype=float_dtype)
        for i in range(n_atoms):    # r_il (i=1,...,n_atoms, l=x,y,z)
        #    for j in range(n_e):    # e_j
        #        for l in range(3):  # x,y,z
        #            molecule_forces[i,l] += forces[3*(n_e*i + j) + l]
            molecule_forces[i] = [np.sum(forces[3*n_e*i+l: 3*n_e*(i+1)+l: 3]) for l in range(3)]
        return molecule_forces

    def molecule_forces_from_e2(self, xyz, weights):
        n_atoms = len(xyz)
        n_rep_atoms = len(self.rep_atoms)
        c_mol = self.coefficients(xyz, weights)
        q_mol = self.c2q(c_mol)
        gradq_mol = self.R_grad_Q(xyz, c_mol, q_mol, weights)
        grad_K_IM = np.empty((n_atoms*3, n_rep_atoms), dtype=float_dtype)
        for k, rep_atom in enumerate(self.rep_atoms):
            grad_K_IM[:, k] = self.R_grad_COV_q(self.qall[rep_atom], q_mol, gradq_mol).flat
        return - grad_K_IM @ self.alpha # the negative sign '-' must be included

    def total_forces_from_e(self, data):
        '''Evaluate GAP total molecular forces'''
        if isinstance(data, dataset):
            xyzs = data.xyzs
        elif isinstance(data, ase.Atoms):
            xyzs = data
        
        ndata = len(xyzs)
        natoms_list = [len(xyz) for xyz in xyzs]
        natoms_all = sum(natoms_list)
        symbol_list = [xyz.get_chemical_symbols() for xyz in xyzs]
        # list of computed forces in data
        #forces = np.zeros((3*natoms_all), dtype=float_dtype)   # flat version
        forces = np.empty((0,3), dtype=float_dtype)

        start_idx = 0
        for i, xyz in enumerate(xyzs):
            atom_test_weights = np.array([self.symbol_to_weight(symbol) for symbol in symbol_list[i]])
            natoms = natoms_list[i]
            end_idx = start_idx + 3*natoms
            #atom_forces = self.molecule_forces_from_e(xyz, atom_test_weights)
            atom_forces = self.molecule_forces_from_e2(xyz, atom_test_weights)
            print(atom_forces.reshape((natoms, 3)), end='\n')
            #forces[start_idx:end_idx] = atom_forces.flat   # flat array version
            forces = np.vstack((forces, atom_forces.reshape((natoms, 3))))
            start_idx = end_idx

        return forces

    def descriptors(self, xyz, weights):
        '''Calculate the (unnormalized) descriptors (q_i) for all atoms (i) in a molecule (xyz)'''

        if timing:
            time_start = timer()

        if profiler:
            import cProfile
            import pstats
            import io
            pr = cProfile.Profile()
            pr.enable()

        c = self.coefficients(xyz, weights)
        q = self.c2q(c)

        if profiler:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            sortby = 'tottime'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        if timing:
            time_end = timer()
            time_elapsed = time_end - time_start
            nint = self.nmax * self.lmax * \
                (2 * self.lmax+1) * self.natoms_max  # not exact
            print(
                '{:.2f} seconds for {} integrations,'.format(time_elapsed, nint),
                '{:.2f} milli-seconds per integration'.format(1000*time_elapsed/nint))

        return q

    def calc_exp_iota(self, r_ij):
        '''Fill self.exp_iota[0:lmax-1, 0:nmax-1] at (r_n[0:nmax-1], r_ij)'''

        # n = 0
        self.exp_iota[:, 0] = 0
        self.exp_iota[0, 0] = math.exp(-self.alpha_atom * r_ij**2)

        # n = 1, ..., (nmax-1)
        for n in range(1, self.nmax):
            r_n = self.r_n[n]
            z = 2 * self.alpha_atom * r_ij * r_n
            e_plus = math.exp(-self.alpha_atom * (r_ij + r_n)**2)
            e_minus = math.exp(-self.alpha_atom * (r_ij - r_n)**2)

            # l = -1
            f_l_1 = 0.5 * (e_minus + e_plus) / z
            # l = 0
            f_l_0 = 0.5 * (e_minus - e_plus) / z
            self.exp_iota[0, n] = f_l_0

            # l = 1, ..., (lmax-1)
            for l in range(1, self.lmax):
                f_l_2 = f_l_1
                f_l_1 = f_l_0
                f_l_0 = f_l_2 - (2*l-1) * f_l_1 / z
                self.exp_iota[l, n] = f_l_0

    def calc_grad_exp_iota(self, r_ij, xyz_ij): # numerically checked
        '''Fill self.grad_exp_iota[0:3, 0:lmax-1, 0:nmax-1] at (r_n[0:nmax-1], r_ij)'''
        #r_ij = np.sqrt(np.sum(xyz_ij**2))
        #r_ij = xyz2r(xyz_ij)
        # n = 0
        self.grad_exp_iota[:, :, 0] = 0
        self.grad_exp_iota[:, 0, 0] = -2 * self.alpha_atom * xyz_ij * self.exp_iota[0, 0]

        # n = 1, ..., (nmax-1)
        for n in range(1, self.nmax):
            r_n = self.r_n[n]
            z = 2 * self.alpha_atom * r_ij * r_n
            #e_plus = math.exp(-self.alpha_atom * (r_ij + r_n)**2)
            #e_minus = math.exp(-self.alpha_atom * (r_ij - r_n)**2)

            # l = -1
            #f_l_m1 = 0.5 * (e_minus + e_plus) / z
            # l = 0
            f_l = self.exp_iota[0, n]
            # l = 1
            f_l_p1 = self.exp_iota[1, n]
            # l = 0
            self.grad_exp_iota[:, 0, n] = 2 * self.alpha_atom * xyz_ij * ( -f_l +
                    (r_n/r_ij) * f_l_p1  )

            # l = lmax
            f_lmax = self.exp_iota[self.lmax-2, n] - (2*self.lmax-1) * self.exp_iota[self.lmax-1, n] / z
            # l = 1, ..., (lmax-1)
            for l in range(1, self.lmax):
                f_l_m1 = self.exp_iota[l-1, n]
                f_l = self.exp_iota[l, n]
                #f_l_p1 = np.heaviside((lmax-1)-l,f_lmax) * self.exp_iota[l+1, n]
                if l == self.lmax-1:
                    f_l_p1 = f_lmax
                else:
                    f_l_p1 = self.exp_iota[l+1, n]
                self.grad_exp_iota[:, l, n] = 2 * self.alpha_atom * xyz_ij * ( -f_l +
                    (r_n/r_ij) * ( l*f_l_m1 + (l+1)*f_l_p1 ) / (2*l+1) )

    
