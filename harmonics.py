
import numpy as np
from math import floor, sqrt, factorial
from scipy.special import sph_harm
from sympy.physics.quantum.cg import CG, Wigner3j
from util import eval_rel_error

# For numerical evaluation of the derivatives of spherical harmonics
dx_for_derivative = 1E-6 # If taken too low, it could degrade the precision of derivative evaluations.

#-----------------------------------------------------------------------------------
# Solid and spherical harmonics
# $ R_l^m = \sqrt{\frac{4 \pi}{2 l + 1}} r^l Y_l^m (\theta, \phi)
#         = \sqrt{\frac{(l-m)!}{(l+m)!}} r^l P_l^m (cos \theta) e^{i m \phi} $
#
# Cartesian reprentation
# $ R_l^m = \sqrt{(l+m)!(l+m)!} \sum_{q=0}^{\floor{(l-m)/2}} *
#           \frac{1}{p!q!r!} (-(x+iy)/2)^p ((x-iy)/2)^q z^r $
#-----------------------------------------------------------------------------------

def check_arg_Harmonics(l,m):
    if not (isinstance(l,int) and isinstance(m,int) and l >= 0 and -l <= m and m <=l):
        raise ValueError( "Check if the values of l={} and m={} are proper ones for harmonic functions.".format(l,m) )

def SolidHarmonicRCartesian(l, m, xyz):
    check_arg_Harmonics(l,m)
    if m < 0:
        m = -m
        return (-1)**m * np.conj(SolidHarmonicRCartesian(l, m, xyz)) 
    x, y, z = xyz
    R = 0
    q_max = int((l-m)/2) #int(floor((l-m)/2))
    
    for q in range(q_max+1):
        p = m + q
        r = l - p - q
        R = R + (-0.5*(x+1j*y))**p * (0.5*(x-1j*y))**q * z**r / ( factorial(p)*factorial(q)*factorial(r) )
    return sqrt( factorial(l+m) * factorial(l-m) ) * R

def SphericalHarmonicYCartesian(l, m, xyz):
    if m < 0:
        m = -m
        return (-1)**m * np.conj(SphericalHarmonicYCartesian(l, m, xyz)) 
#   Perhaps due to the error propagation induced by the exponent evaluation,
#   precision fluctuates if using below
#    r_sq = sum(np.array(xyz)**2)
#    return sqrt( (2*l+1)/(4*np.pi) ) * r_sq**(-0.5*l) * SolidRCartesian(l, m, xyz)
#   Instead use below; precision gets (though very slightly) better 
    r = np.linalg.norm(xyz)
    return sqrt( (2*l+1)/(4*np.pi) ) * r**(-l) * SolidHarmonicRCartesian(l, m, xyz)

def gradSolidHarmonicRCartesian(l, m, xyz):
    check_arg_Harmonics(l,m)
    if m < 0:
        m = -m
        return (-1)**m * np.conj(gradSolidHarmonicRCartesian(l, m, xyz)) 

    x_grad, y_grad, z_grad = 0, 0, 0 
    x, y, z = xyz
    q_max = int((l-m)/2) #int(floor((l-m)/2))
    
    for q in range(q_max+1):
        p = m + q
        r = l - p - q
        if p >= 1: # derivative w.r.t. (x+1j*y)
            x_grad = x_grad - 0.5 * (-0.5*(x+1j*y))**(p-1) * (0.5*(x-1j*y))**q * z**r / (factorial(p-1)*factorial(q)*factorial(r))
            y_grad = y_grad - 0.5j * (-0.5*(x+1j*y))**(p-1) * (0.5*(x-1j*y))**q * z**r / (factorial(p-1)*factorial(q)*factorial(r))
        if q >= 1: # derivative w.r.t. (x-1j*y)
            x_grad = x_grad + 0.5 * (-0.5*(x+1j*y))**p * (0.5*(x-1j*y))**(q-1) * z**r / (factorial(p)*factorial(q-1)*factorial(r))
            y_grad = y_grad - 0.5j * (-0.5*(x+1j*y))**p * (0.5*(x-1j*y))**(q-1) * z**r / (factorial(p)*factorial(q-1)*factorial(r))
        if r >= 1: # derivative w.r.t. z
            z_grad = z_grad + (-0.5*(x+1j*y))**p * (0.5*(x-1j*y))**q * z**(r-1) / (factorial(p)*factorial(q)*factorial(r-1))
    
    return sqrt( factorial(l+m) * factorial(l-m) ) * np.array([x_grad, y_grad, z_grad])


def gradSphericalHarmonicYCartesian(l, m, xyz):
    if m < 0:
        m = -m
        return (-1)**m * np.conj(gradSphericalHarmonicYCartesian(l, m, xyz)) 

    r = np.linalg.norm(xyz)
    gradient = r**(-l) * gradSolidHarmonicRCartesian(l, m, xyz)
    gradient = gradient - l * np.array(xyz) * r**(-l-2)  * SolidHarmonicRCartesian(l, m, xyz) 
    return sqrt( (2*l+1)/(4*np.pi) ) * gradient
    
def grad_sph_harm(m,l,phi,theta):
    check_arg_Harmonics(l,m)
    # Note that the arguments of sph_harm are (by definition) reversed from perspective of QM convention.
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    r = 1
    rho = r*sin_theta
    x, y, z = sin_theta*cos_phi, sin_theta*sin_phi, cos_theta
    dx = dy = dz = dx_for_derivative
#    dr = (x*dx + y*dy + z*dz) / r
#    dtheta = (cos_theta*dr - dz)/rho
    drho = (x*dx + y*dy) / rho
    dtheta = (z*drho - rho*dz) / r**2
    dphi = (x*dy - y*dx) / rho**2
    # r-component of the gradient; the rest (theta_grad, phi_grad) are defined similarly
    r_grad = 0
    theta_grad = ( sph_harm(m,l,phi,theta+dtheta) - sph_harm(m,l,phi,theta-dtheta) ) / (2*dtheta)
    phi_grad = ( sph_harm(m,l,phi+dphi,theta) - sph_harm(m,l,phi-dphi,theta) ) / (2*dphi)
    # the unit basis vectors
    hat_r = np.array([x, y, z]) / r
    hat_theta = np.array([cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta])
    hat_phi = np.array([-sin_phi, cos_phi, 0])
    return hat_r*r_grad + hat_theta*theta_grad/r + hat_phi*phi_grad/rho

# To be later used; currently the built-in function from sympy.physics.quantum.cg will be used
#-----------------------------------------------------------------------------------
# Clebsch-Gordan coefficient 
# $ \left< j_1 m_1 j_2 m_2 | j m \right> = (-1)^{m + j_1 - j_2) * \sqrt{2j+1}
# \left( \begin{array}{ccc}
# j_1 & j_2 & j  \\
# m_1 & m_2 & -m \\
# \end{array} \right) $
# where matrix-like term on the right-hand side is the Wigner 3j-symbol.
#-----------------------------------------------------------------------------------    
      
def ClebschGordan(j1, m1, j2, m2, j3, m3):
    if not ( j1>=0 and j2>=0 and j3>=0 and abs(m1)<=j1 and abs(m2)<=j2 and abs(m3)<=j3 and 
             m1+m2==m3 and abs(j1-j2)<=j3 and j3<=j1+j2 ):
        raise ValueError( "Check the inconsistency of the arguments for Clebsch-Gordan coefficient  \
        <j1={},m1={},j2={},m2={}|j={},m={}>".format(j1, m1, j2, m2, j3, m3) )
    else:
        return (-1)**(m3 + j1 - j2) * sqrt(2*j3 + 1) * Wigner3J(j1,m1,j2,m2,j3,-m3)

#-----------------------------------------------------------------------------------
# Triangle coefficient
# $ \Delta(a,b,c) = \frac{ (a+b-c)! (a-b+c)! (-a+b+c)! }{ (a+b+c+1)! } $
#-----------------------------------------------------------------------------------

def TriangleCoeff(a,b,c):
    return factorial(a+b-c) * factorial(a-b+c) * factorial(-a+b+c) / factorial(a+b+c+1)

#-----------------------------------------------------------------------------------
# Wigner 3j symbol using Racah formula
# $ \left( \begin{array}{ccc}
# j_1 & j_2 & j \\
# m_1 & m_2 & m \\
# end{array} \right) = (-1)^{j_1 - j_2 - m) \sqrt{ \Delta (j_1,j_2,j) }
# \sqrt{ (j_1+m_1)! (j_1-m_1)! (j_2+m_2)! (j_2-m_2)! (j+m)! (j-m)! }
# \sum_k \frac{ (-1)^k }{k! (j-j_2+k+m_1)! (j-j_1+k-m_2)! (j_1+j_2-j-k)!
# (j_1-k-m_1)! (j_2-k+m_2)! } $
#-----------------------------------------------------------------------------------

def Wigner3J(j1, m1, j2, m2, j3, m3):
    prefactor = (-1)**(j1-j2-m3) * sqrt( TriangleCoeff(j1,j2,j3) * 
        factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * 
        factorial(j2-m2) * factorial(j3+m3) * factorial(j3-m3) )
    summation = 0
    # i runs over all integers that have non-negative values in the arguments of 
    # the factorials in the denomiator hence its min & max values are set by
    i_min = max( j2-j3-m1, j1+m2-j3, 0 ) 
    i_max = min( j1+j2-j3, j1-m1, j2+m2)
    for i in range(i_min, i_max+1):
        denom = factorial(i) * factorial(j3-j2+m1+i) * \
                factorial(j3-j1-m2+i) * factorial(j1+j2-j3-i) * \
                factorial(j1-m1-i) * factorial(j2+m2-i)
        summation = summation + (-1)**i / denom
    return prefactor * summation


#-----------------------------------------------------------------------------------
# Wigner 6j symbol to be later defined
#-----------------------------------------------------------------------------------
def Wigner6J(j1, m1, j2, m2, j3, m3):
    pass





#   Test
if __name__ == '__main__':
    l = np.random.randint(0,10)
    m = np.random.randint(-l,l+1)
    theta = np.pi * np.random.random()
    phi = 2 * np.pi * np.random.random()
    xyz = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    Y_scipy, Y_this = sph_harm(m,l,phi,theta), SphericalHarmonicYCartesian(l,m,xyz)
    grad_Y_num, grad_Y_anal = grad_sph_harm(m,l,phi,theta), gradSphericalHarmonicYCartesian(l,m,xyz)
    print("Comparison between the values of spherical harmonics from scipy and this code for \
    l={}, m={}, theta={:.4f}, phi={:.4f}".format(l,m,theta,phi) )
#    print(Y_scipy, Y_this)
    print("relative error: {:.4E}".format(eval_rel_error(Y_this,Y_scipy)))
    print("Comparison of the gradients from numeric and analytic methods:")
#    print(grad_Y_num, grad_Y_anal)
    print("relative error: {:.4E}".format(eval_rel_error(grad_Y_num,grad_Y_anal)))
