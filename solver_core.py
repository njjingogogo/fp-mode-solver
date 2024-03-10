from scipy.special import hermite
from scipy.optimize import minimize

from tqdm import tqdm

# This import all is a legacy issue when this code was first written
# And this is the reason for packaging this 'solver_core' and not merging it into the main class
# Will try to fix this in a later patch...
from numpy import *

def HG_basis(X, Y, xy_res, k, w0, z, m, n, sim_window):
    """Hermite-Gaussian functions as basis functions of the mode solver
    """
    z0 = k*w0**2/2
    qz = z+1j*z0
    wz = sqrt(-2/k/imag(1/qz))
    wave_fun = hermite(m)(sqrt(2)*X/wz) * hermite(n)(sqrt(2)*Y/wz) /qz*exp(-1j*k*(X**2+Y**2)/(2*qz)+1j*(m+n)*arctan(z/z0))
    total_area = xy_res**2 * sum(abs(wave_fun)**2 * sim_window)
    return wave_fun/sqrt(total_area)

def waist_search1(X, Y, xy_res, Z, k, L, w0_guess, w0_range, sim_window):
    """Beam waist optimization for plano_concave cavities
    """
    # Objective function that returns mode mismatch
    def Opt_M00(w0):
        B00 = xy_res**2 * sum(HG_basis(X,Y,xy_res,k,w0,L,0,0,sim_window) 
                              * HG_basis(X,Y,xy_res,k,w0,L,0,0,sim_window) * exp(+2j*k*Z) * sim_window)
        D00 = 1-abs(B00)**2
        return D00
    w0_init = w0_guess
    res = minimize(Opt_M00, w0_init, method='L-BFGS-B',bounds=((w0_init*(1-w0_range), w0_init*(1+w0_range)),))
    w0 = res.x[0]
    return w0

def beamp(u, phi, sim_window):
    """Beam propagation in free space
    """
    u_fft = fft.ifft2(fft.ifftshift(u), norm="ortho")
    u_fft_1 = u_fft*phi
    u_fft_prop = fft.fftshift(fft.fft2(u_fft_1, norm="ortho"))*sim_window
    return u_fft_prop

def scatter_eigen(X, Y, xy_res, mirror1_Z, mirror2_Z, k, L, w0, mode_order_combined, sim_window):
    """Generate scattering matrix and solve the eigen problem of it
    
    Scattering matrix is computed by propagating each Hermite-Guassian mode
    for a full round trip and calculating its coeeficient along the basis 
    """
    # Mode indices for the scattering matrix
    m_list = []
    n_list = []
    for n in range(mode_order_combined+1):
        n_temp = list(arange(n+1))
        n_list = n_list+n_temp
        m_temp = list(n-arange(n+1))
        m_list = m_list+m_temp

    n_list = tuple(n_list)
    m_list = tuple(m_list)
    mode_num = len(n_list)

    # Generate the scattering matrix
    # beam amplitude at the beginning of propagation
    HerFun = zeros((mode_num,size(mirror1_Z)))+1j*zeros((mode_num,size(mirror1_Z))) 
    # beam amplitude at the end of propagagtion
    HerFun_refl = zeros((mode_num,size(mirror1_Z)))+1j*zeros((mode_num,size(mirror1_Z)))  
    
    # mirror phase delay
    mirror1_phi = exp(+2j*k*mirror1_Z)
    mirror2_phi = exp(+2j*k*mirror2_Z)
    # propagation phase
    k_y = 2*pi*fft.fftfreq(mirror1_Z.shape[0], d=xy_res)
    k_x = 2*pi*fft.fftfreq(mirror1_Z.shape[1], d=xy_res)
    K_Y, K_X = meshgrid(k_y, k_x, indexing='ij')
    K_Z = -1j*sqrt(K_X**2+K_Y**2-k**2+0j)
    # Propagation phase
    prop_phi = exp(-1j*K_Z*L)

    print('Constructing scattering matrix:')
    for s in tqdm(range(mode_num)):
        u0temp = HG_basis(X,Y,xy_res,k,w0,0,n_list[s],m_list[s], sim_window)
        HerFun[s,:] = ravel(u0temp)
        HerFun_refl[s,:]= ravel(beamp(beamp(u0temp, prop_phi, sim_window)*mirror1_phi, prop_phi, sim_window)*mirror2_phi)

    M = xy_res**2 * HerFun_refl @ conj(HerFun).T
    
    # Solve the eigen problem of the scattering matrix
    print('Solving eigen problem')
    eigen_value, eigen_vector = linalg.eig(M)
    loss = 1-abs(eigen_value)**2
    mode = (eigen_vector.T @ HerFun).reshape(mode_num,mirror1_Z.shape[0],mirror1_Z.shape[1])

    return mode_num, eigen_value, eigen_vector, loss, mode