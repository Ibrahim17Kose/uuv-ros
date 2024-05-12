import math
import numpy as np


def smtrx(a):
    S = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return S

def Rzyx(phi, theta, psi):
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = [[cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth],
         [spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi],
         [-sth, cth*sphi, cth*cphi]]
    return R

def Tzyx(phi, theta):
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)

    if cth == 0:
        raise("Tzyx is singular for theta = +-90 degrees")

    T = [[1.0, sphi*sth/cth, cphi*sth/cth],
         [0.0, cphi, -sphi],
         [0.0, sphi/cth, cphi/cth]]
    return T

def eulerang(phi, theta, psi):
    J1 = Rzyx(phi,theta,psi)
    J2 = Tzyx(phi,theta)
    
    J = np.concatenate((np.concatenate((J1, np.zeros((3, 3))), axis=1),
                        np.concatenate((np.zeros((3, 3)), J2), axis=1)), axis=0)
    return J, J1, J2

def body2ned(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    J_eta, _, _ = eulerang(phi, theta, psi)
    return J_eta

def ned2body(eta):
    phi = eta[3]
    theta = eta[4]
    psi = eta[5]

    J_eta, _, _ = eulerang(phi, theta, psi)
    return J_eta.T

def m2c(M, nu):
    if not isinstance(M, np.ndarray):
        M = np.array(M)
    M = 0.5 * (M + M.T)  # Symmetrization of the inertia matrix

    M11 = M[0:3,0:3]
    M12 = M[0:3,3:6]
    M21 = M12.T
    M22 = M[3:6,3:6]

    nu1 = nu[0:3]
    nu2 = nu[3:6]
    nu1_dot = np.matmul(M11, nu1) + np.matmul(M12, nu2)
    nu2_dot = np.matmul(M21, nu1) + np.matmul(M22, nu2)

    C = np.concatenate((np.concatenate((np.zeros((3,3)), -smtrx(nu1_dot)), axis=1),
                        np.concatenate((-smtrx(nu1_dot), -smtrx(nu2_dot)), axis=1)), axis=0)
    return C

def gvect(W, B, theta, phi, r_bg, r_bb):
    sth  = math.sin(theta)
    cth  = math.cos(theta)
    sphi = math.sin(phi)
    cphi = math.cos(phi)

    g = [(W-B) * sth,
         -(W-B) * cth * sphi,
         -(W-B) * cth * cphi,
         -(r_bg[1]*W-r_bb[1]*B) * cth * cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
         (r_bg[2]*W-r_bb[2]*B) * sth + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
         -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth]
    return np.array(g)

def ned2enu(ned):
    enu = np.zeros_like(ned)
    enu[0] = ned[0]
    enu[1] = - ned[1]
    enu[2] = - ned[2]
    enu[3] = ned[3]
    enu[4] = - ned[4]
    enu[5] = - ned[5]
    return enu

def enu2ned(enu):
    ned = np.zeros_like(enu)
    ned[0] = enu[0]
    ned[1] = - enu[1]
    ned[2] = - enu[2]
    ned[3] = enu[3]
    ned[4] = - enu[4]
    ned[5] = - enu[5]
    return ned
