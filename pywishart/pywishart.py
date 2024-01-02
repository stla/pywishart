# -*- coding: utf-8 -*-
from math import sqrt, floor
import numpy as np
from scipy.stats import chi, norm, ncx2
from scipy.linalg import ldl

__sepsilon = sqrt(np.finfo(float).eps)


def __matrix_root(S):
    vals, vecs = np.linalg.eigh(S)
    roots = np.empty(len(vals))
    for i, val in enumerate(vals):
        if val < 0:
            if val > -__sepsilon:
                roots[i] = 0
            else:
                return None
        else:
            roots[i] = sqrt(val)
    return np.matmul(vecs, np.transpose(roots * vecs))


def __triangular_gauss(d):
    Z = np.zeros((d, d), dtype=float)
    for j in range(d-1):
        Z[j+1:, j] = norm.rvs(size = d-j-1)
    return Z


def __rwishart_cholesky(n, nu, p):
    D = np.zeros((p, n), dtype=float)
    for k in range(p):
        D[k, :] = chi.rvs(df = nu - k, size = n)
    out = [None]*n
    for sim in range(n):
        Z = __triangular_gauss(p)
        np.fill_diagonal(Z, D[:, sim])
        out[sim] = Z
    return out


def __extended_cholesky(S):
    d = S.shape[1]
    r = np.linalg.matrix_rank(S)
    LU, D, perm = ldl(S, lower=True)
    H = np.matmul(LU[perm, :], np.sqrt(D))
    L = H[0:r, 0:r]
    M = H[r:, 0:r]
    Ctilde = np.hstack(
        (
            np.vstack((L, M)), 
            np.vstack((np.zeros((r, d-r)), np.eye(d-r)))
        )
    )
    return (L, Ctilde, perm)


def __rwishart_aa_m(n, nu, m, Theta):
    d = Theta.shape[0]
    L, Ctilde, perm = __extended_cholesky(Theta[1:, 1:])
    invperm = np.arange(len(perm))[np.argsort(perm)]
    r = L.shape[0]
    xtilde = Theta[np.vstack(([0], (perm+1).reshape((d-1,1)))), 0]
    u1 = np.matmul(np.linalg.inv(L), xtilde[1:(r+1)])
    U11 = ncx2.rvs(df = nu-r, nc = max(0, xtilde[0] - np.vdot(u1, u1)), size = n)
    Z = norm.rvs(size = (r, n))
    U = Z + np.repeat(u1, n, 1)
    B1 = np.vstack(([1.0], np.zeros((d-1, 1))))
    B = np.hstack((B1, np.vstack((np.zeros(d-1), Ctilde[invperm, :]))))
    Wsims = [None]*n
    Y = np.zeros((d, d))
    Dr = np.eye(r)
    tB = np.transpose(B)
    for i in range(n):
        Ui = U[:, i]
        Y[:(r+1), :(r+1)] = np.hstack(
            (
                np.vstack(([U11[i] + np.vdot(Ui, Ui)], Ui.reshape(r, 1))),
                np.vstack((Ui, Dr))
            )
        )
        Wsims[i] = np.matmul(B, np.matmul(Y, tB))
    for k in range(1, m):
        idcs = np.arange(d)
        idcs[k] = 0
        idcs[0] = k
        for i in range(n):
            Wi = Wsims[i][idcs, :][:, idcs]
            L, Ctilde, perm = __extended_cholesky(Wi[1:, 1:])
            invperm = np.arange(len(perm))[np.argsort(perm)]
            r = L.shape[0]
            xtilde = Wi[np.vstack(([0], (perm+1).reshape((d-1,1)))), 0]
            u1 = np.matmul(np.linalg.inv(L), xtilde[1:(r+1)]).flatten()
            U11 = ncx2.rvs(df = nu-r, nc = max(0, xtilde[0] - np.vdot(u1, u1)), size = 1)
            U = norm.rvs(size=r) + u1
            B = np.hstack((B1, np.vstack((np.zeros(d-1), Ctilde[invperm, :]))))
            Y = np.zeros((d, d))
            Dr = np.eye(r)
            Y[:(r+1), :(r+1)] = np.hstack(
                (
                    np.vstack(([U11 + np.vdot(U, U)], U.reshape(r, 1))),
                    np.vstack((U, Dr))
                )
            )
            Wsims[i] = np.matmul(B, np.matmul(Y, np.transpose(B)))[idcs, :][:, idcs]
    return Wsims


def __rwishart_aa(n, nu, Sigma, Theta):
    L, Ctilde, perm = __extended_cholesky(Sigma)
    invperm = np.arange(len(perm))[np.argsort(perm)]
    theta = Ctilde[invperm, :]
    thetainv = np.linalg.inv(Ctilde)[:, invperm]
    Gamma = np.matmul(thetainv, np.matmul(Theta, np.transpose(thetainv)))
    Y = __rwishart_aa_m(n, nu, L.shape[0], Gamma)
    ttheta = np.transpose(theta)
    return list(map(lambda x: np.matmul(theta, np.matmul(x, ttheta)), Y))


def __is_symmetric(S):
    tS = np.transpose(S)
    return S.shape == tS.shape and np.allclose(S, tS)


def rwishart(n, nu, Sigma, Theta=None):
    """Wishart sampler.
    

    Parameters
    ----------
    n : integer
        desired number of simulated matrices
    nu : float
        degrees of freedom, must be >= p-1, where `p` is the dimension (the order of `Sigma`)
    Sigma : matrix
        scale matrix, must be a real positive semidefinite symmetric matrix
    Theta : matrix, optional
        non-centrality matrix, must be a real positive semidefinite symmetric matrix of the same order as `Sigma`; setting it to `None` (the default) is equivalent to setting it to the zero matrix

    Returns
    -------
    W : list of matrices
        A list of `n` matrices.

    """
    if not __is_symmetric(Sigma):
        raise ValueError("`Sigma` is not symmetric.")
    if Theta is not None and not __is_symmetric(Theta):
        raise ValueError("`Theta` is not symmetric.")
    p = Sigma.shape[0]
    if Theta is not None and Theta.shape[0] != p:
        raise ValueError("`Sigma` and `Theta` must have the same dimensions.")
    if nu < p-1:
        raise ValueError("`nu` must be at least equal to `p-1`.")
    W = [None]*n
    if nu > 2*p - 1:
        SigmaRoot = __matrix_root(Sigma)
        if SigmaRoot is None:
            raise ValueError("`Sigma` is not positive.")
        ThetaRoot = __matrix_root(Theta) if Theta is not None else None
        if Theta is not None and ThetaRoot is None:
            raise ValueError("`Theta` is not positive.")
        cholesky_sims = __rwishart_cholesky(n, nu - p, p)
        for i in range(n):
            Z = norm.rvs(size = (p, p))
            M = np.matmul(SigmaRoot, cholesky_sims[i])
            W1 = np.matmul(SigmaRoot, Z)
            W2 = np.transpose(W1)
            if Theta is not None:
                W1 = ThetaRoot + W1
                W2 = ThetaRoot + W2
            W[i] = np.matmul(W1, W2) + np.matmul(M, np.transpose(M))
    elif nu == floor(nu) and nu != p-1: # nu is an integer >= p
        SigmaRoot = __matrix_root(Sigma)
        if SigmaRoot is None:
            raise ValueError("`Sigma` is not positive.")
        ThetaRoot = __matrix_root(Theta) if Theta is not None else None
        if Theta is not None and ThetaRoot is None:
            raise ValueError("`Theta` is not positive.")
        if nu != p:
            for i in range(n):
                Z = norm.rvs(size = (p, p))
                Y = norm.rvs(size = (p, nu - p))
                W1 = np.matmul(SigmaRoot, Z)
                W2 = np.transpose(W1)
                W3 = np.matmul(SigmaRoot, Y)
                W4 = np.transpose(W3)
                if Theta is not None:
                    W1 = ThetaRoot + W1
                    W2 = ThetaRoot + W2
                W[i] = np.matmul(W1, W2) + np.matmul(W3, W4)
        else: # nu = p
            for i in range(n):
                M = np.matmul(SigmaRoot, norm.rvs(size = (p, p)))
                if Theta is not None:
                    M = ThetaRoot + M
                W[i] = np.matmul(M, np.transpose(M))
    else: # nu is not an integer or nu = p-1
        W = __rwishart_aa(n, nu, Sigma, Theta)  
    return W
