import cython   as  cython
import numpy    as  np
cimport numpy   as  np

from scipy.special  import  comb, factorial2
from libc.math      cimport exp,  pow
from numpy          import  dot,  pi, diag, amax, trace, isclose
from numpy.linalg   import  eigh, solve

@cython.boundscheck(False)
@cython.wraparound(False)
def computecorehamiltonian(np.ndarray[np.float64_t, ndim=2]kineticmat, np.ndarray[np.float64_t, ndim=2]enimat):
    cdef int nbasis =   kineticmat[0].size
    cdef np.ndarray[np.float64_t, ndim=2] corehamiltonian   =   np.zeros((nbasis, nbasis))
    corehamiltonian =   kineticmat+enimat
    return corehamiltonian

@cython.boundscheck(False)
@cython.wraparound(False)
def canonicalorthogonalization(np.ndarray[np.float64_t, ndim=2] overlapmat):
    cdef int nbasis =   overlapmat[0].size
    cdef np.ndarray[np.float64_t, ndim=1]   eigenvalues     =   np.zeros(nbasis)
    cdef np.ndarray[np.float64_t, ndim=2]   t_eigenvalues   =   np.zeros((nbasis, nbasis))
    cdef np.ndarray[np.float64_t, ndim=2]   eigenvectors    =   np.zeros((nbasis, nbasis))
    cdef np.ndarray[np.float64_t, ndim=2]   orthogonalmat   =   np.zeros((nbasis, nbasis))
    eigenvalues, eigenvectors   =   eigh(overlapmat)
    t_eigenvalues               =   diag((eigenvalues)**-0.5)
    orthogonalmat               =   dot(eigenvectors, t_eigenvalues)
    return orthogonalmat

@cython.boundscheck(False)
@cython.wraparound(False)
def computehamiltonian(np.ndarray[np.float64_t, ndim=2] densitymat, np.ndarray[np.float64_t, ndim=4] erimat):
    cdef int nbasis =   densitymat[0].size
    cdef int index1
    cdef int index2
    cdef int index3
    cdef int index4
    cdef np.ndarray[np.float64_t, ndim=2]   hamiltonian =   np.zeros((nbasis, nbasis))
    for index1 in range(0, nbasis):
        for index2 in range(0, nbasis):
            for index3 in range(0, nbasis):
                for index4 in range(0, nbasis):
                    hamiltonian[index1, index2] =   hamiltonian[index1, index2]+(densitymat[index3, index4]*(erimat[index1, index2, index3, index4]-0.5*(erimat[index1, index4, index3, index2])))
    return hamiltonian

@cython.boundscheck(False)
@cython.wraparound(False)
def computedensity(int nelectrons, np.ndarray[np.float64_t, ndim=2] overlapmat):
    cdef int nbasis =   overlapmat[0].shape[0]
    cdef np.ndarray[np.float64_t, ndim=2]   densitymat  =   np.zeros((nbasis, nbasis))
    cdef int index1
    cdef int index2
    cdef int index3
    for index1 in range(0, nbasis):
        for index2 in range(0, nbasis):
            for index3 in range(0, int(nelectrons//2)):
                densitymat[index1, index2]  +=  2.0*overlapmat[index1, index3]*overlapmat[index2, index3]
    return densitymat

@cython.boundscheck(False)
@cython.wraparound(False)
def diisorthogonalization(np.ndarray[np.float64_t, ndim=2] overlapmat):
    eigenvalues, eigenvectors   =   eigh(overlapmat)
    halfeigmat                  =   diag((eigenvalues)**-0.5)
    rightmat                    =   dot(eigenvectors, halfeigmat)
    orthomat                    =   dot(rightmat, eigenvectors.T)
    return orthomat

@cython.boundscheck(False)
def diisEngine(molecule, np.ndarray[np.float64_t, ndim=2] fock, np.ndarray[np.float64_t, ndim=2] density):
    cdef int dimensions         =   fock[0].size
    eigenvalues, eigenvectors   =   eigh(molecule.overlapmat)
    halfeigmat                  =   diag((eigenvalues)**-0.5)
    rightmat                    =   dot(eigenvectors, halfeigmat)
    orthomat                    =   dot(rightmat, eigenvectors.T)
    t_errorvector               =   dot(fock, dot(density, molecule.overlapmat))-dot(molecule.overlapmat, dot(density, fock))
    leftmat                     =   dot(orthomat.T, t_errorvector)
    errorvector                 =   dot(leftmat, orthomat)
    molecule.diisfock.append(fock)
    molecule.diiserror.append(errorvector)
    cdef int fockdimension               =   len(molecule.diisfock)
    if fockdimension > 6:
        del molecule.diisfock[0]
        del molecule.diiserror[0]
        fockdimension   =   fockdimension-1
    cdef np.ndarray[np.float64_t, ndim=2] bmatrix   =   np.zeros((fockdimension+1, fockdimension+1))
    bmatrix[-1,:]               =   -1.0
    bmatrix[:,-1]               =   -1.0
    bmatrix[-1,-1]              =   0.0
    for index1 in range(0, fockdimension):
        for index2 in range(0, index1+1):
            bmatrix[index1, index2] =   trace(dot(molecule.diiserror[index1].T, molecule.diiserror[index2]))
            bmatrix[index2, index1] =   trace(dot(molecule.diiserror[index1].T, molecule.diiserror[index2]))
    cdef np.ndarray[np.float64_t, ndim=1] residue    =   np.zeros(fockdimension+1)
    residue[-1]                 =   -1.0
    cdef np.ndarray[np.float64_t, ndim=1] weights    =   solve(bmatrix, residue)
    assert isclose(sum(weights[:-1]),1.0)
    cdef np.ndarray[np.float64_t, ndim=2] newfock    =   np.zeros((dimensions, dimensions))
    for index, fockelement in enumerate(molecule.diisfock):
        newfock =   newfock+(weights[index]*fockelement)
    return newfock

@cython.boundscheck(False)
@cython.wraparound(False)
def restrictedhatreefock(int nelectrons, np.ndarray[np.float64_t, ndim=2] corehamiltonian, np.ndarray[np.float64_t, ndim=2] orthogonalmat, np.ndarray[np.float64_t, ndim=2] densitymat, np.ndarray[np.float64_t, ndim=4] erimat):
    cdef int nbasis =   corehamiltonian[0].size
    cdef np.ndarray[np.float64_t, ndim=2]   hamiltonianmat  =   computehamiltonian(densitymat, erimat)
    cdef np.ndarray[np.float64_t, ndim=2]   fockmatrix      =   hamiltonianmat+corehamiltonian
    cdef np.ndarray[np.float64_t, ndim=2]   orthofockmatrix =   dot(orthogonalmat.conj().T, dot(fockmatrix, orthogonalmat))
    cdef np.ndarray[np.float64_t, ndim=1]   orbitalenergies =   np.zeros(nbasis)
    cdef np.ndarray[np.float64_t, ndim=2]   orthocoeffs     =   np.zeros((nbasis, nbasis))
    orbitalenergies, orthocoeffs                            =   eigh(orthofockmatrix)
    cdef np.ndarray[np.float64_t, ndim=2]   canonicalcoeffs =   dot(orthogonalmat, orthocoeffs)
    cdef np.ndarray[np.float64_t, ndim=2]   t_densitymat    =   computedensity(nelectrons, canonicalcoeffs)
    cdef double maxdensitydiff
    cdef double rmsdensitydiff
    maxdensitydiff, rmsdensitydiff                          =   checkconvergence(densitymat, t_densitymat)
    return orthofockmatrix, fockmatrix, hamiltonianmat, orbitalenergies, canonicalcoeffs, t_densitymat, maxdensitydiff, rmsdensitydiff

@cython.boundscheck(False)
@cython.wraparound(False)
def restrictedhatreefockDIIS(molecule, int nelectrons, np.ndarray[np.float64_t, ndim=2] corehamiltonian, np.ndarray[np.float64_t, ndim=2] orthogonalmat, np.ndarray[np.float64_t, ndim=2] densitymat, np.ndarray[np.float64_t, ndim=4] erimat):
    cdef int nbasis =   corehamiltonian[0].size
    cdef np.ndarray[np.float64_t, ndim=2]   hamiltonianmat  =   computehamiltonian(densitymat, erimat)
    cdef np.ndarray[np.float64_t, ndim=2]   fockmatrix      =   hamiltonianmat+corehamiltonian
    cdef np.ndarray[np.float64_t, ndim=2]   diisfock        =   diisEngine(molecule, fockmatrix, densitymat)
    cdef np.ndarray[np.float64_t, ndim=2]   orthofockmatrix =   dot(orthogonalmat.conj().T, dot(diisfock, orthogonalmat))
    cdef np.ndarray[np.float64_t, ndim=1]   orbitalenergies =   np.zeros(nbasis)
    cdef np.ndarray[np.float64_t, ndim=2]   orthocoeffs     =   np.zeros((nbasis, nbasis))
    orbitalenergies, orthocoeffs                            =   eigh(orthofockmatrix)
    cdef np.ndarray[np.float64_t, ndim=2]   canonicalcoeffs =   dot(orthogonalmat, orthocoeffs)
    cdef np.ndarray[np.float64_t, ndim=2]   t_densitymat    =   computedensity(nelectrons, canonicalcoeffs)
    cdef double maxdensitydiff
    cdef double rmsdensitydiff
    maxdensitydiff, rmsdensitydiff                          =   checkconvergence(densitymat, t_densitymat)
    return orthofockmatrix, fockmatrix, hamiltonianmat, orbitalenergies, canonicalcoeffs, t_densitymat, maxdensitydiff, rmsdensitydiff

@cython.boundscheck(False)
@cython.wraparound(False)
def checkconvergence(np.ndarray[np.float64_t, ndim=2] densitymat, np.ndarray[np.float64_t, ndim=2] t_densitymat):
    cdef np.ndarray[np.float64_t, ndim=2]   densitydiff     =   t_densitymat-densitymat
    cdef double maxdensitydiff  =   amax(densitydiff)
    cdef double rmsdensitydiff  =   pow(sum(sum((densitydiff)**2))/4.0, 0.5)
    return maxdensitydiff, rmsdensitydiff

@cython.boundscheck(False)
@cython.wraparound(False)
def totalenergy(float nucenergy, np.ndarray[np.float64_t, ndim=2] fockmatrix, np.ndarray[np.float64_t, ndim=2] densitymatrix, np.ndarray[np.float64_t, ndim=2] corehamiltonian):
    cdef double t_energy    =   0.0
    cdef int nbasis         =   fockmatrix[0].size
    for index1 in range(0, nbasis):
        for index2 in range(0, nbasis):
            t_energy    =   t_energy+(0.5*densitymatrix[index1,index2]*(fockmatrix[index1,index2]+corehamiltonian[index1,index2]))
    return t_energy+nucenergy
