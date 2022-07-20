import cython   as  cython
import numpy    as  np
cimport numpy   as  np

from scipy.special  import  comb, factorial2
from libc.math      cimport exp,  pow
from numpy          import  dot,  pi

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussiantheorem(np.ndarray[np.float64_t] center1, double exponent1, np.ndarray[np.float64_t] center2, double exponent2):
    cdef np.ndarray gaussiancenter  =   np.zeros(3)
    cdef double gaussianexponent    =   0.0
    cdef double gaussianintegral    =   0.0
    gaussiancenter      =   ((exponent1*center1)+(exponent2*center2))/(exponent1+exponent2)
    gaussianexponent    =   (exponent1*exponent2)/(exponent1+exponent2)
    gaussianintegral    =   exp(-1*gaussianexponent*dot(center1-center2, center1-center2))
    return gaussiancenter, gaussianintegral

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double overlappgtos(double center1, double exponent1, int shell1, double center2, double exponent2, int shell2, double gaussiancenter):
    cdef double t_overlap   =   0.0
    cdef double auxiliary   =   0.0
    cdef int counter1, counter2
    for counter1 in range(0, shell1+1):
        for counter2 in range(0, shell2+1):
            if (counter1+counter2)%2 == 0:
                auxiliary   =   comb(shell1, counter1)
                auxiliary   =   auxiliary*comb(shell2, counter2)
                auxiliary   =   auxiliary*factorial2(counter1+counter2-1)
                auxiliary   =   auxiliary*pow(gaussiancenter-center1, shell1-counter1)
                auxiliary   =   auxiliary*pow(gaussiancenter-center2, shell2-counter2)
                auxiliary   =   auxiliary/pow(2*(exponent1+exponent2), 0.5*(counter1+counter2))
                t_overlap   =   t_overlap+auxiliary
    return t_overlap

@cython.boundscheck(False)
@cython.wraparound(False)
def overlapcgtos(basisobject1, basisobject2):
    cdef np.ndarray center1         =   basisobject1.center
    cdef np.ndarray exponents1      =   basisobject1.exponents
    cdef np.ndarray coefficients1   =   basisobject1.coefficients
    cdef np.ndarray shell1          =   basisobject1.shell
    cdef np.ndarray normcoeffs1     =   basisobject1.normcoeffs

    cdef np.ndarray center2         =   basisobject2.center
    cdef np.ndarray exponents2      =   basisobject2.exponents
    cdef np.ndarray coefficients2   =   basisobject2.coefficients
    cdef np.ndarray shell2          =   basisobject2.shell
    cdef np.ndarray normcoeffs2     =   basisobject2.normcoeffs

    cdef double overlaptotal        =   0.0
    cdef index1                     =   0
    cdef index2                     =   0
    cdef exponent1                  =   0
    cdef exponent2                  =   0
    cdef np.ndarray gaussiancenter  =   np.zeros(3)
    cdef double gaussianintegral    =   0.0
    for index1, exponent1 in enumerate(exponents1):
        for index2, exponent2 in enumerate(exponents2):
            gaussiancenter, gaussianintegral    =   gaussiantheorem(center1, exponent1, center2, exponent2)
            overlapx        =   overlappgtos(center1[0], exponent1, shell1[0], center2[0], exponent2, shell2[0], gaussiancenter[0])
            overlapy        =   overlappgtos(center1[1], exponent1, shell1[1], center2[1], exponent2, shell2[1], gaussiancenter[1])
            overlapz        =   overlappgtos(center1[2], exponent1, shell1[2], center2[2], exponent2, shell2[2], gaussiancenter[2])
            overlap         =   overlapx*overlapy*overlapz*gaussianintegral*pow(pi/(exponent1+exponent2), 1.5)
            overlaptotal    =   overlaptotal+(normcoeffs1[index1]*normcoeffs2[index2]*coefficients1[index1]*coefficients2[index2])*overlap
    return overlaptotal

@cython.boundscheck(False)
@cython.wraparound(False)
def kineticcgtos(basisobject1, basisobject2):
    cdef np.ndarray center1         =   basisobject1.center
    cdef np.ndarray exponents1      =   basisobject1.exponents
    cdef np.ndarray coefficients1   =   basisobject1.coefficients
    cdef np.ndarray shell1          =   basisobject1.shell
    cdef np.ndarray normcoeffs1     =   basisobject1.normcoeffs

    cdef np.ndarray center2         =   basisobject2.center
    cdef np.ndarray exponents2      =   basisobject2.exponents
    cdef np.ndarray coefficients2   =   basisobject2.coefficients
    cdef np.ndarray shell2          =   basisobject2.shell
    cdef np.ndarray normcoeffs2     =   basisobject2.normcoeffs

    cdef double Kx                  =   0.0
    cdef double Ky                  =   0.0
    cdef double Kz                  =   0.0
    cdef np.ndarray gaussiancenter  =   np.zeros(3)
    cdef double gaussianintegral    =   0.0
    for index1, exponent1 in enumerate(exponents1):
        for index2, exponent2 in enumerate(exponents2):
            gaussiancenter, gaussianintegral    =   gaussiantheorem(center1, exponent1, center2, exponent2)
            overlapx    =   overlappgtos(center1[0], exponent1, shell1[0], center2[0], exponent2, shell2[0], gaussiancenter[0])
            overlapy    =   overlappgtos(center1[1], exponent1, shell1[1], center2[1], exponent2, shell2[1], gaussiancenter[1])
            overlapz    =   overlappgtos(center1[2], exponent1, shell1[2], center2[2], exponent2, shell2[2], gaussiancenter[2])

            overlapx11  =   overlappgtos(center1[0], exponent1, shell1[0]-1, center2[0], exponent2, shell2[0]-1, gaussiancenter[0])
            overlapx12  =   overlappgtos(center1[0], exponent1, shell1[0]+1, center2[0], exponent2, shell2[0]-1, gaussiancenter[0])
            overlapx13  =   overlappgtos(center1[0], exponent1, shell1[0]-1, center2[0], exponent2, shell2[0]+1, gaussiancenter[0])
            overlapx14  =   overlappgtos(center1[0], exponent1, shell1[0]+1, center2[0], exponent2, shell2[0]+1, gaussiancenter[0])

            overlapy11  =   overlappgtos(center1[1], exponent1, shell1[1]-1, center2[1], exponent2, shell2[1]-1, gaussiancenter[1])
            overlapy12  =   overlappgtos(center1[1], exponent1, shell1[1]+1, center2[1], exponent2, shell2[1]-1, gaussiancenter[1])
            overlapy13  =   overlappgtos(center1[1], exponent1, shell1[1]-1, center2[1], exponent2, shell2[1]+1, gaussiancenter[1])
            overlapy14  =   overlappgtos(center1[1], exponent1, shell1[1]+1, center2[1], exponent2, shell2[1]+1, gaussiancenter[1])

            overlapz11  =   overlappgtos(center1[2], exponent1, shell1[2]-1, center2[2], exponent2, shell2[2]-1, gaussiancenter[2])
            overlapz12  =   overlappgtos(center1[2], exponent1, shell1[2]+1, center2[2], exponent2, shell2[2]-1, gaussiancenter[2])
            overlapz13  =   overlappgtos(center1[2], exponent1, shell1[2]-1, center2[2], exponent2, shell2[2]+1, gaussiancenter[2])
            overlapz14  =   overlappgtos(center1[2], exponent1, shell1[2]+1, center2[2], exponent2, shell2[2]+1, gaussiancenter[2])

            kx          =   shell1[0]*shell2[0]*overlapx11
            kx          +=  -2*exponent1*shell2[0]*overlapx12
            kx          +=  -2*exponent2*shell1[0]*overlapx13
            kx          +=  4*exponent1*exponent2*overlapx14

            ky          =   shell1[1]*shell2[1]*overlapy11
            ky          +=  -2*exponent1*shell2[1]*overlapy12
            ky          +=  -2*exponent2*shell1[1]*overlapy13
            ky          +=  4*exponent1*exponent2*overlapy14

            kz          =   shell1[2]*shell2[2]*overlapz11
            kz          +=  -2*exponent1*shell2[2]*overlapz12
            kz          +=  -2*exponent2*shell1[2]*overlapz13
            kz          +=  4*exponent1*exponent2*overlapz14

            Kx          +=  0.5*kx*overlapy*overlapz*gaussianintegral*pow(pi/(exponent1+exponent2), 1.5)*normcoeffs1[index1]*coefficients1[index1]*normcoeffs2[index2]*coefficients2[index2]
            Ky          +=  0.5*ky*overlapx*overlapz*gaussianintegral*pow(pi/(exponent1+exponent2), 1.5)*normcoeffs1[index1]*coefficients1[index1]*normcoeffs2[index2]*coefficients2[index2]
            Kz          +=  0.5*kz*overlapx*overlapy*gaussianintegral*pow(pi/(exponent1+exponent2), 1.5)*normcoeffs1[index1]*coefficients1[index1]*normcoeffs2[index2]*coefficients2[index2]
    return Kx+Ky+Kz
