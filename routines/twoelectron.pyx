import cython   as  cython
import numpy    as  np
cimport numpy   as  np

from scipy.special  import  comb, factorial2, factorial, hyp1f1
from libc.math      cimport exp,  pow, sqrt
from numpy          import  dot,  pi
from numpy.linalg   import norm

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
cdef double expansioncoeff1(int expansionindex, int shell1, float center1, int shell2, float center2, float gaussiancenter):
    cdef double t_expansioncoeff    =   0.0
    cdef int counter                =   0
    cdef double auxiliary           =   0.0
    for counter in range(max(0, expansionindex-shell2), min(expansionindex, shell1)+1):
        auxiliary           =   comb(shell1, counter)
        auxiliary           =   auxiliary*comb(shell2, expansionindex-counter)
        auxiliary           =   auxiliary*pow(gaussiancenter-center1, shell1-counter)
        auxiliary           =   auxiliary*pow(gaussiancenter-center2, shell2+counter-expansionindex)
        t_expansioncoeff    =   t_expansioncoeff+auxiliary
    return t_expansioncoeff

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double expansioncoeff2(int expansionindex1, int expansionindex2, int expansionindex3, int shell1, float center1, int shell2, float center2, float atomcenter, float gaussiancenter, float gamma):
    cdef double t_expansioncoeff    =   0.0
    cdef double epsilon             =   1.0/(4*gamma)
    t_expansioncoeff                =   expansioncoeff1(expansionindex1, shell1, center1, shell2, center2, gaussiancenter)
    t_expansioncoeff                =   t_expansioncoeff*factorial(expansionindex1)
    t_expansioncoeff                =   t_expansioncoeff*pow(-1, expansionindex1+expansionindex3)
    t_expansioncoeff                =   t_expansioncoeff*pow(gaussiancenter-atomcenter, expansionindex1-(2*expansionindex2)-(2*expansionindex3))
    t_expansioncoeff                =   t_expansioncoeff*pow(epsilon, expansionindex2+expansionindex3)
    t_expansioncoeff                =   t_expansioncoeff/factorial(expansionindex3, exact=True)
    t_expansioncoeff                =   t_expansioncoeff/factorial(expansionindex2, exact=True)
    t_expansioncoeff                =   t_expansioncoeff/factorial(expansionindex1-(2*expansionindex2)-(2*expansionindex3), exact=True)
    return t_expansioncoeff

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double boysfunction(int boysindex, double boysparameter):
    return hyp1f1(0.5+boysindex, 1.5+boysindex, -1*boysparameter)/((2*boysindex)+1)

@cython.boundscheck(False)
@cython.wraparound(False)
def nuclearcgtos(basisobject1, basisobject2, atomobjects):
    cdef double t_eniintegral   =   0.0
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
    for index1, exponent1 in enumerate(exponents1):
        for index2, exponent2 in enumerate(exponents2):
            t_nuclearintegral   =   0.0
            gamma               =   exponent1+exponent2
            gaussiancenter, gaussianintegral    =   gaussiantheorem(center1, exponent1, center2, exponent2)
            for atom in atomobjects:
                for expansionindex1A in range(0, shell1[0]+shell2[0]+1):
                    for expansionindex1B in range(0, int(expansionindex1A//2)+1):
                        for expansionindex1C in range(0, int((expansionindex1A-(2*expansionindex1B))//2)+1):
                            auxiliary1A =   expansioncoeff2(expansionindex1A, expansionindex1B, expansionindex1C, shell1[0], center1[0], shell2[0], center2[0], atom.center[0], gaussiancenter[0], gamma)
                            
                            for expansionindex2A in range(0, shell1[1]+shell2[1]+1):
                                for expansionindex2B in range(0, int(expansionindex2A//2)+1):
                                    for expansionindex2C in range(0, int((expansionindex2A-(2*expansionindex2B))//2)+1):
                                        auxiliary2A =   expansioncoeff2(expansionindex2A, expansionindex2B, expansionindex2C, shell1[1], center1[1], shell2[1], center2[1], atom.center[1], gaussiancenter[1], gamma)

                                        for expansionindex3A in range(0, shell1[2]+shell2[2]+1):
                                            for expansionindex3B in range(0, int(expansionindex3A//2)+1):
                                                for expansionindex3C in range(0, int((expansionindex3A-(2*expansionindex3B))//2)+1):
                                                    auxiliary3A =   expansioncoeff2(expansionindex3A, expansionindex3B, expansionindex3C, shell1[2], center1[2], shell2[2], center2[2], atom.center[2], gaussiancenter[2], gamma)
                                                    boysindex   =   (expansionindex1A+expansionindex2A+expansionindex3A)-2*(expansionindex1B+expansionindex2B+expansionindex3B)-(expansionindex1C+expansionindex2C+expansionindex3C)
                                                    boysparam   =   dot(atom.center-gaussiancenter, atom.center-gaussiancenter)*gamma
                                                    auxiliary4A =   boysfunction(boysindex, boysparam)
                                                    t_nuclearintegral   =   t_nuclearintegral+(auxiliary1A*auxiliary2A*auxiliary3A*auxiliary4A)*atom.charge*-1.0
            t_eniintegral   =   t_eniintegral+(t_nuclearintegral*coefficients1[index1]*coefficients2[index2]*normcoeffs1[index1]*normcoeffs2[index2]*gaussianintegral*(2*pi/gamma))
    return t_eniintegral

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double enigaussian(int shell1, int shell2, int nodes, float center1, float center2, float exponent1, float exponent2):
    cdef double gamma       =   exponent1+exponent2
    cdef double exponent    =   exponent1*(exponent2/gamma)

    if (nodes < 0) or (nodes > (shell1+shell2)):
        return 0
    elif nodes == shell1 == shell2 == 0:
        return exp(-1*exponent*pow(center1-center2, 2))
    elif shell2 == 0:
        return (1/(2*gamma))*enigaussian(shell1-1, shell2, nodes-1, center1, center2, exponent1, exponent2) - (exponent*(center1-center2)/exponent1)*enigaussian(shell1-1, shell2, nodes, center1, center2, exponent1, exponent2) + (nodes+1)*enigaussian(shell1-1, shell2, nodes+1, center1, center2, exponent1, exponent2)
    else:
        return (1/(2*gamma))*enigaussian(shell1, shell2-1, nodes-1, center1, center2, exponent1, exponent2) + (exponent*(center1-center2)/exponent2)*enigaussian(shell1, shell2-1, nodes, center1, center2, exponent1, exponent2) + (nodes+1)*enigaussian(shell1, shell2-1, nodes+1, center1, center2, exponent1, exponent2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double auxiliaryhermiteeri(int nodes1, int nodes2, int nodes3, int boysorder, float exponent, float xdist, float ydist, float zdist, float dist):
    cdef double boysparam   =   exponent*pow(dist, 2)
    cdef double auxhermite  =   0.0

    if nodes1 == nodes2 == nodes3 == 0:
        auxhermite  +=  pow(-2*exponent, boysorder)*boysfunction(boysorder, boysparam)
    elif nodes1 == nodes2 == 0:
        if nodes3 > 1:
            auxhermite  +=  (nodes3-1)*auxiliaryhermiteeri(nodes1, nodes2, nodes3-2, boysorder+1, exponent, xdist, ydist, zdist, dist)
        auxhermite  +=  zdist*auxiliaryhermiteeri(nodes1, nodes2, nodes3-1, boysorder+1, exponent, xdist, ydist, zdist, dist)
    elif nodes1 == 0:
        if nodes2 > 1:
            auxhermite  +=  (nodes2-1)*auxiliaryhermiteeri(nodes1, nodes2-2, nodes3, boysorder+1, exponent, xdist, ydist, zdist, dist)
        auxhermite  +=  ydist*auxiliaryhermiteeri(nodes1, nodes2-1, nodes3, boysorder+1, exponent, xdist, ydist, zdist, dist)
    else:
        if nodes1 > 1:
            auxhermite  +=  (nodes1-1)*auxiliaryhermiteeri(nodes1-2, nodes2, nodes3, boysorder+1, exponent, xdist, ydist, zdist, dist)
        auxhermite  +=  xdist*auxiliaryhermiteeri(nodes1-1, nodes2, nodes3, boysorder+1, exponent, xdist, ydist, zdist, dist)
    return auxhermite

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double eripgtos(exponent1, shell1, center1, exponent2, shell2, center2, exponent3, shell3, center3, exponent4, shell4, center4):
    cdef double     eripgto             =   0.0
    cdef np.ndarray gaussiancenterA     =   np.zeros(3)
    cdef double     gaussianintegralA   =   0.0
    cdef np.ndarray gaussiancenterB     =   np.zeros(3)
    cdef double     gaussianintegralB   =   0.0
    gaussiancenterA, gaussianintegralA  =   gaussiantheorem(center1, exponent1, center2, exponent2)
    gaussiancenterB, gaussianintegralB  =   gaussiantheorem(center3, exponent3, center4, exponent4)
    cdef double xdist                   =   gaussiancenterA[0]-gaussiancenterB[0]
    cdef double ydist                   =   gaussiancenterA[1]-gaussiancenterB[1]
    cdef double zdist                   =   gaussiancenterA[2]-gaussiancenterB[2]
    cdef double distance                =   norm(gaussiancenterA-gaussiancenterB)
    cdef double combexponentA           =   exponent1+exponent2
    cdef double combexponentB           =   exponent3+exponent4
    cdef double combexponent            =   combexponentA*(combexponentB/(combexponentA+combexponentB))

    cdef int index1                     =   0
    cdef int index2                     =   0
    cdef int index3                     =   0
    cdef int index4                     =   0
    cdef int index5                     =   0
    cdef int index6                     =   0
    for index1 in range(0, shell1[0]+shell2[0]+1):
        for index2 in range(0, shell1[1]+shell2[1]+1):
            for index3 in range(0, shell1[2]+shell2[2]+1):
                for index4 in range(0, shell3[0]+shell4[0]+1):
                    for index5 in range(0, shell3[1]+shell4[1]+1):
                        for index6 in range(0, shell3[2]+shell4[2]+1):
                            erigaussianA    =   enigaussian(shell1[0], shell2[0], index1, center1[0], center2[0], exponent1, exponent2)
                            erigaussianB    =   enigaussian(shell1[1], shell2[1], index2, center1[1], center2[1], exponent1, exponent2)
                            erigaussianC    =   enigaussian(shell1[2], shell2[2], index3, center1[2], center2[2], exponent1, exponent2)
                            erigaussianD    =   enigaussian(shell3[0], shell4[0], index4, center3[0], center4[0], exponent3, exponent4)
                            erigaussianE    =   enigaussian(shell3[1], shell4[1], index5, center3[1], center4[1], exponent3, exponent4)
                            erigaussianF    =   enigaussian(shell3[2], shell4[2], index6, center3[2], center4[2], exponent3, exponent4)
                            erigaussianG    =   auxiliaryhermiteeri(index1+index4, index2+index5, index3+index6, 0, combexponent, xdist, ydist, zdist, distance)
                            result          =   erigaussianA*erigaussianB*erigaussianC*erigaussianD*erigaussianE*erigaussianF*erigaussianG*pow(-1, index4+index5+index6)
                            eripgto         =   eripgto+result
    eripgto *=  2*pow(pi, 2.5)/(combexponentA*combexponentB*sqrt(combexponentA+combexponentB))
    return eripgto

@cython.boundscheck(False)
@cython.wraparound(False)
def ericgtos(basisobject1, basisobject2, basisobject3, basisobject4):
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

    cdef np.ndarray center3         =   basisobject3.center
    cdef np.ndarray exponents3      =   basisobject3.exponents
    cdef np.ndarray coefficients3   =   basisobject3.coefficients
    cdef np.ndarray shell3          =   basisobject3.shell
    cdef np.ndarray normcoeffs3     =   basisobject3.normcoeffs

    cdef np.ndarray center4         =   basisobject4.center
    cdef np.ndarray exponents4      =   basisobject4.exponents
    cdef np.ndarray coefficients4   =   basisobject4.coefficients
    cdef np.ndarray shell4          =   basisobject4.shell
    cdef np.ndarray normcoeffs4     =   basisobject4.normcoeffs

    cdef double ericgto =   0.0
    for index1, exponent1 in enumerate(exponents1):
        for index2, exponent2 in enumerate(exponents2):
            for index3, exponent3 in enumerate(exponents3):
                for index4, exponent4 in enumerate(exponents4):
                    result  =   eripgtos(exponent1, shell1, center1, exponent2, shell2, center2, exponent3, shell3, center3, exponent4, shell4, center4)
                    result  *=  coefficients1[index1]*normcoeffs1[index1]*coefficients2[index2]*normcoeffs2[index2]*coefficients3[index3]*normcoeffs3[index3]*coefficients4[index4]*normcoeffs4[index4]
                    ericgto =   ericgto+result
    return ericgto
