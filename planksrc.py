from numpy                      import array, float32, int32, zeros, pi, exp, dot, savetxt, diag, trace
from numpy.linalg               import norm, eigh, solve
from math                       import ceil, sqrt, log
from scipy.special              import factorial2, comb
from sys                        import argv
from itertools                  import combinations_with_replacement, combinations
from mpi4py                     import MPI
from plank.routines.oneelectron import overlapcgtos, kineticcgtos
from plank.routines.twoelectron import nuclearcgtos, ericgtos
from plank.routines.hatreefock  import computecorehamiltonian, canonicalorthogonalization, computehamiltonian
from plank.routines.hatreefock  import restrictedhatreefock, restrictedhatreefockDIIS, totalenergy
from plank.routines.hatreefock  import diisorthogonalization
from yaml                       import safe_load


with open("periodicTable.yaml") as periodic:
    periodictable   =   safe_load(periodic)
    periodicTable   =   {y: x[y] for x in periodictable for y in x}


def getatom(atomname):
    return periodicTable[atomname]


def nuclearenergy(atomobjects):
    atompairs       =   list(combinations(atomobjects,2))
    nuclearenergy   =   0.0
    for atompair in atompairs:
        center1         =   atompair[0].center
        charge1         =   atompair[0].charge
        center2         =   atompair[1].center
        charge2         =   atompair[1].charge
        distance        =   norm(center1-center2)
        energy          =   charge1*charge2/distance
        nuclearenergy   =   nuclearenergy+energy
    return nuclearenergy


def overlap(basisobjects):
    overlaps    =   []
    for basisobjecttuple in basisobjects:
        result  =   overlapcgtos(basisobjecttuple[0], basisobjecttuple[1])
        overlaps.append((result, basisobjecttuple[0].basisindex, basisobjecttuple[1].basisindex))
    return overlaps


def kinetic(basisobjects):
    kineticintegrals =   []
    for basisobjecttuple in basisobjects:
        result  =   kineticcgtos(basisobjecttuple[0], basisobjecttuple[1])
        kineticintegrals.append((result, basisobjecttuple[0].basisindex, basisobjecttuple[1].basisindex))
    return kineticintegrals


def nuclear(basisobjects, atomobjects):
    eniintegrals    =   []
    for basisobjecttuple in basisobjects:
        result      =   nuclearcgtos(basisobjecttuple[0], basisobjecttuple[1], atomobjects)
        eniintegrals.append((result, basisobjecttuple[0].basisindex, basisobjecttuple[1].basisindex))
    return eniintegrals


def electronic(basisobjects):
    eriintegrals    =   []
    for basisobjectquartet in basisobjects:
        result  =   ericgtos(basisobjectquartet[0], basisobjectquartet[1], basisobjectquartet[2], basisobjectquartet[3])
        eriintegrals.append((result, basisobjectquartet[0].basisindex, basisobjectquartet[1].basisindex, basisobjectquartet[2].basisindex, basisobjectquartet[3].basisindex))
    return eriintegrals


def integralEngine(molecule, basisobjects, dimensions, calctype="overlap"):
    results     =   []
    if calctype == "overlap":
        rank            =   comm.Get_rank()
        nprocs          =   comm.Get_size()
        shellpairs      =   list(combinations_with_replacement(basisobjects, 2))
        nshellpairs     =   len(shellpairs)
        nshellpair      =   ceil(nshellpairs/nprocs)
        shellpairi      =   rank*nshellpair
        shellpairf      =   (rank+1)*nshellpair
        overlapresults  =   overlap(shellpairs[shellpairi:shellpairf])

        if rank != 0:
            comm.send(overlapresults, dest=0)
        else:
            for result in overlapresults:
                results.append(result)
            for proc in range(1, nprocs):
                overlapresults  =   comm.recv(source=proc)
                for result in overlapresults:
                    results.append(result)

        if rank == 0:
            molecule.overlapmat =   zeros((dimensions, dimensions))
            for data in results:
                molecule.overlapmat[data[1], data[2]]   =   data[0]
                molecule.overlapmat[data[2], data[1]]   =   data[0]

        molecule.overlapmat =   comm.bcast(molecule.overlapmat, root=0)

    if calctype == "kinetic":
        rank            =   comm.Get_rank()
        nprocs          =   comm.Get_size()
        shellpairs      =   list(combinations_with_replacement(basisobjects, 2))
        nshellpairs     =   len(shellpairs)
        nshellpair      =   ceil(nshellpairs/nprocs)
        shellpairi      =   rank*nshellpair
        shellpairf      =   (rank+1)*nshellpair
        kineticresults  =   kinetic(shellpairs[shellpairi:shellpairf])

        if rank != 0:
            comm.send(kineticresults, dest=0)
        else:
            for result in kineticresults:
                results.append(result)
            for proc in range(1, nprocs):
                kineticresults  =   comm.recv(source=proc)
                for result in kineticresults:
                    results.append(result)

        if rank == 0:
            molecule.kineticmat =   zeros((dimensions, dimensions))
            for data in results:
                molecule.kineticmat[data[1], data[2]]   =   data[0]
                molecule.kineticmat[data[2], data[1]]   =   data[0]

        molecule.kineticmat =   comm.bcast(molecule.kineticmat, root=0)

    if calctype == "eni":
        rank            =   comm.Get_rank()
        nprocs          =   comm.Get_size()
        shellpairs      =   list(combinations_with_replacement(basisobjects, 2))
        atomobjects     =   [x for x in molecule.geometry]
        nshellpairs     =   len(shellpairs)
        nshellpair      =   ceil(nshellpairs/nprocs)
        shellpairi      =   rank*nshellpair
        shellpairf      =   (rank+1)*nshellpair
        eniresults      =   nuclear(shellpairs[shellpairi:shellpairf], atomobjects)

        if rank != 0:
            comm.send(eniresults, dest=0)
        else:
            for result in eniresults:
                results.append(result)
            for proc in range(1, nprocs):
                eniresults  =   comm.recv(source=proc)
                for result in eniresults:
                    results.append(result)

        if rank == 0:
            molecule.enimat =   zeros((dimensions, dimensions))
            for data in results:
                molecule.enimat[data[1], data[2]]   =   data[0]
                molecule.enimat[data[2], data[1]]   =   data[0]

        molecule.enimat =   comm.bcast(molecule.enimat, root=0)

    if calctype == "eri":
        rank            =   comm.Get_rank()
        nprocs          =   comm.Get_size()
        shellquartets  =   []
        for basisobjectA in basisObjects:
            for basisobjectB in basisObjects:
                for basisobjectC in basisObjects:
                    for basisobjectD in basisObjects:
                        index1  =   basisobjectA.basisindex
                        index2  =   basisobjectB.basisindex
                        index3  =   basisobjectC.basisindex
                        index4  =   basisobjectD.basisindex
                        index12 =   (index1*(index1+1)//2)+index2
                        index34 =   (index3*(index3+1)//2)+index4
                        if index12 >= index34:
                            shellquartets.append((basisobjectA, basisobjectB, basisobjectC, basisobjectD))
        nshellquartets  =   len(shellquartets)
        nshellquartet   =   ceil(nshellquartets/nprocs)
        shellpairi      =   rank*nshellquartet
        shellpairf      =   (rank+1)*nshellquartet
        eriresults      =   electronic(shellquartets[shellpairi:shellpairf])

        if rank != 0:
            comm.send(eriresults, dest=0)
        else:
            for result in eriresults:
                results.append(result)
            for proc in range(1, nprocs):
                eriresults  =   comm.recv(source=proc)
                for result in eriresults:
                    results.append(result)

        if rank == 0:
            molecule.erimat =   zeros((dimensions, dimensions, dimensions, dimensions))
            for result in results:
                molecule.erimat[result[1], result[2], result[3], result[4]]  =   result[0]
                molecule.erimat[result[3], result[4], result[1], result[2]]  =   result[0]
                molecule.erimat[result[2], result[1], result[4], result[3]]  =   result[0]
                molecule.erimat[result[4], result[3], result[2], result[1]]  =   result[0]
                molecule.erimat[result[2], result[1], result[3], result[4]]  =   result[0]
                molecule.erimat[result[4], result[3], result[1], result[2]]  =   result[0]
                molecule.erimat[result[1], result[2], result[4], result[3]]  =   result[0]
                molecule.erimat[result[3], result[4], result[2], result[1]]  =   result[0]

        molecule.erimat =   comm.bcast(molecule.erimat, root=0)


def scfEngine(molecule, dimensions):
    density     =   zeros((dimensions, dimensions))
    orthomat    =   canonicalorthogonalization(molecule.overlapmat)
    diisortho   =   diisorthogonalization(molecule.overlapmat)
    core        =   computecorehamiltonian(molecule.kineticmat, molecule.enimat) 
    for iteration in range(1, molecule.maxiterations):
        if iteration <= 2 or molecule.DIIS == False:
            orthofock, fock, hamiltonian, orbitals, coeffs, density, rmsdensity, maxdensity =   restrictedhatreefock(molecule.nelectrons, core, diisortho, density, molecule.erimat)
            molenergy   =   totalenergy(molecule.nuclearenergy, fock, density, core) 
            print("{:10.0f}{:20.5f}{:20.5f}{:20.5f}".format(iteration, molenergy, log(rmsdensity), log(maxdensity)))
        if iteration > 2 and molecule.DIIS == True:
            orthofock, fock, hamiltonian, orbitals, coeffs, density, rmsdensity, maxdensity =   restrictedhatreefockDIIS(molecule, molecule.nelectrons, core, diisortho, density, molecule.erimat)
            molenergy   =   totalenergy(molecule.nuclearenergy, fock, density, core) 
            print("{:10.0f}{:20.5f}{:20.5f}{:20.5f}".format(iteration, molenergy, log(rmsdensity), log(maxdensity)))            
        if log(rmsdensity) <= -20.0 and log(maxdensity) <= -20.0:
            molecule.scfstatus  =   True
            molecule.density    =   density
            molecule.fock       =   fock
            molecule.orbitals   =   orbitals
            molecule.coeffs     =   coeffs
            print("SCF CONVERGENCE ACHIEVED IN {:10.0f} ITERATIONS".format(iteration))
            break



class Basis(object):
    """ docstring for Basis

    Basis class constructs one basis object of type (s,p,d or f) depending on the basis set
    passed to the code. The generation of basis object happens during the construction of the
    atom object in the class Atom, with the list of basis objects tied to the atom object.
    """
    def __init__(self, shell, exponents, coefficients, center):
        super(Basis, self).__init__()
        self.shell          =   array(shell)
        self.exponents      =   array(exponents)
        self.coefficients   =   array(coefficients)
        self.normcoeffs     =   zeros(self.coefficients.size)
        self.center         =   array(center)
        self.basisindex     =   0

        self.normalizepGTO()

    def normalizepGTO(self):
        ll, mm, nn          =   self.shell
        totalangmomentum    =   sum(self.shell)
        prefactorpGTO       =   pow(2, 2*totalangmomentum)*pow(2, 1.5)/factorial2(2*ll-1)/factorial2(2*mm-1)/factorial2(2*nn-1)/pow(pi, 1.5)

        for index, exponent in enumerate(self.exponents):
            self.normcoeffs[index]  =   sqrt(pow(exponent, totalangmomentum)*pow(exponent, 1.5)*prefactorpGTO)

        prefactorcGTO   =   pow(pi, 1.5)*factorial2(2*ll-1)*factorial2(2*mm-1)*factorial2(2*nn-1)/pow(2.0, totalangmomentum)
        normalfactor    =   0.0

        for index1, coefficient1 in enumerate(self.coefficients):
            for index2, coefficient2 in enumerate(self.coefficients):
                t_normalfactor  =   (self.normcoeffs[index1]*self.normcoeffs[index2]*self.coefficients[index1]*self.coefficients[index2])/(pow(self.exponents[index1]+self.exponents[index2], totalangmomentum+1.5))
                normalfactor    =   normalfactor + t_normalfactor

        normalfactor    =   prefactorcGTO*normalfactor
        normalfactor    =   pow(normalfactor, -0.5)

        for index, coefficient in enumerate(self.coefficients):
            self.coefficients[index]  =   self.coefficients[index]*normalfactor


class Atom(object):
    """docstring for Atom."""

    def __init__(self, atomname, center, charge, basisset='sto-3g'):
        super(Atom, self).__init__()
        self.atomname   =   atomname
        self.center     =   array(center)*1.8897259886
        self.basisset   =   basisset
        self.shells     =   []
        self.charge     =   charge
        self.readBasis()

    def readBasis(self):
        self.basisfile  =   "./basis/"+self.atomname+"."+self.basisset+".gbs"
        with open(self.basisfile) as target:
            basisdata   =   target.readlines()
            headerline  =   basisdata[0].split()[0]
            _atomname   =   headerline
            assert (_atomname == self.atomname), "Wrong basis set"
            for lnumber, line in enumerate(basisdata[1:]):
                if "S" in line and "P" not in line:
                    nprims          =   int(line.split()[1])
                    pgtodata        =   [x.replace('D', 'E').split() for x in basisdata[lnumber+2:lnumber+2+nprims]]
                    exponents       =   [float(x[0]) for x in pgtodata]
                    coefficients    =   [float(x[1]) for x in pgtodata]
                    shell00         =   [0, 0, 0]
                    self.shells.append(Basis(shell00, exponents, coefficients, self.center))

                if "P" in line:
                    nprims          =   int(line.split()[1])
                    pgtodata        =   [x.replace('D', 'E').split() for x in basisdata[lnumber+2:lnumber+2+nprims]]
                    coefficient1    =   [float(x[1]) for x in pgtodata]
                    coefficient2    =   [float(x[2]) for x in pgtodata]
                    exponents       =   [float(x[0]) for x in pgtodata]
                    shell00         =   [0, 0, 0]
                    self.shells.append(Basis(shell00, exponents, coefficient1, self.center))
                    shell11         =   [1, 0, 0]
                    self.shells.append(Basis(shell11, exponents, coefficient2, self.center))
                    shell12         =   [0, 1, 0]
                    self.shells.append(Basis(shell12, exponents, coefficient2, self.center))
                    shell13         =   [0, 0, 1]
                    self.shells.append(Basis(shell13, exponents, coefficient2, self.center))

                if "D" in line and "+" not in line:
                    nprims          =   int(line.split()[1])
                    pgtodata        =   [x.replace('D', 'E').split() for x in basisdata[lnumber+2:lnumber+2+nprims]]
                    exponents       =   [float(x[0]) for x in pgtodata]
                    coefficients    =   [float(x[1]) for x in pgtodata]
                    shell20         =   [2, 0, 0]
                    self.shells.append(Basis(shell20, exponents, coefficients, self.center))
                    shell21         =   [1, 1, 0]
                    self.shells.append(Basis(shell21, exponents, coefficients, self.center))
                    shell22         =   [1, 0, 1]
                    self.shells.append(Basis(shell22, exponents, coefficients, self.center))
                    shell23         =   [0, 2, 0]
                    self.shells.append(Basis(shell23, exponents, coefficients, self.center))
                    shell24         =   [0, 1, 1]
                    self.shells.append(Basis(shell24, exponents, coefficients, self.center))
                    shell25         =   [0, 0, 2]
                    self.shells.append(Basis(shell25, exponents, coefficients, self.center))


class Geometry(object):
    """docstring for Geometry."""

    def __init__(self, inputfile):
        super(Geometry, self).__init__()
        self.inputfile      =   inputfile
        self.charge         =   0
        self.natoms         =   0
        self.nelectrons     =   0
        self.calctype       =   'energy'
        self.basisset       =   'sto-3g'
        self.geometry       =   []
        self.atomobjects    =   []
        self.density        =   []
        self.scfstatus      =   False
        self.maxiterations  =   100
        self.readgeometry()
        self.overlapmat     =   []
        self.kineticmat     =   []
        self.enimat         =   []
        self.erimat         =   []
        self.nuclearenergy  =   0.0
        self.totalenergy    =   0.0
        self.diisfock       =   []
        self.diiserror      =   []
        self.DIIS

    def readgeometry(self):
        with open(self.inputfile) as file:
            inputdata       =   file.readlines()
            calcbasis       =   inputdata[0].split()
            self.calctype   =   calcbasis[0]
            self.basisset   =   calcbasis[1]
            chargemul       =   inputdata[1].split()
            self.charge     =   int(chargemul[0])
            multiplicity    =   int(chargemul[1])
            try:
                assert multiplicity == 1, "[ERROR] : Only closed-shell singlets are currently supported."
            except AssertionError as Message:
                exit(Message)
            natoms          =   int(inputdata[2].split()[0])
            coorddata       =   [x for x in inputdata[3:3+natoms]]
            for atom in coorddata:
                atomdata        =   atom.split()
                atomname        =   atomdata[0]
                charge          =   getatom(atomname)
                coords          =   [float(x) for x in atomdata[1:]]
                self.nelectrons +=  charge
                self.geometry.append(Atom(atomname, coords, charge, self.basisset))
            iterdiisline     =   inputdata[-1].split()
            if iterdiisline  != []:
                self.maxiterations  =   int(iterdiisline[0])   
                self.DIIS           =   True if (iterdiisline[1] == 'T') else False


inputfilename       =   argv[1]
logfilename         =   inputfilename[:-4]+".log"
molecule            =   Geometry(inputfilename)
comm                =   MPI.COMM_WORLD
atomObjects         =   [x for x in molecule.geometry]
basisObjects        =   [y for x in atomObjects for y in x.shells]
nbasisObjects       =   len(basisObjects)
for counter, x in enumerate(basisObjects):
    x.basisindex    =   counter

integralEngine(molecule, basisObjects, nbasisObjects, "overlap")
integralEngine(molecule, basisObjects, nbasisObjects, "kinetic")
integralEngine(molecule, basisObjects, nbasisObjects, "eni")
integralEngine(molecule, basisObjects, nbasisObjects, "eri")

rank    =   comm.Get_rank()
if rank == 0:
    with open("overlap."+inputfilename[:-4]+".txt", "w") as t:
        for row in molecule.overlapmat:
            for column in row:
                t.write("{:10.3f}".format(column))
            t.write("\n")

    with open("kinetic."+inputfilename[:-4]+".txt", "w") as t:
        for row in molecule.kineticmat:
            for column in row:
                t.write("{:10.3f}".format(column))
            t.write("\n")

    with open("nuclear."+inputfilename[:-4]+".txt", "w") as t:
        for row in molecule.enimat:
            for column in row:
                t.write("{:10.3f}".format(column))
            t.write("\n")

    with open("electronic."+inputfilename[:-4]+".txt", "w") as t:
        for axis1 in molecule.erimat:
            for axis2 in axis1:
                for axis3 in axis2:
                    for axis4 in axis3:
                        t.write("{:10.3f}".format(axis4))
                    t.write("\n")
                t.write("\n")
            t.write("\n")

    molecule.nuclearenergy  =   nuclearenergy(atomObjects)
    scfEngine(molecule, nbasisObjects)

    with open("density."+inputfilename[:-4]+".txt", "w") as d:
        savetxt(d, molecule.density, "%10.3f")

MPI.Finalize()
