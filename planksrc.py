from numpy                      import array, float32, int32, zeros, pi, exp, dot
from numpy.linalg               import norm
from math                       import ceil, sqrt, log
from scipy.special              import factorial2, comb
from sys                        import argv
from itertools                  import combinations_with_replacement, combinations
from mpi4py                     import MPI
from yaml                       import safe_load
from plank.routines.oneelectron import overlapcgtos, kineticcgtos
from plank.routines.twoelectron import nuclearcgtos

with open("periodicTable.yaml") as periodic:
    periodictable   =   safe_load(periodic)
    periodicTable   =   {y: x[y] for x in periodictable for y in x}


def getatom(atomname):
    return periodicTable[atomname]


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


def logFile(molecule, data, heading, lastaccess=0):
    logfilename     =   molecule.inputfile[:-4]+".log"
    accessmodifier  =   "w" if (lastaccess == 0) else "a"
    with open(logfilename, accessmodifier) as logf:
        logf.write("{:^45}\n".format(heading))
        logf.write("{:^45}\n".format("-"*40))
        if heading == "OVERLAP INTEGRALS":
            logf.write("{:^15}{:^15}{:^15}\n".format("INDEX1", "INDEX2", "OVERLAP"))
            logf.write("{:^45}\n".format("-"*40))
            for result in data:
                logf.write("{:8.0f}{:15.0f}{:17.3f}\n".format(result[1], result[2], result[0]))
        logf.write("{}".format('\n'))

        if heading == "KINETIC INTEGRALS":
            logf.write("{:^15}{:^15}{:^15}\n".format("INDEX1", "INDEX2", "KINETIC"))
            logf.write("{:^45}\n".format("-"*40))
            for result in data:
                logf.write("{:8.0f}{:15.0f}{:17.3f}\n".format(result[1], result[2], result[0]))
        logf.write("{}".format('\n'))


def integralEngine(molecule, basisobjects, dimensions, calctype="overlap", verbose=False, lastaccess=0):
    results     =   []
    if calctype == "overlap":
        comm            =   MPI.COMM_WORLD
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
            molecule.overlapmat =   results
            if verbose is True:
                logFile(molecule, results, "OVERLAP INTEGRALS", lastaccess)
                lastaccess  =   lastaccess+1

        molecule.overlapmat =   comm.bcast(results, root=0)
        return lastaccess
        MPI.Finalize

    if calctype == "kinetic":
        comm            =   MPI.COMM_WORLD
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
            molecule.kineticpmat =   results
            if verbose is True:
                logFile(molecule, results, "KINETIC INTEGRALS", lastaccess)
                lastaccess  =   lastaccess+1

        molecule.kineticmat =   comm.bcast(results, root=0)
        return lastaccess
        MPI.Finalize

    if calctype == "eni":
        comm            =   MPI.COMM_WORLD
        rank            =   comm.Get_rank()
        nprocs          =   comm.Get_size()
        shellpairs      =   list(combinations_with_replacement(basisobjects, 2))
        atomobjects     =   [x for x in molecule.geometry]
        nshellpairs     =   len(shellpairs)
        nshellpair      =   ceil(nshellpairs/nprocs)
        shellpairi      =   rank*nshellpair
        shellpairf      =   (rank+1)*nshellpair
        kineticresults  =   kinetic(shellpairs[shellpairi:shellpairf])


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
        self.readgeometry()
        self.overlapmat     =   []
        self.kineticmat     =   []

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

        for atom in self.geometry:
            test    =   {atom.atomname: [atom.basisset, "{:6.3f}".format(atom.center[0]), "{:6.3f}".format(atom.center[1]), "{:6.3f}".format(atom.center[2])]}
            self.atomobjects.append(test)
        self.atomobjects.append({"Calculation type": self.calctype})


inputfilename       =   argv[1]
logfilename         =   inputfilename[:-4]+".log"
molecule            =   Geometry(inputfilename)
atomObjects         =   [x for x in molecule.geometry]
basisObjects        =   [y for x in atomObjects for y in x.shells]
nbasisObjects       =   len(basisObjects)
for counter, x in enumerate(basisObjects):
    x.basisindex    =   counter

lastaccess  =   integralEngine(molecule, basisObjects, nbasisObjects, "overlap", True)
lastaccess  =   integralEngine(molecule, basisObjects, nbasisObjects, "kinetic", True, lastaccess)
