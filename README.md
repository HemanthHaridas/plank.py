# plank.py
A fully parallel HF code to calculate orbital energies for a molecule. plank is written in Python using MPI4Py for parallel constructs and Cython. 
The code avoids the usage of recursive routines for the calculation of overlap, kinetic integrals and electron-nuclear interaction integrals, with iterative routines adapted from refrences [1] and [2] used instead. Electron-electron repulsion integrals are implemented using a recursive MD scheme. Currently plank only supports calculations on closed-shell singlet molecules.

### Dependencies
1. Python 3.5+
2. mpi4py (for efficient parallelization)
3. numpy
4. math
5. yaml (for writing log files and reading Periodic Table)
6. Cython (for static compilation of libraries)

### Installation
Follow the steps given below for installation
1. conda create -n plankhf && conda activate plankhf
2. conda install numpy cython mpi4py setuptools pyyaml
3. run python setup.py build_ext install --user

Since plank is written to be run in parallel mode, It is expected that openmpi exectubales (mpiexec or mpirun) is available on the machine. plank can be spawned as **mpiexec -np [number of processors] python planksrc.py [input file]**. Logfiles are generated automatically if a verbose option is provided in the input file.
```
mpiexec -np <procs> python plankmain.py <inputfile>
```
Users must obtain the required basis sets from [Basis Set Exhange](https://www.basissetexchange.org/). Please ensure that the format of the downloaded basis set is Gaussian (.gbs extension).
     
### Sample input file
Format for the input file is as follows:
```
[calculation type] [basis set]
[charge] [multiplicity]
[number of atoms]
[atom name] [coordinates]
[max number of SCF iterations] [verbose] (set 0 for low and 1 for high)
```
The user can choose to leave the last line blank, which would result in max number of SCF iterations = 100 and verbose = 0, meaning no log file would be generated.

Sample input calculates the energy of a water molecule at sto-3g level of theory.

```
energy	sto-3g
0	1
3
H                     0.866811829        0.601435779   0.000000
H                    -0.866811829        0.601435779   0.000000
O                     0.000000000       -0.075791844   0.000000
```
Running this input must generate four files overlap.txt, kinetic.txt, nuclear.txt and electron.txt. The contents of the files must resemble the ones that have been copied below.

**overlap.txt**
```
     1.000     0.182     0.038     0.386     0.268     0.210     0.000
     0.182     1.000     0.038     0.386    -0.268     0.210     0.000
     0.038     0.038     1.000     0.237     0.000    -0.000     0.000
     0.386     0.386     0.237     1.000     0.000     0.000     0.000
     0.268    -0.268     0.000     0.000     1.000     0.000     0.000
     0.210     0.210    -0.000     0.000     0.000     1.000     0.000
     0.000     0.000     0.000     0.000     0.000     0.000     1.000
```

**kinetic.txt**
```
     0.760    -0.004    -0.008     0.071     0.147     0.115     0.000
    -0.004     0.760    -0.008     0.071    -0.147     0.115     0.000
    -0.008    -0.008    29.003    -0.168     0.000     0.000     0.000
     0.071     0.071    -0.168     0.808     0.000     0.000     0.000
     0.147    -0.147     0.000     0.000     2.529     0.000     0.000
     0.115     0.115     0.000     0.000     0.000     2.529     0.000
     0.000     0.000     0.000     0.000     0.000     0.000     2.529
```

### References
1. https://www.mathematica-journal.com/2012/02/16/evaluation-of-gaussian-molecular-integrals/
2. Cook, David B. Handbook of computational quantum chemistry. Courier Corporation, 2005.
3. https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.540030314
4. https://joshuagoings.com/2017/04/28/integrals/
