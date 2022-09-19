# Verify NNE
## Coding style
We should try to use as much Python's type annotation
as possible.
## Env for Alpha-Beta-Crown
```
conda env create --name abc --file abc_env.yaml
```
## Build and run Marabou
_NOTE_: If you see the error complaining about -Wunused-but-set-variable or -Wunused-but-set-parameter,
you can fix it searching and removing -Werror in the CMakeList.txt, or add -Wno-error=unused-but-set-variable or -Wno-error=unused-but-set-parameter to the CMakeList.txt

The root issue is that in the original code, warnings are treated as errors. In modern C++ compiler, a new kind of warning is reported: a variable that is set but not used anywhere. The code indeed contains this issue, so the correct fix should be fixing the code. But the simpliest one should be ignoring it. 

You can read more about it here https://gcc.gnu.org/gcc-4.6/porting_to.html 

In short, to build and run Marabou:

```
git clone https://github.com/NeuralNetworkVerification/Marabou.git
###Fix the code so that it can compile with modern g++/clang as noted above
cd Marabou
mkdir build 
cd build
cmake .. -DBUILD_PYTHON=ON
cmake --build .
###Add Maraboupy to the conda env
conda develop ../
```

## Verify ACAS's activation pattern using Marabou
```
python marabou_acas.py
```
To kill it, in a different terminal, run
```
pkill -f marabou_acas.py
```
You may need to run the `pkill` command multiple times.

## Marabou's InputQuery format
```
    unsigned numVars = atoi( input->readLine().trim().ascii() );
    unsigned numLowerBounds = atoi( input->readLine().trim().ascii() );
    unsigned numUpperBounds = atoi( input->readLine().trim().ascii() );
    unsigned numEquations = atoi( input->readLine().trim().ascii() );
    unsigned numConstraints = atoi( input->readLine().trim().ascii() );
```