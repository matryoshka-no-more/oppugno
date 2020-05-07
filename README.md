# oppugno
Empowering Python with CUDA-tongue

## How to give a try
### Environment setup
```bash
# use python3 as the default python
virtualenv env -p `which python3`
source env/bin/activate
pip install -r requirements.txt
```
### Try CUDA benchmark with subprocess
```bash
# the benchmark is done via repeatative saxpy,
# which is a normal saxpy function with an additional O(n) iternation in the inner loop

source env/bin/activate
cd src/test_subprocess

# run the benchmark for repeatative saxpy
python test_subprocess.py
```
### Try CUDA benchmark with cython
```bash
# the benchmark is done via matrix multiplication
source env/bin/activate
cd src/test_cython

# compile the CUDA files via cython and install it for Python usage as a shared object
python setup.py install

# run the benchmark for matrix multiplication
python test_cython.py
```

## What we did for parallelism
- wrote benchmark tests for saxpy, repeatative saxpy (additional `O(n)`), and matrix multiplication
- wrote 2 versions of CUDA interfacing implementation (subprocess, cython)
- tested wide range of data sizes for respective performance changes
- summarized observation of the benefits, limitation, and when to use the oppugno for CUDA speedup

## What we did for transcompilation from Python to CUDA
- analyse the abstract syntax tree
- categorize arguments to groups (dims, input arrays, output arrays, extra params)
- render CUDA source code with argument groups in jinja2
- check existing CUDA shared object for execution before recompiling one

## What we plan for future
- translate the statements from the inner for loop to kernels
- render kernel functions with jinja2
- rewrite the original python function to substitute the for loops with the cython entry function for CUDA
