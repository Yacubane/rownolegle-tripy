#!/bin/bash -l
mpirun -np 1 --oversubscribe ./task1.py 4 debug
mpirun -np 2 --oversubscribe ./task1.py 4 debug
mpirun -np 1 --oversubscribe ./task2.py 4 debug
mpirun -np 2 --oversubscribe ./task2.py 4 debug

mpirun -np 1 --oversubscribe ./task1.py 6 debug
mpirun -np 3 --oversubscribe ./task1.py 6 debug
mpirun -np 1 --oversubscribe ./task2.py 6 debug
mpirun -np 3 --oversubscribe ./task2.py 6 debug