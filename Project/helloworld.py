#!/usr/bin/env python
"""
Parallel Hello World
"""

from mpi4py import MPI
import sys


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

print('my rank is:', rank)

data = (rank)
data = comm.gather(data, root=0)
if rank == 0:
    print(data)
