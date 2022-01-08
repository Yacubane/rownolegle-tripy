#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import itertools
import time
import sys
import debug

start_time = time.time()

G = 6.67e-11
EPS = 1e-50

stars_count = int(sys.argv[1])
is_debug = len(sys.argv) > 2

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

rank_stars_range = ((rank * stars_count) // size, ((rank+1) * stars_count) // size)
rank_stars_count = rank_stars_range[1] - rank_stars_range[0]
rank_stars = np.random.rand(rank_stars_count, 4).astype(np.float32)

if is_debug:
    rank_stars = debug.get_debug_stars(stars_count, rank, size)

other_ranks_stars_send = np.zeros(shape=(rank_stars_count, 4), dtype=np.float32)
other_ranks_stars_recv = np.zeros(shape=(rank_stars_count, 4), dtype=np.float32)

rank_accelerations = np.zeros(shape=(rank_stars_count, 3), dtype=np.float32)

def calculate_acceleration(star1, star2):
    position1 = star1[0:3]
    position2 = star2[0:3]
    mass2 = star2[3]
    m_div_r = mass2 / (np.linalg.norm(position1-position2) ** 3 + EPS)
    return tuple(map(lambda d: G * m_div_r * (d[1] - d[0]), zip(position1, position2)))

other_ranks_stars_send[:rank_stars_count,:] = rank_stars.copy()
for _ in range(size - 1):
    comm.Isend([other_ranks_stars_send, MPI.FLOAT], dest=(rank + 1) % size)
    comm.Recv([other_ranks_stars_recv, MPI.FLOAT], source=(rank - 1) % size)
    for i, j in itertools.product(range(rank_stars_count), range(rank_stars_count)):
        rank_accelerations[i, :] += calculate_acceleration(rank_stars[i, :], other_ranks_stars_recv[j, :])
    other_ranks_stars_send = other_ranks_stars_recv

for i, j in itertools.product(range(rank_stars_count), range(rank_stars_count)):
    if i == j:
        continue
    rank_accelerations[i,:] += calculate_acceleration(rank_stars[i, :], rank_stars[j, :])

if rank != 0:
    comm.Isend([rank_accelerations, MPI.FLOAT], dest=0)
else:
    accelerations = np.empty((stars_count, 3), dtype=np.float32)
    accelerations[rank_stars_range[0]:rank_stars_range[1], :] = rank_accelerations

    received_accelerations = np.empty((rank_stars_count, 3), dtype=np.float32)
    for i in range(1, size):
        received_stars_range = ((i * stars_count) // size, ((i+1) * stars_count) // size)
        received_stars_count = received_stars_range[1] - received_stars_range[0]
        comm.Recv([received_accelerations[:received_stars_count, :], MPI.FLOAT], source=i)
        accelerations[received_stars_range[0]:received_stars_range[1],:] = received_accelerations[:received_stars_count, :]

    end_time = time.time()
    if is_debug:
        print(accelerations)
    print(end_time-start_time)