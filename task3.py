#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import itertools
import time
import sys

from numpy.core.fromnumeric import repeat

start_time = time.time()

G = 6.67e-11
EPS = 1e-50

stars_count = int(sys.argv[1])
delta_time = float(sys.argv[2])
iterations = int(sys.argv[3])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

rank_stars_range = ((rank * stars_count) // size, ((rank+1) * stars_count) // size)
rank_stars_count = rank_stars_range[1] - rank_stars_range[0]
rank_stars = np.random.rand(rank_stars_count, 4).astype(np.float32)
rank_stars[:,3]*=1e5
    
def calculate_accelerations(rank_stars):
    other_ranks_stars_send = np.zeros(shape=(rank_stars_count, 4), dtype=np.float32)
    other_ranks_stars_recv = np.zeros(shape=(rank_stars_count, 4), dtype=np.float32)
    rank_accelerations = np.zeros(shape=(rank_stars_count, 3), dtype=np.float32)
    other_ranks_accelerations_send = np.zeros(shape=(rank_stars_count, 3), dtype=np.float32)
    other_ranks_accelerations_recv = np.zeros(shape=(rank_stars_count, 3), dtype=np.float32)

    def calculate_acceleration(star1, star2, calculate_both):
        position1 = star1[0:3]
        position2 = star2[0:3]
        mass2 = star2[3] * 100
        if calculate_both:
            mass1 = star1[3]
            div_r = 1 / (np.linalg.norm(position1-position2) ** 3 + EPS)
            return [tuple(map(lambda d: G * mass * div_r * (d[1] - d[0]), zip(position1, position2))) for mass in [mass2, -mass1]]
        else:
            m_div_r = mass2 / (np.linalg.norm(position1-position2) ** 3 + EPS)
            return tuple(map(lambda d: G * m_div_r * (d[1] - d[0]), zip(position1, position2)))

    other_ranks_stars_send[:rank_stars_count,:] = rank_stars.copy()
    iterations = int(np.ceil(size / 2)) - 1
    for _ in range(iterations):
        comm.Isend([other_ranks_stars_send, MPI.FLOAT], dest=(rank + 1) % size)
        comm.Isend([other_ranks_accelerations_send, MPI.FLOAT], dest=(rank + 1) % size)
        comm.Recv([other_ranks_stars_recv, MPI.FLOAT], source=(rank - 1) % size)
        comm.Recv([other_ranks_accelerations_recv, MPI.FLOAT], source=(rank - 1) % size)
        for i, j in itertools.product(range(rank_stars_count), range(rank_stars_count)):
            star1accel, star2accel = calculate_acceleration(rank_stars[i, :], other_ranks_stars_recv[j, :], True)
            rank_accelerations[i, :] += star1accel
            other_ranks_accelerations_recv[j, :] += star2accel
        
        other_ranks_stars_send = other_ranks_stars_recv
        other_ranks_accelerations_send = other_ranks_accelerations_recv

    if size % 2 == 0:
        comm.Isend([other_ranks_stars_send, MPI.FLOAT], dest=(rank + 1) % size)
        comm.Recv([other_ranks_stars_recv, MPI.FLOAT], source=(rank - 1) % size)
        for i, j in itertools.product(range(rank_stars_count), range(rank_stars_count)):
            rank_accelerations[i,:] += calculate_acceleration(rank_stars[i, :], other_ranks_stars_recv[j, :], False)

    comm.Isend([other_ranks_accelerations_send, MPI.FLOAT], dest=(rank - iterations) % size)
    comm.Recv([other_ranks_accelerations_recv, MPI.FLOAT], source=(rank + iterations) % size)

    for i, j in itertools.product(range(rank_stars_count), range(rank_stars_count)):
        if i == j:
            continue
        rank_accelerations[i,:] += calculate_acceleration(rank_stars[i, :], rank_stars[j, :], False)

    rank_accelerations[:,:] += other_ranks_accelerations_recv[:rank_stars_count, :]
    return rank_accelerations

stars = np.empty((stars_count,4), dtype=np.float32) if rank == 0 else None
velocity = np.array(np.random.rand(rank_stars_count,3), dtype=np.float32) 
velocity -= 0.5
velocity *= 0.1
acceleration = np.zeros(shape=(rank_stars_count, 3), dtype=np.float32)
frames = []
old_velocity, old_acceleration = velocity.copy(), acceleration.copy()
for i in range(iterations):
    acceleration = calculate_accelerations(rank_stars)
    velocity[:,:] = old_velocity +  0.5 * (acceleration + old_acceleration) * delta_time
    rank_stars[:, 0:3] += old_velocity * delta_time +  0.5 * old_acceleration * (delta_time * delta_time)
    old_acceleration, acceleration = acceleration, old_acceleration
    old_velocity, velocity = velocity, old_velocity

    if rank != 0:
        comm.Send([rank_stars, MPI.FLOAT], dest=0)
    else:
        stars[rank_stars_range[0]:rank_stars_range[1],:] = rank_stars
        other_stars = np.empty((rank_stars_count, 4), dtype=np.float32)
        for i in range(1, size): 
            comm.Recv([other_stars, MPI.FLOAT], source=i)
            stars[(i * rank_stars_count):((i+1) * rank_stars_count),:] = other_stars
        frames.append(stars.copy())
frames = np.asarray(frames)


# if rank == 0:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     limit = 1
#     ax.set_xlim3d([-limit, limit])
#     ax.set_xlabel('X')
#     ax.set_ylim3d([-limit, limit])
#     ax.set_ylabel('Y')
#     ax.set_zlim3d([-limit, limit])
#     ax.set_zlabel('Z')

#     def update(num, datas, lines):
#         for i, data in enumerate(datas):
#             lines[i].set_data(data[:2, :num])
#             lines[i].set_3d_properties(data[2, :num])

#     stars_pos_frames = np.moveaxis(frames.T, 1, 0)
#     lines = [ax.plot([], [], [])[0] for _ in range(stars_count)]
#     anim = animation.FuncAnimation(
#                 fig,
#                 update,
#                 len(frames),
#                 fargs=(stars_pos_frames, lines),
#                 interval=1,
#                 blit=False,
#                 repeat=True)

#     plt.show()