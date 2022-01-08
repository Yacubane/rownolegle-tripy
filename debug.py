import numpy as np

def get_debug_stars(stars_count, rank, size):
    if stars_count == 4:
        if rank == 0 and size == 1:
            return np.asarray([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]).astype(np.float32)
        elif rank == 0:
            return np.asarray([[1,1,1,1],[2,2,2,2]]).astype(np.float32)
        elif rank == 1:
            return np.asarray([[3,3,3,3],[4,4,4,4]]).astype(np.float32)
    if stars_count == 6:
        if rank == 0 and size == 1:
            return np.asarray([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6]]).astype(np.float32)
        elif rank == 0:
            return np.asarray([[1,1,1,1],[2,2,2,2]]).astype(np.float32)
        elif rank == 1:
            return np.asarray([[3,3,3,3],[4,4,4,4]]).astype(np.float32)
        elif rank == 2:
            return np.asarray([[5,5,5,5],[6,6,6,6]]).astype(np.float32)