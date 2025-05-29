from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(1))

    dataset = get_dataset('mdot')
    return trackers, dataset

def keep_track_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('keep_track', 'default', range(1))

    dataset = get_dataset('mdot')
    return trackers, dataset

def kys_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('kys', 'default', range(1))

    dataset = get_dataset('mdot')
    return trackers, dataset

def faemtrack():
    n = 6 # 通过改变n，可以一次评价多个跟踪器（也就是能让自己的算法在一个执行过程中评价多次）
    trackers = trackerlist('dimp', 'super_dimp', range(n))

    dataset = get_dataset('mdot')
    return trackers, dataset
