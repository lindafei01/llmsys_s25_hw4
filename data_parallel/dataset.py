from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


# ASSIGNMENT 4.1
class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN SOLUTION
        # raise NotImplementedError("Data Parallel Not Implemented Yet")
        return self.data[self.index][index]
        # END SOLUTION

# ASSIGNMENT 4.1
class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN SOLUTION
        # raise NotImplementedError("Data Parallel Not Implemented Yet")
        length  = len(data)
        indices = list(range(length))
        rng.shuffle(indices)

        # calculate the actual sizes in terms of the number of elements
        partition_sizes = [int(length*size) for size in sizes]
        # The last partition gets all the remaining elements to handle rounding
        partition_sizes.append(length - sum(partition_sizes))

        start = 0
        for size in partition_sizes:
            self.partitions.append(indices[start:start+size])
            start = start + size

        # END SOLUTION

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN SOLUTION
        # raise NotImplementedError("Data Parallel Not Implemented Yet")
        return Partition(self.data, self.partitions[partition])
        # END SOLUTION

# ASSIGNMENT 4.1
def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN SOLUTION
    # raise NotImplementedError("Data Parallel Not Implemented Yet")
    partitioned_batch_size = batch_size // rank
    partitioner = DataPartitioner(data=dataset, sizes=[1 / world_size]*world_size)
    current_partition = partitioner.use(rank)
    dataloader = DataLoader(current_partition, collate_fn=collate_fn)
    return dataloader
    # END SOLUTION

