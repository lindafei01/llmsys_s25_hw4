from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    # raise NotImplementedError("Schedule Generation Not Implemented Yet")
    num_cycles = num_batches + num_partitions - 1
    for clock in range(num_cycles):
        schedule = []

        for partition in range(min(clock + 1, num_partitions)):
            microbatch = clock - partition

            if microbatch < num_batches:
                schedule.append((microbatch, partition))

        yield schedule
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        # raise NotImplementedError("Pipeline Parallel Not Implemented Yet")
        micro_batches = x.chunk(self.split_size)
        batches = list(micro_batches)

        num_batches = len(batches)
        num_partitions = len(self.partitions)

        for schedule in _clock_cycles(num_batches, num_partitions):
            self.compute(batches, schedule)
        
        last_device = self.devices[-1]
        batches = [batch.to(last_device) for batch in batches]
        return torch.cat(batches)

        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN SOLUTION
        # raise NotImplementedError("Pipeline Parallel Not Implemented Yet")
        for microbatch_idx, partition_idx in schedule:
            partition = partitions[partition_idx]
            
            def compute_fn():
                batch = batches[microbatch_idx]
                batch = batch.to(devices[partition_idx])
                return partition(batch)
            
            task = Task(compute_fn)
            self.in_queues[partition_idx].put(task)
        
        for _ in range(len(schedule)):
            success, result = self.out_queues[partition_idx].get()
            if success:
                task, batch = result
                batches[microbatch_idx] = batch
            else:
                raise RuntimeError("Pipeline computation failed")
        
        # END SOLUTION

