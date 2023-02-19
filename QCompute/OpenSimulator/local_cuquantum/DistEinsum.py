#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DistEinsum
"""
import sys

if 'sphinx' not in sys.modules:
    import cupy
    import mpi4py.MPI
    import cuquantum

    MPI_ROOT = 0
    MPI_COMM = mpi4py.MPI.COMM_WORLD
    MPI_RANK, MPI_SIZE = MPI_COMM.Get_rank(), MPI_COMM.Get_size()
    CUDA_DEVICE_ID = MPI_RANK % cupy.cuda.runtime.getDeviceCount()  # Assign the device for each process.
    CUDA_COMPUTE_TYPE = None


# print(f'{RANK=}, {SIZE=}, {DEVICE_ID=}')


def DistEinsum(expr, *operands):
    """
    Distributed version of cuquantum-einsum.
    It is implemented by mpi.
    """

    # Broadcast the operand data.
    operands = MPI_COMM.bcast(operands, MPI_ROOT)

    # Create network object.
    with cuquantum.Network(expr, *operands,
                           options={'device_id': CUDA_DEVICE_ID, 'compute_type': CUDA_COMPUTE_TYPE}) as network:
        # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
        path, info = network.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': max(16, MPI_SIZE)}})

        # Select the best path from all ranks.
        opt_cost, sender = MPI_COMM.allreduce(sendobj=(info.opt_cost, MPI_RANK), op=mpi4py.MPI.MINLOC)
        # if RANK == ROOT:
        #     print(f'Process {sender} has the path with the lowest FLOP count {opt_cost}.')

        # Broadcast info from the sender to all other ranks.
        info = MPI_COMM.bcast(info, sender)

        # Set path and slices.
        path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})

        # Calculate this process's share of the slices.
        num_slices = info.num_slices
        chunk, extra = num_slices // MPI_SIZE, num_slices % MPI_SIZE
        slice_begin = MPI_RANK * chunk + min(MPI_RANK, extra)
        slice_end = num_slices if MPI_RANK == MPI_SIZE - 1 else (MPI_RANK + 1) * chunk + min(MPI_RANK + 1, extra)
        slices = range(slice_begin, slice_end)

        # print(f'Process {RANK} is processing slice range: {slices}.')

        # Contract the group of slices the process is responsible for.
        result = network.contract(slices=slices)

    # Sum the partial contribution from each process on root.
    result = MPI_COMM.reduce(sendobj=result, op=mpi4py.MPI.SUM, root=MPI_ROOT)

    if MPI_RANK == MPI_ROOT:
        return result
    else:
        return None


def main():
    """
    test for this module (DistEinsum function)
    """

    expr = 'ehl,gj,edhg,bif,d,c,k,iklj,cf,a->ba'
    shapes = [(8, 2, 5), (5, 7), (8, 8, 2, 5), (8, 6, 3), (8,), (6,), (5,), (6, 5, 5, 7), (6, 3), (3,)]
    if MPI_RANK == MPI_ROOT:
        operands = [cupy.random.rand(*shape) for shape in shapes]
    else:
        operands = None

    result = DistEinsum(expr, *operands)

    # Check correctness.
    if MPI_RANK == MPI_ROOT:
        result_np = cupy.einsum(expr, *operands, optimize=True)
        assert cupy.allclose(result, result_np)


if __name__ == '__main__':
    main()
