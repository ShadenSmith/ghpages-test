import torch
import torch.distributed as dist
import time
import argparse
import os

TRIALS = 5

N = 500000
M = 40

def timed_allreduce(mat):
    pre = time.perf_counter()
    dist.all_reduce(mat)
    print('ignore me', mat[0][0])  # required due to lazy evaluation
    duration = time.perf_counter() - pre
    print("duration: %f sec" % duration)
    tput = ((M*N*4*2)/duration)*8
    print("algo throughput: %f bps, %f Gbps" % (tput, tput/1e9))
    size = M * N * 4
    n = dist.get_world_size()
    busbw = (size / duration) * (2 * (n - 1) / n) * 8
    print("busbw: %f Gbps" % (busbw / 1e9))

def run(local_rank):
    global_rank = dist.get_rank()
    print(global_rank, "data size:", M*N*4/1e9, "GB")
    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)

    for _ in range(TRIALS):
        timed_allreduce(mat)

def init_processes(local_rank, fn, backend='nccl'):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend)
    fn(local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    rank = args.local_rank
    print("local_rank: %d" % rank)
    init_processes(local_rank=rank, fn=run)
