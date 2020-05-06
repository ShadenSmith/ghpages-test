## ZeRO Stage 1 with reduced communication

We are happy to report an update to DeepSpeed's implementation of ZeRO that
reduces the communication volume of ZeRO-powered data parallelism by 1/3. Now
ZeRO-powered data parallelism incurs the same communication as classic data
parallel while achieving memory savings up to 4x. To try it out, you only
need to update to the latest version of DeepSpeed and turn on the [ZeRO
DeepSpeed configuration
flag](https://www.deepspeed.ai/docs/config-json/#fp16-training-options).


### Background

As introduced in our paper, [ZeRO: Memory Optimization Towards Training A
Trillion Parameter Models](https://arxiv.org/abs/1910.02054), we propose a set
of techniques to drastically reduce the memory overhead required to train large
deep learning models. For more details please refer to our previous [blog
post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
or our paper (referenced above).

DeepSpeed's current implementation of ZeRO is stage 1 (ZeRO-OS), which
partitions optimization states, as shown in the figure below. While classic
data parallelism replicates optimizer states across devices, ZeRO-OS
_partitions_ this state across devices, thus introducing significant memory
savings. Max memory saving is 4x compared with classic data parallelism. To
find out more details, please refer to our paper referenced above. The key
point of this post is to discuss the communication pattern and volume required
by ZeRO's partitioned optimizer states.

![](../../../assets/images/zero_stages.PNG)


### Communication of previous ZeRO-OS implementation

The top half of the figure below shows the communication pattern and volume for
classic data parallelism when training a single mini-batch. Classic data
parallelism uses a ring
[all-reduce](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce)
collective to gather and average gradients across all data-parallel processes.
The ring all-reduce algorithm incurs a communication volume of 2N, where N are
the number of data-parallel processes used in training.

The previous version of ZeRO-OS performs the first step discussed above but
also adds an additional step seen in the bottom half of the figure in order to
collect the updated model parameters after the gradients have been applied.
This is done with an
[all-gather](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allgather)
operation which inccurs an additional communication volume of N. This results
in ZeRO-OS's overall communication volume of 3N.

For more details on communication analysis of ring all-reduce we encourage you
to read the following post, [Visual intuition on ring-Allreduce for distributed
Deep
Learning](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da).

![](../../../assets/images/zero_comm_overhead.png)

### Improved communication of ZeRO-OS with reduce-scatter

In order to reduce communication, we can replace ZeRO-OS's all-reduce seen
above with a
[reduce-scatter](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#reducescatter).
The key idea is that each data-parallel process only requires the gradients
corresponding to its parameter partition instead of all the gradients, which
can be achieved with reduce-scatter, resulting in a communication volume of N.
The figure below shows our updated overall communication volume is now 2N (N
for reduce-scatter + N for all-gather).

In comparison, classic data-parallelism incurs one all-reduce per minibatch,
with total communication of 2N.  Our updated reduce-scatter implementation of
ZeRO-OS has the same communication volume while obtaining significant memory
savings!

![](../../../assets/images/zero_rs_comm_overhead.png)

### Emperical results

We have evaluated our reduce-scatter implementation of ZeRO-OS on two
different types of hardware and compared the relative gradient communication
time with our original all-reduce implementation. The amount of communication
time reduction is relative to the bandwidth available in your cluster. For
example, we see a more dramatic impact of reduce-scatter on lower speed
interconnects.

| Cluster         | Node Count | GPUs/node | Total GPUs | GPU Memory | Internode Bandwidth | Reduction in comm time |
| --------------- | ---------- | --------- | ---------- | ---------- | ------------------- |----------------------- |
| Azure NC24r3_v3 | 2          | 4         | 8          | 16 GB      | 40 Gbps             | 1.98x                  |
| DGX-2H          | 8          | 16        | 128        | 32 GB      | 800 Gbps            | 0.50x                  |

###  Implementation details

To try out our updated version of ZeRO-OS you simply need to update to the
latest version of DeepSpeed and turn on ZeRO. If you are interested in some of
the internals, here is how we implemented it.

In the process of evaluating our proposed techniques we implemented ZeRO Stage
1 (P<sub>os</sub>) that partitions optimizer states across data parallel ranks.
However, as discussed in Section 9.1 of our paper, instead of using a
[reduce-scatter](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#reducescatter)
operation to reduce gradients to the their respective partition owners we
instead used an
[all-reduce](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce)
which increased Stage 1 of ZeRO's overall communication overhead by 1.5x. This
all-reduce happens in DeepSpeed at the end of the backward pass regardless of
if you are using ZeRO or not (note: DeepSpeed without ZeRO does not inccur this
1.5x overhead). More details can be seen in
[deepspeed/pt/deepspeed_light.py](https://github.com/microsoft/DeepSpeed/blob/90017d3a31beee0ef5421ac08edcd0fa441eea11/deepspeed/pt/deepspeed_light.py#L802-L827),
however we have simplified the code below for readability.

```python
def allreduce_bucket(self, bucket):
    # Flatten a bucket of tensors into a single tensor
    tensor_to_allreduce = flatten(bucket)
    
    # Perform all_reduce on the data parallel process group
    dist.all_reduce(tensor_to_allreduce, 
                    op=ReduceOp.SUM, 
                    group=self.data_parallel_group)

    # Average gradients w.r.t. data parallel world size 
    tensor_to_allreduce.mul_(1.0 / self.dp_world_size)
    
    return tensor_to_allreduce
```

Each bucket in the above code is capped at a certain size threshold (e.g., 2
GB), this is to ensure we keep a fixed memory overhead w.r.t. our all-reduce
operations while still achieving high throughput between nodes during training.

In this initial implementation once the backward pass has completed and all of
the model's gradients are averaged we are ready to apply them to our model via
an optimizer step. However, each data parallel rank is only responsible for
updating the parameters of a subset of the model. This is because ZeRO
pre-partitions the optimizer state across data parallel ranks. This means that,
in practice, for a given data parallel rank it is not using the vast majority
of the averaged gradients it has received! Specifically, each rank is only
updating its model parameters using `1/N` of the averaged gradients that it has
in memory. `N` in this case represents the total number of data parallel ranks
in our training, which can easily be in the 100s or higher if you are training
large sized models like we are.

We recognized the opportunity for removing ZeRO's 1.5x communication overhead
early in the design of ZeRO through the use of a reduce-scatter instead of an
all-reduce. We are pleased to say we are now updating our public implementation
of ZeRO Stage 1 with reduce-scatter and thus eliminating this previous 1.5x
communication overhead.

### ZeRO with reduce-scatter

In order to understand how this works let's first dive into how ZeRO partitions
optimizer state across parameters in a model. Let's first consider a small
model with only 10 parameters as seen below as a single list, the width of each
parameter (p) represents the relative size of the parameter in the model.

![](../../../assets/images/zero_params.PNG)

When we apply ZeRO P<sub>os</sub> to this model the parameters above are
associated with a specific data parallel rank during the model update phase of
training. Let's consider a training job with 4 ranks, the allocation of ranks
to parameters would be something like the following. Each rank is responsible
for updating the parameters in its partition during the optimizer step phase.

![](../../../assets/images/zero_params_ranks.PNG)

This means that during the gradient averaging phase of training each rank is
only required to receive the gradients for the parameters it is responsible
for. In our example above, rank 2 only requires the entire gradients for
parameters 4 and 5 and only partial gradients for parameters 3 and 6.

ZeRO maintains flat partitions for each data parallel rank, along with all
bookkeeping needed to unflatten partitions back to their original parameter
sizes. We can then think about are original all-reduce code snippet above now
using a reduce-scatter operation instead. Below is a simplified version of the
code needed to support reduce-scatter for our 10 parameter example model.

```python
def reduce_scatter_gradients(self):
    # Fetch list of flat gradient partitions across all data 
    # parallel ranks, each index of the list corresponds to a 
    # different rank.
    flat_partitions = self.get_flat_partitions()
    
    # Accumulate output of reduce scatter is our local rank's 
    # partition, send our local copy of all partitions.
    dist.reduce_scatter(output=flat_partitions[rank],
                        input_list=flat_partitions,
                        op=ReduceOp.SUM,
                        group=self.dp_process_group)

    # Divide by data parallel world size thus averaging gradients
    flat_partitions[rank].mul_(1.0 / self.dp_world_size)
    
    return flat_partitions
```

However, life is not this simple in practice. In order for this reduce scatter
approach to support large models we must implement a version of this that
limits the gradient exchanges to no more than a pre-defined size threshold (e.g.,
2 GB worth of data). This allows us to ensure we keep a fixed memory overhead
w.r.t. our communication operations while still achieving high throughput
between nodes during training.

We won't go into all the details in this post on how this part was implemented
but we urge you to read our code for more details. Let's assume our simple 10
parameter model represents 6 GB of data and our comminication threshold is 2
GB. We will now require 3 separate reduce\_scatter invocations to exchange all
the gradients in the model. This requires us to partition our ranks in a
different way to respect communication boundaries so we can exchange the
gradients as they become available during training. Below we see an example of
a partitioning that respects our communication thresholding.

![](../../../assets/images/zero_w_comm.PNG)


