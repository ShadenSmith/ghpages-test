## ZeRO Stage 1 with reduced communication

As introduced in our paper, [ZeRO: Memory Optimization Towards Training A Trillion Parameter Models](https://arxiv.org/abs/1910.02054), we propose three stages of ZeRO that build on top of one another in order to drastically reduce the memory overhead required to train large deep learning models. Specifically these stages are described the the figure below.
![](/ghpages-test/assets/images/zero_stages.PNG)

In the process of evaluating our proposed techniques we implemented ZeRO Stage 1 (P<sub>os</sub>) that partitions optimizer states across data parallel ranks. However, as discussed in Section 9.1 of our paper, instead of using a [reduce-scatter](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#reducescatter) operation to reduce gradients to the their respective partition owners we instead used an [all-reduce](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce) which increased Stage 1 of ZeRO's communication overhead by 1.5x. This all-reduce happens in DeepSpeed at the end of the backward pass regardless of if you are using ZeRO or not. More details can be seen in [deepspeed/pt/deepspeed_light.py](https://github.com/microsoft/DeepSpeed/blob/90017d3a31beee0ef5421ac08edcd0fa441eea11/deepspeed/pt/deepspeed_light.py#L802-L827), however we have simplified the code below for readability.

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

Each bucket in the above code represents at most 2GB worth of data, this is to ensure we keep a fixed memory overhead w.r.t. our all-reduce operations while still achieving high throughput between nodes during training.

In this initial implementation once the backward pass has completed and all of the model's gradients are averaged we are ready to apply them to our model via an `optimizer.step()`. However, each data parallel rank is only responsible for updating the parameters of a subset of the model. This is because ZeRO pre-partitions the optimizer state across data parallel ranks. This means that, in practice, for a given data parallel rank it is not using the vast majority of the averaged gradients it has received! Specifically, each rank is only updating its model parameters using `1/N` of the averaged gradients that it has in memory. `N` in this case represents the total number of data parallel ranks in our training, which can easily be in the 100s or higher if you are training large sized models like we are.

We recognized the opportunity for removing ZeRO's communication overhead from 1.5x early in the design of ZeRO through the use of a reduce-scatter instead of an all-reduce. We are pleased to say we are now updating our public implementation of ZeRO Stage 1 with reduce-scatter and thus eliminating this previous 1.5x communication overhead.

### ZeRO with reduce-scatter

In order to understand how this works let's first dive into how ZeRO partitions optimizer state across parameters in a model. Let's first consider a small model with only 10 parameters as seen below as a single list, the width of each parameter (p) represents the relative size of the parameter in the model.
![](/ghpages-test/assets/images/zero_params.PNG)

When we apply ZeRO P<sub>os</sub> to this model the parameters above are associated with a specific data parallel rank during the model update phase of training. Let's consider a training job with 4 ranks, the allocation of ranks to parameters would be something like the following. Each rank is responsible for updating the parameters in its partition during the optimizer.step() phase.
![](/ghpages-test/assets/images/zero_params_ranks.PNG)

This means that during the gradient averaging phase of training each rank is only required to receive the gradients for the parameters it is responsible for. In our example above, rank 2 only requires the entire gradients for parameters 4 and 5 and only requires partial gradients for parameters 3 and 6.

ZeRO maintains flat partitions for each data parallel rank, along with all bookkeeping needed to unflatten partitions back to their original parameter sizes. We can then think about are original all-reduce code snippet above now using a reduce-scatter operation instead. Below is a simplified version of the code needed to support reduce-scatter for our 10 parameter example model.

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

However, life is not this simple in practice. In order for this reduce scatter approach to support large models we must implement a version of this that limits the gradient exchanges to no more than 2GB worth of data. This allows us to ensure we keep a fixed memory overhead w.r.t. our communication operations while still achieving high throughput between nodes during training.

We won't go into all the details in this post on how this part was implemented but we urge you to read our code for more details. Let's assume our simple 10 parameter model represents 6GB of data. We will now require 3 separate reduce_scatter invocations to exchange all the gradients in the model. This requires us to partition our ranks in a different way to respect communication boundaries so we can exchange the gradients as they become available. We can see an example partitioning below.

![](/ghpages-test/assets/images/zero_w_comm2.png)

### Results

We have evaluated our reduce-scatter implementation of ZeRO Stage 1 on two different types of hardware. 

| Cluster         | Node Count | GPUs/node | Total GPUs | GPU Memory | Internode Bandwidth |
| --------------- | ---------- | --------- | ---------- | ---------- | ------------------- |
| Azure NC24r3_v3 | 8          | 4         | 32         | 16 GB      | 40 Gbps             |
| DGX-2H          | 8          | 16        | 128        | 32 GB      | 800 Gbps            |

