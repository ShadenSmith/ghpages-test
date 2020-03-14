---
layout: single
title: "Getting Started"
permalink: /getting-started/
categories: tutorials
---

## Installation

* Please see our [Azure tutorial](docs/azure.md) to get started with DeepSpeed on Azure!
* If you're not on Azure, we recommend using our docker image via `docker pull deepspeed/deepspeed:latest` which contains a pre-installed version of DeepSpeed and all the necessary dependencies.
* If you want to install DeepSpeed manually, we provide an install script [install.sh](install.sh) to help install on a local machine or across an entire cluster.

## Writing DeepSpeed Models
DeepSpeed model training is accomplished using the DeepSpeed engine. The engine
can wrap any arbitrary model of type `torch.nn.module` and has a minimal set of APIs
for training and checkpointing the model. Please see the tutorials for detailed
examples.

To initialize the DeepSpeed engine:
```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
```

`deepspeed.inialize` ensures that all of the necessary setup required for
distributed data parallel or mixed precision training are done
appropriately under the hood.  In addition to wrapping the model, DeepSpeed can
construct and manage the training optimizer, data loader, and the learning rate
scheduler based on the parameters passed to `deepspeed.initialze` and the
DeepSpeed [configuration file](#deepspeed-configuration).


### Training

Once the DeepSpeed engine has been initialized, it can be used to train the
model using three simple APIs for forward propagation (`()`), backward
propagation (`backward`), and weight updates (`step`).

```python
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()
```


Under the hood, DeepSpeed automatically performs the necessary operations
required for distributed data parallel training, in mixed precision, with a
pre-defined learning rate schedule:

* **Gradient Averaging**: in distributed data parallel training, `backward`
  ensures that gradients are averaged across data parallel processes after
  training on an `train_batch_size`.

* **Loss Scaling**: in FP16/mixed precision training, the DeepSpeed
  engine automatically handles scaling the loss to avoid precision loss in the
  gradients.

* **Learning Rate Schedule**: if using DeepSpeed's learning rate
  schedule, then DeepSpeed automatically handles any updates to the learning
  rate when `step` is executed.



### Model Checkpointing
Saving and loading the training state is handled via the `save_checkpoint` and
`load_checkpoint` API in DeepSpeed which takes two arguments to uniquely
identify a checkpoint:
  * `ckpt_dir`: the directory where checkpoints will be saved.
  * `ckpt_id`: an identifier that uniquely identifies a checkpoint in the directory.
    In the following code snippet, we use the loss value as the checkpoint identifier.

```python
#load checkpoint
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

#advance data loader to ckpt step
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):

    #forward() method
    loss = model_engine(batch)

    #runs backpropagation
    model_engine.backward(loss)

    #weight update
    model_engine.step()

    #save checkpoint
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
```

DeepSpeed can automatically save and restore the model, optimizer, and the
learning rate scheduler states while hiding away these details from the user.
However, the user may want to save other data in addition to these that are
unique to a given model training. To support these items, `save_checkpoint`
accepts a client state dictionary `client_sd` for saving. These items can be
retrieved from `load_checkpoint` as a return argument. In the example above,
the `step` value is stored as part of the `client_sd`.


## DeepSpeed Configuration
DeepSpeed features can be enabled, disabled, or configured using a config JSON
file that should be specified as `args.deepspeed_config`. A sample config file
is shown below. For a full set of features see [core API
doc](https://microsoft.github.io/DeepSpeed/docs/htmlfiles/api/full/index.html).

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}
```

## Multi-Node Environment Variables

When training across multiple nodes we have found it useful to support
propagating user-defined environment variables. By default DeepSpeed will
propagate all NCCL and PYTHON related environment variables that are set. If
you would like to propagate additional variables you can specify them in a
dot-file named `.deepspeed_env` that contains a new-line separated list of
`VAR=VAL` entries. The DeepSpeed launcher will look in the local path you are
executing from and also in your home directory (`~/`).

As a concrete example, some clusters require special NCCL variables to set
prior to training. The user can simply add these variables to a
`.deepspeed_env` file in their home directory that looks like this:
```
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```
DeepSpeed will then make sure that these environment variables are set when
launching each process on every node across their training job.

# Launching DeepSpeed Training
DeepSpeed installs the entry point `deepspeed` to launch distributed training.
We illustrate an example usage of DeepSpeed with the following assumptions:

1. You have already integrated DeepSpeed into your model
2. `client_entry.py` is the entry script for your model
3. `client args` is the `argparse` command line arguments
4. `ds_config.json` is the configuration file for DeepSpeed


## Resource Configuration (multi-node)
DeepSpeed configures multi-node compute resources with hostfiles that are compatible with
[OpenMPI](https://www.open-mpi.org/) and [Horovod](https://github.com/horovod/horovod).
A hostfile is a list of *hostnames* (or SSH aliases), which are machines accessible via passwordless
SSH, and *slot counts*, which specify the number of GPUs available on the system. For
example,
```
worker-1 slots=4
worker-2 slots=4
```
specifies that two machines named *worker-1* and *worker-2* each have four GPUs to use
for training.

Hostfiles are specified with the `--hostfile` command line option. If no hostfile is
specified, DeepSpeed searches for `/job/hostfile`. If no hostfile is specified or found,
DeepSpeed queries the number of GPUs on the local machine to discover the number of local
slots available.


The following command launches a PyTorch training job across all available nodes and GPUs
specified in `myhostfile`:
```bash
deepspeed <client_entry.py> <client args> \
  --deepspeed --deepspeed_config ds_config.json --hostfile=myhostfile
```

Alternatively, DeepSpeed allows you to restrict distributed training of your model to a
subset of the available nodes and GPUs. This feature is enabled through two command line
arguments: `--num_nodes` and `--num_gpus`. For example, distributed training can be
restricted to use only two nodes with the following command:
```bash
deepspeed --num_nodes=2 \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```
You can instead include or exclude specific resources using the `--include` and
`--exclude` flags. For example, to use all available resources **except** GPU 0 on node
*worker-2* and GPUs 0 and 1 on *worker-3*:
```bash
deepspeed --exclude="worker-2:0@worker-3:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```
Similarly, you can use **only** GPUs 0 and 1 on *worker-2*:
```bash
deepspeed --include="worker-2:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```

### MPI Compatibility
As described above, DeepSpeed provides its own parallel launcher to help launch
multi-node/multi-gpu training jobs. If you prefer to launch your training job
using MPI (e.g., mpirun), we provide support for this. It should be noted that
DeepSpeed will still use the torch distributed NCCL backend and *not* the MPI
backend. To launch your training job with mpirun + DeepSpeed you simply pass us
an additional flag `--deepspeed_mpi`. DeepSpeed will then use
[mpi4py](https://pypi.org/project/mpi4py/) to discover the MPI environment (e.g.,
rank, world size) and properly initialize torch distributed for training. In this
case you will explicitly invoke `python` to launch your model script instead of using
the `deepspeed` launcher, here is an example:
```bash
mpirun <mpi-args> python \
	<client_entry.py> <client args> \
	--deepspeed_mpi --deepspeed --deepspeed_config ds_config.json
```

If you want to use this feature of DeepSpeed, please ensure that mpi4py is
installed via `pip install mpi4py`.

## Resource Configuration (single-node)
In the case that we are only running on a single node (with one or more GPUs)
DeepSpeed *does not* require a hostfile as described above. If a hostfile is
not detected or passed in then DeepSpeed will query the number of GPUs on the
local machine to discover the number of slots available. The `--include` and
`--exclude` arguments work as normal, but the user should specify 'localhost'
as the hostname.


