# A test on its own
import torch
import pytest
import json
import argparse
from common import distributed_test
from simple_model import SimpleModel, create_config_from_dict, random_dataloader
import torch.distributed as dist

# A test on its own
import deepspeed
from deepspeed.runtime.config import DeepSpeedConfig

from deepspeed.config import *


def test_cuda():
    assert (torch.cuda.is_available())


def test_check_version():
    assert hasattr(deepspeed, "__git_hash__")
    assert hasattr(deepspeed, "__git_branch__")
    assert hasattr(deepspeed, "__version__")
    assert hasattr(deepspeed, "__version_major__")
    assert hasattr(deepspeed, "__version_minor__")
    assert hasattr(deepspeed, "__version_patch__")


def _run_batch_config(ds_config, train_batch=None, micro_batch=None, gas=None):
    ds_config.train_batch_size = train_batch
    ds_config.train_micro_batch_size_per_gpu = micro_batch
    ds_config.gradient_accumulation_steps = gas
    success = True
    try:
        ds_config._configure_train_batch_size()
    except AssertionError:
        success = False
    return success


def _batch_assert(status, ds_config, batch, micro_batch, gas, success):

    if not success:
        assert not status
        print("Failed but All is well")
        return

    assert ds_config.train_batch_size == batch
    assert ds_config.train_micro_batch_size_per_gpu == micro_batch
    assert ds_config.gradient_accumulation_steps == gas
    print("All is well")


#Tests different batch config provided in deepspeed json file
@pytest.mark.parametrize('num_ranks,batch,micro_batch,gas,success',
                         [(2,32,16,1,True),
                         (2,32,8,2,True),
                         (2,33,17,2,False),
                         (2,32,18,1,False)]) # yapf: disable
def test_batch_config(num_ranks, batch, micro_batch, gas, success):
    @distributed_test(world_size=2)
    def _test_batch_config(num_ranks, batch, micro_batch, gas, success):
        assert dist.get_world_size() == num_ranks, \
        'The test assumes a world size of f{num_ranks}'

        ds_batch_config = 'tests/unit/ds_batch_config.json'
        ds_config = DeepSpeedConfig(ds_batch_config)

        #test cases when all parameters are provided
        status = _run_batch_config(ds_config,
                                   train_batch=batch,
                                   micro_batch=micro_batch,
                                   gas=gas)
        _batch_assert(status, ds_config, batch, micro_batch, gas, success)

        #test cases when two out of three parameters are provided
        status = _run_batch_config(ds_config, train_batch=batch, micro_batch=micro_batch)
        _batch_assert(status, ds_config, batch, micro_batch, gas, success)

        if success:
            #when gas is provided with one more parameter
            status = _run_batch_config(ds_config, train_batch=batch, gas=gas)
            _batch_assert(status, ds_config, batch, micro_batch, gas, success)

            status = _run_batch_config(ds_config, micro_batch=micro_batch, gas=gas)
            _batch_assert(status, ds_config, batch, micro_batch, gas, success)

            #test the case when only micro_batch or train_batch is provided
            if gas == 1:
                status = _run_batch_config(ds_config, micro_batch=micro_batch)
                _batch_assert(status, ds_config, batch, micro_batch, gas, success)

                status = _run_batch_config(ds_config, train_batch=batch)
                _batch_assert(status, ds_config, batch, micro_batch, gas, success)
        else:
            #when only gas is provided
            status = _run_batch_config(ds_config, gas=gas)
            _batch_assert(status, ds_config, batch, micro_batch, gas, success)

            #when gas is provided with something else and gas does not divide batch
            if gas != 1:
                status = _run_batch_config(ds_config, train_batch=batch, gas=gas)
                _batch_assert(status, ds_config, batch, micro_batch, gas, success)

    """Run batch config test """
    _test_batch_config(num_ranks, batch, micro_batch, gas, success)


def test_temp_config_json(tmpdir):
    config_dict = {
        "train_batch_size": 1,
    }
    config_path = create_config_from_dict(tmpdir, config_dict)
    config_json = json.load(open(config_path, 'r'))
    assert 'train_batch_size' in config_json


def test_deprecated_deepscale_config(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        }
    }

    config_path = create_config_from_dict(tmpdir, config_dict)
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args='')
    args.deepscale_config = config_path
    args.local_rank = 0

    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_deprecated_deepscale_config(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=5,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_deprecated_deepscale_config(args=args, model=model, hidden_dim=hidden_dim)


def test_dist_init_true(tmpdir):
    config_dict = {
        "train_batch_size": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.00015
            }
        },
        "fp16": {
            "enabled": True
        }
    }

    config_path = create_config_from_dict(tmpdir, config_dict)
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args='')
    args.deepscale_config = config_path
    args.local_rank = 0

    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1])
    def _test_dist_init_true(args, model, hidden_dim):
        model, _, _,_ = deepspeed.initialize(args=args,
                                             model=model,
                                             model_parameters=model.parameters(),
                                             dist_init_required=True)
        data_loader = random_dataloader(model=model,
                                        total_samples=5,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_dist_init_true(args=args, model=model, hidden_dim=hidden_dim)


def test_init_no_optimizer(tmpdir):

    config_dict = {"train_batch_size": 1, "fp16": {"enabled": True}}
    config_path = create_config_from_dict(tmpdir, config_dict)

    @distributed_test(world_size=1)
    def _helper():
        parser = argparse.ArgumentParser()
        args = parser.parse_args(args='')
        args.deepscale_config = config_path
        args.local_rank = 0

        hidden_dim = 10

        model = SimpleModel(hidden_dim=hidden_dim)

        model, _, _, _ = deepspeed.initialize(args=args, model=model)
        data_loader = random_dataloader(model=model,
                                        total_samples=5,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            with pytest.raises(AssertionError):
                model.backward(loss)
            with pytest.raises(AssertionError):
                model.step()

    _helper()


# New Config setup


def _compare(config, base):
    for key, val in base.items():
        assert getattr(config, key) == val


def test_base():
    c = Config(name='jeff')
    assert c.name == 'jeff'
    assert c['name'] == 'jeff'

    # Overwrite
    c.name = 'samyam'
    assert c.name == 'samyam'


def test_dict():
    d = {'name': 'tygra', 'color': 'orange'}
    c = Config(**d)
    _compare(c, d)

    c = Config.from_dict(d)
    _compare(c, d)


def test_multiconfig():
    # This tests that the metaprogramming works.
    b1 = FP16Config(enabled=True)
    b2 = FP16Config(enabled=False)
    assert b1.enabled
    assert not b2.enabled


def test_nested():
    c = TrainingConfig()
    assert c.batch.train_batch_size is None

    c.batch = BatchConfig(train_batch_size=32)
    assert c.batch.train_batch_size == 32


def test_valid():
    b = BatchConfig()
    # XXX should return False instead of raising an Exception?
    # config.validate() might be a good alias for raising the Exception
    with pytest.raises(ConfigError):
        b.is_valid()
