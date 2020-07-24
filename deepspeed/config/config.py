
import json

class Config:
    """Base class for DeepSpeed configurations.

    ``Config`` is a struct with subclassing. They are initialized from dictionaries
    and thus also keyword arguments:

    >>> c = Config(verbose=True)
    >>> c.verbose
    True

    You can initialize them from dictionaries:

    >>> myconf = {'verbose' : True}
    >>> c = Config.from_dict(myconf)
    >>> c.verbose
    True

    Configurations should be subclassed to group by topic.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # First grab defaults
        print()
        for key, val in vars(self.__class__).items():
            print(f'SETTING key={key} val={val}')
            #setattr(self, key, val)

        # Overwrite anything specified
        for key, val in kwargs.items():
            # recursive update
            if isinstance(val, dict):
                pass
            setattr(self, key, val)
        
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as fin:
            config_dict = json.load(fin)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
    
    def is_valid(self):
        return super().is_valid()
    
    def __str__(self):
        return self.dot_str()

    def dot_str(self, depth=0, dots_width=50):
        indent_width = 4
        indent = ' ' * indent_width
        lines = []
        lines.append(f'{indent * depth}{self.__class__.__name__} = {{')

        for key in vars(self.__class__):
            if key.startswith('_'):
                continue
            val = getattr(self, key)

            # Recursive configurations
            if isinstance(val, Config):
                lines.append(val.dot_str(depth=depth+1))
                continue

            dots = '.' * (dots_width - len(key) - (depth * indent_width))
            lines.append(f'{indent * (depth+1)}{key} {dots} {val}')
        lines.append(f'{indent * depth}}}')
        return '\n'.join(lines)


class BatchConfig(Config):
    """ Batch size related parameters. """

    train_batch_size = 1
    """ The effective training batch size.
    
    This is the amount of data samples that leads to one step of model
    update. ``train_batch_size`` is aggregated by the batch size that a
    single GPU processes in one forward/backward pass (a.k.a.,
    ``train_step_batch_size``), the gradient accumulation steps (a.k.a.,
    ``gradient_accumulation_steps``), and the number of GPUs.

    .. IMPORTANT::
        ``train_batch_size`` is a required configuration.
    """

    train_micro_batch_size_per_gpu = 1
    """Batch size to be processed per device each forward/backward step.

    When specified, ``gradient_accumulation_steps`` is automatically
    calculated using ``train_batch_size`` and the number of devices. Should
    not be concurrently specified with ``gradient_accumulation_steps``.
    """

    gradient_accumulation_steps = 1
    """ Number of training steps to accumulate gradients before averaging and
    applying them.
    
    This feature is sometimes useful to improve scalability
    since it results in less frequent communication of gradients between
    steps. Another impact of this feature is the ability to train with larger
    batch sizes per GPU. When specified, ``train_step_batch_size`` is
    automatically calculated using ``train_batch_size`` and number of GPUs.
    Should not be concurrently specified with ``train_step_batch_size``.
    """


class FP16Config(Config):
    """ FP16 configuration. """

    #: Enable/disable FP16
    enabled = False

    #: gradient clipping
    clip = 1.0


class TrainingConfig(Config):
    """Top-level configuration for all aspects of training with DeepSpeed.

    >>> from deepspeed.config import TrainingConfig
    >>> myconfig = TrainingConfig()
    >>> myconfig
    TrainingConfig = {
        BatchConfig = {
            train_batch_size .............................. 1
            train_micro_batch_size_per_gpu ................ 1
            gradient_accumulation_steps ................... 1
        }
        ...
        FP16Config = {
            enabled ....................................... False
            clip .......................................... 1.0
        }
    }
    >>> myconfig.fp16
    FP16Config = {
        enabled ........................................... False
        clip .............................................. 1.0
    }
    >>> myconfig.fp16.enabled
    False
    """

    fp16 = FP16Config()
    batch = BatchConfig()

    #def __init__(self):
    #    #: FP16 configuration
    #    self._fp16 = FP16Config()
    #    self._batch = BatchConfig()

    #@property
    #def batch(self):
    #    """See :class:`.BatchConfig`"""
    #    return self._batch
    
    #@property
    #def fp16(self):
    #    """See :class:`.FP16Config`"""
    #    return self._fp16
    

def _compare(config, base):
    for key, val in base.items():
        assert getattr(config, key) == val

def test_base():
    c = Config(name='jeff')
    assert c.name == 'jeff'

def test_dict():
    d = {
        'name' : 'tygra',
        'color' : 'orange'
    }
    c = Config(**d)
    _compare(c, d)

    c = Config.from_dict(d)
    _compare(c, d)

def test_training():
    c = TrainingConfig()


if __name__ == '__main__':
    test_base()
    test_dict()

    test_training()

    c = BatchConfig(gradient_accumulation_steps=3)
    d = BatchConfig(gradient_accumulation_steps=6)
    print(c)
    print()
    print(d)

    c = TrainingConfig()
    print(c)
    print(vars(c))
    print(vars(TrainingConfig))
