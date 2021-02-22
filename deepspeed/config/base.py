class ConfigError(Exception):
    """Errors related to DeepSpeed configuration. """
    pass


class ConfigArg:
    def __init__(self, default=None, value=None):
        self.default = default
        if value is not None:
            self.value = value
        else:
            self.value = default

    def is_valid(self):
        return True

    def __repr__(self):
        return str(self.value)


class RequiredArg(ConfigArg):
    def __init__(self):
        super().__init__(default=None)

    def is_valid(self):
        # Ensure the required argument is provided.
        if self.value is None:
            return False
        return super().is_valid()


class SubConfig(ConfigArg):
    def __init__(self, config):
        if not isinstance(config, Config):
            raise TypeError(f'Expecting type Config, got {type(config)}')
        super().__init__(value=config)

    def is_valid(self):
        return self.value.is_valid()


class MetaConfig(type):
    """Metaclass (e.g, class factory) for :class:`Config`.

    This is used to extract the argument class attributes and stash them in
    `cls._class_args`.
    """
    def __new__(cls, name, bases, dct):
        config_args = dict()

        # Extract configs from the class dictionary and move them to _class_args
        for key, val in dct.items():
            if isinstance(val, ConfigArg):
                config_args[key] = val
        for key in config_args.keys():
            del dct[key]
        dct['_class_args'] = config_args

        return super().__new__(cls, name, bases, dct)


class Config(metaclass=MetaConfig):
    """Base class for DeepSpeed configurations.

    ``Config`` is a struct with subclassing. They are initialized from dictionaries
    and thus also keyword arguments:

    >>> c = Config(verbose=True)
    >>> c.verbose
    True
    >>> c['verbose']
    True

    You can initialize them from dictionaries:

    >>> myconf = {'verbose' : True}
    >>> c = Config.from_dict(myconf)
    >>> c.verbose
    True

    Configurations should be subclassed to group arguments by topic.
    """
    def __init__(self, **kwargs):
        super().__init__()

        # The config arguments we are tracking. Maps name -> ConfigArg
        self._args = dict()

        # Initialize config structure and defaults. _class_args is a dict of the args
        # in the class definition.
        for key, val in self._class_args.items():
            self._set_arg(key, val)

        # First grab defaults
        # Overwrite any non-defaults specified
        for key, val in kwargs.items():
            self._set_arg(key, ConfigArg(value=val))

    def _set_arg(self, key, value):
        # Is this a fresh arg?
        if isinstance(value, ConfigArg):
            self._args[key] = value
        else:
            # Update the value
            self._args[key].value = value

    def __setattr__(self, name, value):
        # We may be at the start of __init__ before these are set
        args = self.__dict__.get('_args')

        # Updating an argument?
        if (args is not None) and (name in args):
            self._set_arg(name, value)
            return

        # base case
        super().__setattr__(name, value)

    def __getattr__(self, name):
        args = self.__dict__.get('_args')
        if (args is not None) and (name in args):
            return args[name].value

        raise AttributeError(
            f'{self.__class__.__name__} does not have attribute "{name}"')

    def __getitem__(self, name):
        return getattr(self, name)

    def resolve(self):
        """Infer any missing arguments, if possible.

        This is useful for configs such as :class:`BatchConfig` in only a
        subset of arguments are required to complete a valid config.
        """

        # Walk the tree of subconfigs and also resolve().
        for arg in self._args:
            if isinstance(arg, SubConfig):
                arg.resolve()

    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as fin:
            config_dict = json.load(fin)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def is_valid(self):
        """Resolve any missing configurations and determine in the configuration is valid.

        Returns:
            bool: Whether the config and all sub-configs are valid.
        """
        self.resolve()
        return all(arg.is_valid() for arg in self._args.values())

    def __str__(self):
        return self.dot_str()

    def dot_str(self, depth=0, dots_width=50):
        indent_width = 4
        indent = ' ' * indent_width
        lines = []
        lines.append(f'{indent * depth}{self.__class__.__name__} = {{')

        for key, val in self._args.items():
            # Recursive configurations
            if isinstance(val, SubConfig):
                config = val.value
                lines.append(config.dot_str(depth=depth + 1))
                continue

            dots = '.' * (dots_width - len(key) - (depth * indent_width))
            lines.append(f'{indent * (depth+1)}{key} {dots} {val}')
        lines.append(f'{indent * depth}}}')
        return '\n'.join(lines)
