from argparse import ArgumentParser, Namespace
from pathlib import Path

class Args:
    def __init__(self):
        self._data = {}

    def __setattr__(self, key, value):
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'Args' object has no attribute '{key}'")

    def as_dict(self):
        """Return the stored arguments as a dictionary."""
        return self._data

    def get_parser(self):
        """Generate an ArgumentParser with stored values as defaults."""
        parser = ArgumentParser(description="Argument parser for the script")
        for key, value in self._data.items():
            arg_type = type(value)
            if isinstance(value, bool):  # Special handling for boolean flags
                parser.add_argument(f"--{key}", action="store_true" if not value else "store_false")
            else:
                parser.add_argument(f"--{key}", type=arg_type, default=value)
        return parser

    @classmethod
    def from_namespace(cls, namespace: Namespace):
        """Convert a parsed Namespace back into an Args object."""
        obj = cls()
        for key, value in vars(namespace).items():
            obj._data[key] = value
        return obj

    def __add__(self, other):
        """Merges two Args objects, creating a shared dictionary."""
        if not isinstance(other, Args):
            raise TypeError("Can only add Args objects together.")

        new_args = Args()
        new_args._data = {**self._data, **other._data}  # Shared dictionary
        return new_args