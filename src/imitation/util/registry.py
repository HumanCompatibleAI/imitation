import importlib


def load_attr(name):
  """Load an attribute in format path.to.module:attribute."""
  module_name, attr_name = name.split(":")
  module = importlib.import_module(module_name)
  attr = getattr(module, attr_name)
  return attr
