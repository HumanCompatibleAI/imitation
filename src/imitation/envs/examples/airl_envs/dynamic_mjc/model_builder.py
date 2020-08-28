"""model_builder.py A small library for programmatically building MuJoCo XML files."""
import tempfile
from contextlib import contextmanager

import numpy as np


def default_model(name):
    """Get a model with basic settings such as gravity and RK4 integration."""
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true")
    default = root.default()
    default.joint(armature=1, damping=1, limited="true")
    default.geom(contype=0, friction="1 0.1 0.1", rgba="0.7 0.7 0 1")
    root.option(gravity="0 0 -9.81", integrator="RK4", timestep=0.01)
    return model


def pointmass_model(name):
    """Get a model with basic settings such as gravity and Euler integration."""
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true", coordinate="local")
    default = root.default()
    default.joint(limited="false", damping=1)
    default.geom(
        contype=2,
        conaffinity="1",
        condim="1",
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
    )
    root.option(timestep=0.01, gravity="0 0 0", iterations="20", integrator="Euler")
    return model


class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr("model", name)

    @contextmanager
    def asfile(self):
        """Usage:
        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model
        """
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".xml", delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def find_attr(self, attr, value):
        return self.root.find_attr(attr, value)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = " ".join([str(val).lower() for val in value])
        else:
            value = str(value).lower()

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode = MJCTreeNode(name)
            for (k, v) in kwargs.items():
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode

        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def find_attr(self, attr, value):
        """Run DFS to find a matching attr."""
        if attr in self.attrs and self.attrs[attr] == value:
            return self
        for child in self.children:
            res = child.find_attr(attr, value)
            if res is not None:
                return res
        return None

    def write(self, ostream, tabs=0):
        contents = " ".join(['%s="%s"' % (k, v) for (k, v) in self.attrs.items()])
        if self.children:
            ostream.write("\t" * tabs)
            ostream.write("<%s %s>\n" % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs + 1)
            ostream.write("\t" * tabs)
            ostream.write("</%s>\n" % self.name)
        else:
            ostream.write("\t" * tabs)
            ostream.write("<%s %s/>\n" % (self.name, contents))

    def __str__(self):
        s = "<" + self.name
        s += " ".join(['%s="%s"' % (k, v) for (k, v) in self.attrs.items()])
        return s + ">"
