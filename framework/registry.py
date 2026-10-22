"""Auto-discovery registry for framework components.

Components are discovered by scanning their respective directories
for modules containing classes that inherit from the appropriate base class.
"""
import os, sys, importlib, inspect
from typing import List, Dict, Type

from .core import Attack, Defense, Dataset, Agent, Scorer


def _discover(directory: str, base_class: Type, package: str) -> Dict[str, object]:
    """Scan directory for modules and register their classes.

    Args:
        directory: Absolute path to the directory to scan.
        base_class: Base class to filter by.
        package: Python package prefix for relative imports.

    Returns:
        Dict mapping name -> instance.
    """
    registry: Dict[str, object] = {}

    if not os.path.isdir(directory):
        return registry

    # Ensure directory has __init__.py
    # init_path = os.path.join(directory, '__init__.py')
    # (already created during setup)

    for fname in sorted(os.listdir(directory)):
        if fname.startswith('_') or fname.startswith('.'):
            continue
        if not fname.endswith('.py'):
            continue

        module_name = fname[:-3]  # strip .py
        full_module = f"{package}.{module_name}"

        try:
            mod = importlib.import_module(full_module)
        except Exception as e:
            print(f"  [WARN] Failed to import {full_module}: {e}")
            continue

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if not inspect.isclass(attr):
                continue
            if not issubclass(attr, base_class):
                continue
            if attr is base_class:
                continue
            if getattr(attr, 'name', ''):
                try:
                    instance = attr()
                    registry[instance.name] = instance
                except TypeError:
                    # Class needs constructor args — instantiate lazily
                    registry[attr.name] = attr
            elif attr_name.lower() == module_name.lower():
                try:
                    instance = attr()
                    if getattr(instance, 'name', ''):
                        registry[instance.name] = instance
                except TypeError:
                    pass

    return registry


# Paths relative to project root
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_attacks() -> Dict[str, Attack]:
    """Discover all attack methods in the attacks/ directory."""
    return _discover(
        os.path.join(_BASE, 'attacks'), Attack, 'attacks'
    )


def get_defenses() -> Dict[str, Defense]:
    """Discover all defense methods in the defenses/ directory."""
    return _discover(
        os.path.join(_BASE, 'defenses'), Defense, 'defenses'
    )


def get_datasets() -> Dict[str, Dataset]:
    """Discover all datasets in the datasets/ directory."""
    return _discover(
        os.path.join(_BASE, 'datasets'), Dataset, 'datasets'
    )


def get_agents() -> Dict[str, Type[Agent]]:
    """Discover all agent implementations in the agents/ directory.

    Returns classes (not instances) because agents need constructor args.
    """
    return _discover(
        os.path.join(_BASE, 'agents'), Agent, 'agents'
    )
