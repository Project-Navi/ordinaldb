"""Keep ``unittest.assertWarns`` safe from lazy alias modules.

``unittest``'s ``_AssertWarnsContext.__enter__`` scans every entry in
``sys.modules`` and evaluates ``getattr(module, "__warningregistry__",
None)`` so it can reset warning registries. ``transformers`` 5.x registers
~90 backward-compatibility alias modules whose module-level ``__getattr__``
forwards *any* attribute lookup to a lazily imported target module; when
that target requires an uninstalled optional backend (``torchvision``), the
forwarding raises ``ModuleNotFoundError`` — which ``getattr`` with a default
does not suppress. The result: after anything transitively imports
``transformers`` (e.g. llama-index's LangChain bridge probe), every
subsequent ``assertWarns``/``assertWarnsRegex`` in the process explodes.

``inoculate_lazy_module_aliases()`` plants a real, empty
``__warningregistry__`` dict on any module that defines a module-level
``__getattr__`` but has no registry yet, so the scan finds a concrete
attribute and never falls through to the lazy forwarder. Planting the empty
dict is exactly the state ``assertWarns`` itself leaves behind (it resets
found registries to ``{}``). Call it from ``setUp`` in test classes that use
``assertWarns`` after framework imports may have happened.
"""

import sys


def inoculate_lazy_module_aliases() -> None:
    """Plant ``__warningregistry__ = {}`` on lazy alias modules (see above)."""
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if namespace is None:
            continue
        if "__getattr__" in namespace and "__warningregistry__" not in namespace:
            namespace["__warningregistry__"] = {}
