# test/dynamo/test_dicts.py

import collections
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import unittest
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import run_tests


# ---------------------------------------------------------------------------
# Helper subclasses
# ---------------------------------------------------------------------------


class MyDict(dict):
    pass


class MyDictChild(MyDict):
    # Multi-level subclass — no override, should still be traced
    pass


class MyDictWithOverride(dict):
    @classmethod
    def fromkeys(cls, iterable, value=None):
        # Overrides fromkeys — should trigger a graph break
        result = super().fromkeys(iterable, value)
        result["_custom"] = True
        return result


class MyOrderedDict(collections.OrderedDict):
    pass


class MyDefaultDict(collections.defaultdict):
    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFromkeys(torch._dynamo.test_case.TestCase):
    # -- dict subclass, basic happy path -------------------------------------

    def test_dict_subclass_fromkeys_basic(self):
        def fn(x):
            d = MyDict.fromkeys(["a", "b", "c"], x)
            return d, 1 if isinstance(d, MyDict) else 0

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.tensor(1.0)
        result, is_md = opt(x)
        self.assertEqual(is_md, 1)
        self.assertIsInstance(result, MyDict)

    def test_dict_subclass_fromkeys_returns_correct_type(self):
        def fn(x):
            return MyDict.fromkeys(["a", "b"], x)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.tensor(1.0)
        result = opt(x)
        self.assertIsInstance(result, MyDict)
        self.assertIsInstance(result, dict)

    # -- dict subclass, empty key iterable ----------------------------------

    def test_dict_subclass_fromkeys_empty(self):
        def fn(x):
            d = MyDict.fromkeys([], x)
            return d, 1 if isinstance(d, MyDict) else 0

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.tensor(1.0)
        result, is_md = opt(x)
        self.assertEqual(is_md, 1)
        self.assertEqual(result, {})

    # -- dict subclass, None value (default) --------------------------------
    # (None is the default; calling fromkeys with one arg)

    def test_dict_subclass_fromkeys_default_value(self):
        def fn():
            return MyDict.fromkeys(["x", "y"])

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt()
        self.assertIsInstance(result, MyDict)
        self.assertIsNone(result["x"])
        self.assertIsNone(result["y"])

    # -- multi-level subclass (MyDictChild -> MyDict -> dict) ---------------

    def test_dict_subclass_fromkeys_multilevel(self):
        def fn(x):
            return MyDictChild.fromkeys(["a", "b"], x)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.tensor(2.0)
        result = opt(x)
        self.assertIsInstance(result, MyDictChild)
        self.assertEqual(list(result.keys()), ["a", "b"])

    # -- override triggers graph break -------------------------------------

    def test_dict_subclass_fromkeys_override_graph_break(self):
        def fn(x):
            d = MyDictWithOverride.fromkeys(["a", "b"], x)
            return d["a"]

        cnt = CompileCounter()
        opt = torch.compile(fn, backend=cnt)
        x = torch.tensor(3.0)
        # Correctness must still hold even with a graph break
        self.assertEqual(fn(x), opt(x))
        # A graph break means more than one compiled frame
        self.assertGreater(cnt.frame_count, 1)

    # -- OrderedDict subclass -----------------------------------------------

    def test_ordered_dict_subclass_fromkeys_basic(self):
        def fn(x):
            return MyOrderedDict.fromkeys(["p", "q"], x)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.tensor(1.0)
        result = opt(x)
        self.assertIsInstance(result, MyOrderedDict)
        self.assertIsInstance(result, collections.OrderedDict)

    def test_ordered_dict_subclass_fromkeys_empty(self):
        def fn(x):
            return MyOrderedDict.fromkeys([], x)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt(torch.tensor(0.0))
        self.assertEqual(result, collections.OrderedDict())

    # -- defaultdict subclass -----------------------------------------------

    def test_defaultdict_subclass_fromkeys_basic(self):
        def fn(x):
            return MyDefaultDict.fromkeys(["m", "n"], x)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.tensor(5.0)
        result = opt(x)
        self.assertIsInstance(result, MyDefaultDict)
        self.assertIsInstance(result, collections.defaultdict)

    @unittest.expectedFailure
    def test_defaultdict_subclass_fromkeys_empty(self):
        def fn(x):
            return MyDefaultDict.fromkeys([], x)

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt(torch.tensor(0.0))
        self.assertEqual(dict(result), {})

    def test_defaultdict_subclass_fromkeys_preserves_factory(self):
        # fromkeys on defaultdict does NOT set default_factory; it stays None.
        # This test documents that behaviour is preserved after compilation.
        def fn(x):
            d = MyDefaultDict.fromkeys(["a"], x)
            return d

        opt = torch.compile(fn, backend="eager", fullgraph=True)
        result = opt(torch.tensor(1.0))
        # default_factory is None because fromkeys doesn't accept one
        self.assertIsNone(result.default_factory)


if __name__ == "__main__":
    run_tests()
