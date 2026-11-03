"""Offline unit-test adapters for the synthetic CPU test suite.

Production APIs still use the real dependencies. These doubles make synthetic tests independent of installed optional runtime versions
and guarantee that no GPU backend or model loader is initialized.
"""
import contextlib
from importlib.machinery import ModuleSpec
import socket
import sys
import types

import pytest


def forbidden(*args, **kwargs):
    raise AssertionError('network/model/dataset loading is forbidden in unit tests')


# Always replace model/dataset runtimes in unit tests, even if installed.
module = types.ModuleType('datasets')
module.__spec__ = ModuleSpec('datasets',loader=None)
module.load_dataset = forbidden
module.load_from_disk = forbidden
module.concatenate_datasets = forbidden
module.Dataset = type('Dataset', (), {'from_pandas':staticmethod(forbidden)})
sys.modules['datasets'] = module

module = types.ModuleType('transformers')
module.__spec__ = ModuleSpec('transformers',loader=None)
loader = type('NoDownloadLoader', (), {'from_pretrained':staticmethod(forbidden)})
for name in ('AutoTokenizer','AutoModelForSequenceClassification','AutoModelForCausalLM','AutoModel'):
    setattr(module,name,type(name,(loader,),{}))
module.Trainer = type('Trainer', (), {})
module.TrainingArguments = types.SimpleNamespace
module.set_seed = lambda seed: None
module.pipeline = forbidden
sys.modules['transformers'] = module

module = types.ModuleType('torch')
module.__spec__ = ModuleSpec('torch',loader=None)
module.__version__ = 'unavailable-test-double'
module.version = types.SimpleNamespace(cuda=None)
module.cuda = types.SimpleNamespace(is_available=lambda:False,get_device_name=lambda index:'synthetic CPU')
module.no_grad = contextlib.nullcontext
module.float16, module.bfloat16 = 'float16','bfloat16'
sys.modules['torch'] = module


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(socket,'create_connection',forbidden)
    for name in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):
        monkeypatch.setenv(name,'1')
