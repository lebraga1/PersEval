"""Freeze the existing dataset/split API into offline, JSON-only evaluation context."""
from __future__ import annotations

import copy
import hashlib
import math
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

from .. import config
from .contract import ProtocolError, canonical_json, digest
from .io import atomic_write


def plain(value):
    """Convert numpy scalars and dataset missing values to finite JSON values."""
    if isinstance(value, dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return sorted((plain(v) for v in value), key=lambda v:canonical_json(v))
    if hasattr(value, 'item'):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ProtocolError(f'cannot record non-JSON metadata value of type {type(value).__name__}')


def code(value):
    value = plain(value)
    if value is None or isinstance(value, (dict, list, bool)) or str(value) == '':
        raise ProtocolError('dataset IDs and labels must be nonempty scalar values')
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def identifier(value):
    value = plain(value)
    if value is None or isinstance(value,(dict,list,bool)) or str(value) == '':
        raise ProtocolError('dataset IDs must be nonempty scalar values')
    return str(value)


def retain_selection(dataset):
    """Keep the task/split selected when a model was built, without copying datasets."""
    selected = copy.copy(dataset)
    selected.labels = copy.deepcopy(dataset.labels)
    if hasattr(dataset, '_result_split_definition'):
        selected._result_split_definition = copy.deepcopy(dataset._result_split_definition)
    return selected


def capture_split(function):
    """Record parameters actually used, without changing the split algorithm/return."""
    @wraps(function)
    def wrapped(self, extended, user_adaptation, named, baseline=False):
        definition = {'protocol':'perseval.get_splits.v1',
                      'parameters':copy.deepcopy(config.dataset_specific_splits[self.name]),
                      'seed':config.seed, 'baseline':baseline}
        result = function(self, extended=extended, user_adaptation=user_adaptation, named=named, baseline=baseline)
        self._result_split_definition = definition
        return result
    return wrapped


def validate_settings(task, adaptation, extended):
    if task not in ('baseline','named','unnamed') or adaptation not in ('none','train','test') or not isinstance(extended,bool):
        raise ProtocolError('invalid task/adaptation/extended setting')
    if (task == 'baseline' and adaptation != 'none') or (task == 'unnamed' and adaptation == 'none'):
        raise ProtocolError('baseline requires no adaptation; unnamed requires train or test adaptation')


def _annotations(split, label):
    result = []
    for row in split.annotation:
        if label not in row['label']:
            raise ProtocolError(f'label {label!r} is missing from a split annotation')
        result.append([identifier(row['user']),identifier(row['text']),code(row['label'][label])])
    result.sort(key=lambda row:(row[0],row[1]))
    pairs = [(row[0],row[1]) for row in result]
    if len(pairs) != len(set(pairs)):
        raise ProtocolError('dataset test/split IDs are duplicated or collide after string normalization')
    return result


def resolved_dataset_fingerprint(dataset):
    source = getattr(dataset, 'dataset', None)
    if source is None:
        return None
    hasher = hashlib.sha256()
    hasher.update(b'perseval.resolved-dataset.rows.v1\n')
    # Original resolved row order is part of dataset identity. No sampling or downloads.
    for row in source:
        hasher.update(canonical_json(plain(row)))
    return 'sha256:' + hasher.hexdigest()


def make_context(dataset, label=None):
    label = label or dataset.label
    name = dataset.name
    if name not in config.dataset_label:
        raise ProtocolError(f'unknown PersEval dataset {name!r}')
    if label not in dataset.available_labels():
        raise ProtocolError(f'unknown label {label!r}; available: {dataset.available_labels()}')
    if not isinstance(dataset.named,bool) or not isinstance(dataset.extended,bool) or dataset.user_adaptation not in (False,'train','test'):
        raise ProtocolError('select a task with dataset.get_splits(...) before recording results')
    adaptation = 'none' if dataset.user_adaptation is False else dataset.user_adaptation
    task = 'named' if dataset.named else ('baseline' if adaptation == 'none' else 'unnamed')
    validate_settings(task,adaptation,dataset.extended)
    annotations = _annotations(dataset.test_set,label)
    if not annotations:
        raise ProtocolError('selected test split is empty')
    definition = getattr(dataset,'_result_split_definition',None)
    split = {'protocol':definition['protocol'] if definition else 'perseval.local-snapshot.v1',
             'parameters':copy.deepcopy(definition['parameters']) if definition else {},
             'seed':definition['seed'] if definition else None,
             'test_fingerprint':digest(canonical_json(annotations))}
    users = {}
    for uid,user in dataset.test_set.users.items():
        sid = identifier(uid)
        if sid in users:
            raise ProtocolError('dataset user IDs collide after string normalization')
        users[sid] = plain(user.traits)
    if not {row[0] for row in annotations} <= users.keys():
        raise ProtocolError('test annotations reference users absent from the dataset')
    labels = sorted({code(value) for value in dataset.labels[label]} | {row[2] for row in annotations})
    configured = config.label_map.get(label, config.label_map.get(label+'_pred'))
    if configured:
        labels = sorted(set(labels) | {code(value) for value in configured.values()})
    mapping = {str(key):code(value) for key,value in configured.items()} if configured else {value:value for value in labels}
    if not set(labels) <= set(mapping.values()):
        mapping.update({value:value for value in labels if value not in mapping.values()})
    partition_hashes = {}
    for key in ('training_set','adaptation_set','test_set'):
        source = getattr(dataset,key,None)
        partition_hashes[key] = digest(canonical_json(_annotations(source,label))) if source is not None else None
    context = {'format':'perseval-local-dataset-v1',
        'dataset':{'name':name,'revision':getattr(dataset,'result_dataset_revision',None),'fingerprint':resolved_dataset_fingerprint(dataset)},
        'label':label, 'available_labels':dataset.available_labels(), 'label_codes':labels, 'label_mapping':mapping,
        'settings':{'task':task,'adaptation':adaptation,'extended':dataset.extended}, 'split':split,
        'test_annotations':annotations,'test_users':users,
        'provenance':{'dataset_fingerprint_algorithm':'perseval.resolved-dataset.rows.v1; finite JSON rows in resolved source order, missing/nonfinite normalized to null',
                      'test_fingerprint_algorithm':'sha256(canonical JSON sorted [user_id,text_id,gold_label_code] triples)',
                      'source_cache_fingerprint':getattr(getattr(dataset,'dataset',None),'_fingerprint',None),
                      'partition_fingerprints':partition_hashes,
                      'split_definition_recorded':definition is not None}}
    validate_context(context)
    return context


def validate_context(context):
    required = {'format','dataset','label','available_labels','label_codes','label_mapping','settings','split','test_annotations','test_users','provenance'}
    if not isinstance(context,dict) or set(context) != required or context['format'] != 'perseval-local-dataset-v1':
        raise ProtocolError('invalid local dataset context; use dataset.save_result_context(path)')
    if context['dataset']['name'] not in config.dataset_label or context['label'] not in context['available_labels']:
        raise ProtocolError('unknown dataset or label in local dataset context')
    validate_settings(**context['settings'])
    rows = context['test_annotations']
    if not rows or any(not isinstance(row,list) or len(row)!=3 or not all(isinstance(v,str) and v for v in row) for row in rows):
        raise ProtocolError('invalid test annotations in local dataset context')
    pairs = [(row[0],row[1]) for row in rows]
    if pairs != sorted(set(pairs)):
        raise ProtocolError('test annotation pairs must be unique and sorted')
    if digest(canonical_json(rows)) != context['split']['test_fingerprint']:
        raise ProtocolError('test fingerprint differs from local test annotations')
    if not {row[2] for row in rows} <= set(context['label_codes']) or not set(context['label_codes']) <= set(context['label_mapping'].values()):
        raise ProtocolError('test labels and label mapping disagree')
    if not {row[0] for row in rows} <= context['test_users'].keys():
        raise ProtocolError('test users are missing from local context')
    for traits in context['test_users'].values():
        if not isinstance(traits,dict) or any(not isinstance(v,list) or not v for v in traits.values()):
            raise ProtocolError('test traits must map trait names to nonempty value lists')
    return context


def save_context(dataset, path, *, label=None):
    path = Path(path).expanduser()
    atomic_write(path,canonical_json(make_context(dataset,label)))
    return path


def evaluator_split(context):
    label = context['label']
    annotations = [{'user':u,'text':t,'label':{label:y}} for u,t,y in context['test_annotations']]
    users = {uid:SimpleNamespace(id=uid,traits=traits) for uid,traits in context['test_users'].items()}
    return SimpleNamespace(annotation=annotations,users=users)
