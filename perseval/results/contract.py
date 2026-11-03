"""Independent offline implementation of the authoritative v1 byte profile.

Schema bytes are vendored separately; no results-package import is used.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


class ProtocolError(ValueError):
    pass


def timestamp(value: str):
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})', value):
        raise ProtocolError('timestamp must include seconds and timezone in the v1 RFC3339 profile')
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ProtocolError(f'invalid timestamp: {value}') from exc


def canonical_json(value) -> bytes:
    """Normative v1 profile: Python JSON shortest numbers, sorted keys, UTF-8, LF."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def digest(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(data: bytes):
    try:
        value = json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ProtocolError(f'nonfinite number: {value}')))
        if canonical_json(value) != data:
            raise ProtocolError('JSON must use the canonical v1 byte profile')
        return value
    except (UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolError(f'invalid JSON: {exc}') from exc


def canonical_path(name: str) -> str:
    p = PurePosixPath(name)
    if (not name or '\\' in name or p.is_absolute() or str(p) != name
            or any(part in ('.', '..') or not part or not part[0].isascii() or not part[0].isalnum() for part in name.split('/'))
            or any(not (c.isascii() and (c.isalnum() or c in '._-/')) for c in name)):
        raise ProtocolError(f'noncanonical path: {name!r}')
    return name


def checksum_bytes(files: dict[str, bytes]) -> bytes:
    return ''.join(f'{digest(files[name])[7:]}  {name}\n' for name in sorted(files)).encode('ascii')


class Contract:
    def __init__(self):
        directory = Path(__file__).parent / 'schemas' / '1.0'
        self.schemas = {}
        resources = []
        files = {p.name: p.read_bytes() for p in directory.glob('*.schema.json')}
        inventory = checksum_bytes(files)
        if inventory != (directory / 'SHA256SUMS').read_bytes():
            raise ProtocolError('vendored schema checksum mismatch (schema drift). Restore exact schema 1.0 files and SHA256SUMS from the reviewed PersEval-experimental-results contract; do not edit vendored schemas individually or fetch them during export.')
        self.schema_digest = digest(inventory)
        self.vendor_source = json.loads((directory.parent / 'vendor-source.json').read_bytes())
        if self.vendor_source['schema_digest'] != self.schema_digest:
            raise ProtocolError('vendored schema digest differs from source record; restore the matching schemas, SHA256SUMS and vendor-source.json together from the reviewed contract release.')
        for raw in files.values():
            schema = json.loads(raw)
            Draft202012Validator.check_schema(schema)
            self.schemas[schema['$id']] = schema
            resources.append((schema['$id'], Resource.from_contents(schema)))
        self.registry = Registry().with_resources(resources)

    def validate(self, name, value):
        if isinstance(value,dict) and value.get('schema_version','1.0') != '1.0':
            raise ProtocolError(f'unsupported bundle schema version {value["schema_version"]!r}; this exporter supports 1.0. Use a compatible, reviewed offline contract release.')
        uri = f'https://perseval.org/schemas/result-bundle/1.0/{name}.schema.json'
        validator = Draft202012Validator(self.schemas[uri], registry=self.registry, format_checker=FormatChecker())
        errors = list(validator.iter_errors(value))
        if errors:
            error = errors[0]
            raise ProtocolError(f'{name}: {list(error.absolute_path)}: {error.message}')


def configuration_id(configuration) -> str:
    return digest(canonical_json(configuration))
