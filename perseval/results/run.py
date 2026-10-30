"""A mutable local draft, finalized only by an explicit portable export."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from .. import config
from .bundle import MAX_ARCHIVE, pack_files, seal_payload, validate_files
from .context import make_context, plain, validate_context
from .contract import Contract, ProtocolError, canonical_json, configuration_id, digest, read_json, timestamp
from .environment import capture_environment, now, software_provenance
from .io import atomic_write, local_lock, selected_artifacts
from .scoring import evaluate, predictions_bytes

REPORT_PATH = 'artifacts/perseval-evaluation.json'
MUTABLE = {'method','representation','notes','display_metadata','positive_label','model_revision','training','inference','benchmark_version','seed'}


class LocalRun:
    """Local execution record. No upload/submission or results-package dependency."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if self.path.is_symlink() or not (self.path / 'descriptor.json').is_file():
            raise ProtocolError(f'not a local run directory: {self.path}')
        descriptor = self._descriptor()
        if descriptor.get('format') != 'perseval-local-run-v1' or descriptor.get('run_id') != self.path.name:
            raise ProtocolError('local run identity does not match its directory')

    @property
    def run_id(self):
        return self._descriptor()['run_id']

    @property
    def finalized(self):
        return (self.path / 'finalized').exists()

    def _descriptor(self):
        return read_json((self.path / 'descriptor.json').read_bytes())

    @classmethod
    def record(cls, dataset, predictions, *, model_identifier, model_family=None,
               method=None, representation=None, seed=None, model_revision=None,
               training=None, inference=None, notes=None, display_metadata=None,
               positive_label=None, benchmark_version='perseval-split-v1',
               started_at=None, finished_at=None, environment=None, provenance=None,
               execution_details=None, run_root=None, label=None):
        context = validate_context(copy.deepcopy(dataset)) if isinstance(dataset,dict) else make_context(dataset,label)
        canonical = predictions_bytes(predictions,context)
        # Explicit parse failures in existing decoder APIs participate in the scoring union.
        if b',-1\n' in canonical:
            context['label_mapping']['__parse_failure__'] = '-1'
        started_at = started_at or now()
        finished_at = finished_at or now()
        if timestamp(finished_at) < timestamp(started_at):
            raise ProtocolError('execution finished before it started')
        environment = copy.deepcopy(environment) if environment is not None else capture_environment()
        provenance = copy.deepcopy(provenance) if provenance is not None else software_provenance()
        root = Path(run_root or (config.output_root / 'runs')).expanduser()
        root.mkdir(parents=True,exist_ok=True)
        rid = 'run_' + str(uuid.uuid4())
        destination = root / rid
        temporary = Path(tempfile.mkdtemp(prefix='.record-',dir=root))
        try:
            snapshots = {'predictions.csv':canonical, 'dataset.json':canonical_json(context),'environment.json':canonical_json(environment)}
            for name,raw in snapshots.items():
                atomic_write(temporary / name,raw)
            import pandas as pd
            source_path = None if isinstance(predictions,pd.DataFrame) else str(Path(predictions).expanduser().resolve())
            source_digest = None if source_path is None else digest(Path(source_path).read_bytes())
            descriptor = plain({'format':'perseval-local-run-v1','run_id':rid,'recorded_at':now(),
                'execution':{'started_at':started_at,'finished_at':finished_at,'command':None,'executor':None},
                'execution_details':execution_details or {},'seed':seed,'model':model_identifier,'model_family':model_family,
                'model_revision':model_revision,'method':method,'representation':representation,
                'training':training or {},'inference':inference or {},'notes':notes,'display_metadata':display_metadata,
                'positive_label':positive_label,'benchmark_version':benchmark_version,'software_provenance':provenance,
                'prediction_path':'predictions.csv','source_prediction_path':source_path,'source_prediction_digest':source_digest,
                'snapshot_digests':{name:digest(raw) for name,raw in snapshots.items()}})
            atomic_write(temporary / 'descriptor.json',canonical_json(descriptor))
            if destination.exists():
                raise ProtocolError('new run ID collides with an existing local run')
            os.replace(temporary,destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return cls(destination)

    def update_metadata(self, **values):
        unknown = set(values) - MUTABLE
        if unknown:
            raise ProtocolError(f'unsupported metadata fields: {sorted(unknown)}')
        with local_lock(self.path):
            if self.finalized:
                raise ProtocolError('finalized run metadata is immutable')
            descriptor = self._descriptor()
            descriptor.update(plain(values))
            atomic_write(self.path / 'descriptor.json',canonical_json(descriptor))
        return self

    def replace_predictions(self, source):
        """Explicitly correct a draft before finalization; never edit the source CSV."""
        with local_lock(self.path):
            if self.finalized:
                raise ProtocolError('finalized predictions are immutable; record a separate execution')
            context = self._context()
            raw = predictions_bytes(source,context)
            descriptor = self._descriptor()
            descriptor['snapshot_digests']['predictions.csv'] = digest(raw)
            atomic_write(self.path / 'predictions.csv',raw)
            atomic_write(self.path / 'descriptor.json',canonical_json(descriptor))
        return self

    def _context(self):
        return validate_context(read_json((self.path / 'dataset.json').read_bytes()))

    def _snapshots(self, descriptor):
        result = {}
        for name, expected in descriptor['snapshot_digests'].items():
            if name not in ('predictions.csv','dataset.json','environment.json'):
                raise ProtocolError('unknown local snapshot file')
            path = self.path / name
            if path.is_symlink():
                raise ProtocolError('local run snapshots must not be symlinks')
            raw = path.read_bytes()
            if digest(raw) != expected:
                raise ProtocolError(f'changed local snapshot {name}; finalized content cannot be reused. For a draft use replace_predictions() to make an explicit correction.')
            result[name] = raw
        if set(result) != {'predictions.csv','dataset.json','environment.json'}:
            raise ProtocolError('local run is missing required snapshots')
        return result

    def _payload(self, descriptor, snapshots):
        context = validate_context(read_json(snapshots['dataset.json']))
        canonical = predictions_bytes(self.path / 'predictions.csv',context)
        if canonical != snapshots['predictions.csv']:
            raise ProtocolError('local predictions must use canonical bytes')
        if b',-1\n' in canonical:
            context['label_mapping']['__parse_failure__'] = '-1'
        missing = [key for key in ('model','method','representation') if not descriptor.get(key)]
        if missing:
            raise ProtocolError(f'run metadata is incomplete: {", ".join(missing)}; use run.update_metadata(...) before export')
        configuration = {'benchmark_version':descriptor['benchmark_version'], 'dataset':context['dataset'],
            'label':context['label'],'split':context['split'],'model':{'name':descriptor['model'],'revision':descriptor['model_revision']},
            **context['settings'],'method':descriptor['method'],'representation':descriptor['representation'],
            'hyperparameters':{'training':descriptor['training'],
                               'inference':{**descriptor['inference'],'model_family':descriptor['model_family']}}}
        run = {'schema_version':'1.0','run_id':descriptor['run_id'],'configuration_id':configuration_id(configuration),
            'configuration':configuration,'seed':descriptor['seed'],'execution':descriptor['execution'],
            'provenance':{**descriptor['software_provenance'],'label_mapping':context['label_mapping'],
                          'split_provenance':context['provenance'],'notes':descriptor['notes']}}
        contract = Contract()
        contract.validate('run',run)
        contract.validate('environment',read_json(snapshots['environment.json']))
        metrics,reports = evaluate(canonical,context,positive_label=descriptor['positive_label'])
        reports['run_metadata'] = {'model_family':descriptor['model_family'],'display_metadata':descriptor['display_metadata'],
                                  'execution_details':descriptor['execution_details']}
        payload = {'run.json':canonical_json(run),'metrics.json':canonical_json(metrics),
                   'predictions.csv':canonical,'environment.json':snapshots['environment.json'],
                   REPORT_PATH:canonical_json(reports)}
        declaration = {'path':REPORT_PATH,'media_type':'application/json',
                       'description':'Complete public Evaluator reports and task score, selected by explicit result export.'}
        return payload, declaration

    def _frozen(self, descriptor, *, artifacts=None):
        directory = self.path / 'finalized'
        if directory.is_symlink():
            raise ProtocolError('finalized run directory must not be a symlink')
        finalization = read_json((directory / 'finalization.json').read_bytes())
        if digest(canonical_json(descriptor)) != finalization['descriptor_digest']:
            raise ProtocolError('changed finalized run descriptor')
        archive = (directory / 'result.tar.gz').read_bytes()
        if digest(archive) != finalization['archive_digest']:
            raise ProtocolError('changed finalized archive')
        # Always validate exact payload snapshots even when replaying a cached archive.
        from .bundle import read_archive
        files = read_archive(directory / 'result.tar.gz')
        docs = validate_files(files,Contract())
        if docs['run']['run_id'] != descriptor['run_id'] or files['predictions.csv'] != (self.path/'predictions.csv').read_bytes():
            raise ProtocolError('finalized payload differs from local run identity/predictions')
        if artifacts is not None:
            extra,_ = selected_artifacts(artifacts)
            expected = {k:v for k,v in files.items() if k.startswith('artifacts/') and k != REPORT_PATH}
            if extra != expected:
                raise ProtocolError('artifact selection is frozen; omit artifacts to replay the original bundle')
        return archive,files,docs

    def export_result(self, output_dir='exports/', *, artifacts=None):
        """Explicitly finalize and export one portable, deterministic result archive."""
        with local_lock(self.path):
            descriptor = self._descriptor()
            snapshots = self._snapshots(descriptor)
            output = Path(output_dir).expanduser()
            if output.resolve().is_relative_to(self.path):
                raise ProtocolError('export directory must be outside the local run record')
            destination = output / f'{descriptor["run_id"]}.perseval-result.tar.gz'
            if destination.is_symlink():
                raise ProtocolError('export destination must not be a symlink')
            if self.finalized:
                archive,_,_ = self._frozen(descriptor,artifacts=artifacts)
                if destination.exists() and destination.read_bytes() != archive:
                    raise ProtocolError(f'export would overwrite different content: {destination}')
                atomic_write(destination,archive)
                return destination
            payload,report = self._payload(descriptor,snapshots)
            extra,declarations = selected_artifacts(artifacts)
            files = seal_payload({**payload,**extra},Contract(),[report,*declarations])
            validate_files(files,Contract())
            archive = pack_files(files)
            if len(archive) > MAX_ARCHIVE:
                raise ProtocolError('compressed result archive exceeds 64 MiB')
            if destination.exists() and destination.read_bytes() != archive:
                raise ProtocolError(f'export would overwrite different content: {destination}')
            # Complete private finalization and portable output atomically, with no partial .tar.gz.
            temporary = Path(tempfile.mkdtemp(prefix='.finalize-',dir=self.path))
            installed = False
            try:
                atomic_write(temporary / 'result.tar.gz',archive)
                atomic_write(temporary / 'finalization.json',canonical_json({
                    'descriptor_digest':digest(canonical_json(descriptor)),
                    'archive_digest':digest(archive),'payload_digest':read_json(files['manifest.json'])['payload_digest']}))
                os.replace(temporary,self.path/'finalized')
                installed = True
                atomic_write(destination,archive)
            except Exception:
                if installed:
                    shutil.rmtree(self.path/'finalized')
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return destination

    def inspect(self, *, artifacts=None):
        """Return a notebook-friendly summary; no archive creation or finalization."""
        descriptor = self._descriptor()
        summary = {'run_id':descriptor['run_id'],'state':'finalized' if self.finalized else 'draft',
                   'metadata':descriptor,'completeness':{'complete':False,'exportable':False,'missing':[],'errors':[]},
                   'predictions':{'path':str(self.path/'predictions.csv')},'metrics':None,
                   'included_artifacts':[], 'estimated_artifact_bytes':None,
                   'schema_source':Contract().vendor_source}
        missing = summary['completeness']['missing']
        try:
            snapshots = self._snapshots(descriptor)
            context = self._context()
            summary['environment'] = read_json(snapshots['environment.json'])
            summary['dataset'] = {k:context[k] for k in ('dataset','label','settings','split')}
            for key,value in {'dataset revision':context['dataset']['revision'],'dataset fingerprint':context['dataset']['fingerprint'],
                              'split seed':context['split']['seed'],'model revision':descriptor['model_revision'],'execution seed':descriptor['seed'],
                              'method':descriptor['method'],'representation':descriptor['representation'],
                              'PersEval Git revision':descriptor['software_provenance']['perseval_git_revision'],
                              **{'environment '+key:summary['environment'][key] for key in ('python','platform','machine','packages')}}.items():
                if value is None:
                    missing.append(key)
            if self.finalized:
                _,files,docs = self._frozen(descriptor,artifacts=artifacts)
            else:
                payload,report = self._payload(descriptor,snapshots)
                extra,declarations = selected_artifacts(artifacts)
                files = seal_payload({**payload,**extra},Contract(),[report,*declarations])
                docs = validate_files(files,Contract())
            summary['predictions'].update(count=docs['metrics']['prediction_count'],digest=digest(files['predictions.csv']))
            summary['metrics'] = docs['metrics']
            summary['task_score'] = read_json(files[REPORT_PATH])['task_score']
            summary['included_artifacts'] = [item['path'] for item in docs['manifest']['artifacts']]
            summary['estimated_artifact_bytes'] = sum(len(data) for name,data in files.items() if name.startswith('artifacts/'))
            summary['completeness']['exportable'] = True
            summary['completeness']['complete'] = not missing
        except (ProtocolError,OSError,KeyError,TypeError) as exc:
            summary['completeness']['errors'].append(str(exc))
        return summary
