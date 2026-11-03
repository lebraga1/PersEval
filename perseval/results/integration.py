"""Non-disruptive adapters for the existing prediction APIs."""
from __future__ import annotations

import warnings

from .context import make_context, plain
from .contract import ProtocolError, canonical_json, digest
from .environment import capture_environment, now, software_provenance
from .run import LocalRun, MUTABLE

TRAINING_FIELDS = (
    'learning_rate','num_train_epochs','max_steps','per_device_train_batch_size',
    'per_device_eval_batch_size','gradient_accumulation_steps','weight_decay',
    'warmup_steps','warmup_ratio','lr_scheduler_type','optim','adam_beta1','adam_beta2',
    'adam_epsilon','max_grad_norm','fp16','bf16','dataloader_drop_last',
    'label_smoothing_factor','load_best_model_at_end','metric_for_best_model',
    'greater_is_better','eval_strategy','save_strategy',
)


def training_parameters(trainer):
    args = getattr(trainer,'args',None)
    values = {}
    for key in TRAINING_FIELDS:
        if args is not None and hasattr(args,key):
            value = getattr(args,key)
            if hasattr(value,'value'):
                value = value.value
            values[key] = plain(value)
    return values


def model_revision(model):
    revision = getattr(getattr(model,'config',None),'_commit_hash',None)
    return revision if isinstance(revision,str) and revision else None


def begin_model_run(model, dataset, *, method, representation, family, seed=None,
                    training=None, inference=None, result_metadata=None, execution_details=None):
    """Snapshot before inference; recording failure must not change legacy prediction output."""
    try:
        metadata = {**getattr(model,'result_metadata',{}),**(result_metadata or {})}
        allowed = MUTABLE | {'model_identifier','model_family','execution_details'}
        if set(metadata)-allowed:
            raise ProtocolError(f'unknown result_metadata fields: {sorted(set(metadata)-allowed)}')
        return {'dataset':make_context(dataset,model.label),'model_identifier':model.model_id,
                'model_family':family,'model_revision':model_revision(getattr(model,'model',None)),
                'method':method,'representation':representation,'seed':seed,
                'training':training or {},'inference':inference or {},
                'started_at':now(),'environment':capture_environment(),'provenance':software_provenance(),
                'execution_details':execution_details or {},**metadata}
    except (ProtocolError,OSError,ValueError,TypeError,KeyError) as exc:
        model.last_run_error = str(exc)
        warnings.warn(f'Prediction continues without a local result record: {exc}', RuntimeWarning,stacklevel=2)
        return None


def finish_model_run(model, context, predictions, *, key=None, metadata=None):
    if context is None:
        return None
    try:
        context = {**context,**(metadata or {}),'finished_at':now()}
        run = LocalRun.record(predictions=predictions,**context)
        model.last_runs[key or 'default'] = run
        model.last_run = run if len(model.last_runs)==1 else None
        return run
    except (ProtocolError,OSError,ValueError,TypeError,KeyError) as exc:
        model.last_run_error = str(exc)
        warnings.warn(f'Predictions were saved, but no publishable local run was recorded: {exc}',RuntimeWarning,stacklevel=2)
        return None


def prompt_digest(prompts):
    return digest(canonical_json(plain(prompts)))


def parsed_label(text, mapping):
    """Normalize LaMP's legacy raw output for a separate logical prediction snapshot."""
    value = text.strip().casefold()
    if '{' in value:
        value = value.rsplit('{',1)[1].split('}',1)[0].strip().strip("'")
    normalized = {key.casefold():str(value) for key,value in mapping.items()}
    return normalized.get(value, '-1')
