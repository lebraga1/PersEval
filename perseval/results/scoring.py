"""Validate coverage first, then use the unchanged public Evaluator mathematics."""
from __future__ import annotations

import contextlib
import csv
import io

from .context import evaluator_split, plain
from .contract import ProtocolError

COLUMNS = ['user_id','text_id','predictions']

def predictions_bytes(source, context):
    import pandas as pd
    if isinstance(source,pd.DataFrame):
        if len(source.columns) != 3 or set(source.columns) != set(COLUMNS):
            raise ProtocolError('predictions must have exactly user_id,text_id,predictions columns')
        if source.isna().any().any():
            raise ProtocolError('predictions contain missing values')
        rows = [[str(value) for value in row] for row in source[COLUMNS].itertuples(index=False,name=None)]
    else:
        from pathlib import Path
        try:
            rows = list(csv.reader(io.StringIO(Path(source).read_text(encoding='utf-8-sig'),newline=''),strict=True))
        except (csv.Error,UnicodeError) as exc:
            raise ProtocolError(f'invalid prediction CSV: {exc}') from exc
        header = rows.pop(0) if rows else []
        if len(header) != 3 or set(header) != set(COLUMNS):
            raise ProtocolError('predictions must have exactly user_id,text_id,predictions columns')
        if any(len(row) != 3 for row in rows):
            raise ProtocolError('predictions must contain three nonmissing cells per row')
        positions = [header.index(name) for name in COLUMNS]
        rows = [[row[i] for i in positions] for row in rows]
    if not rows or any(len(row)!=3 or not all(row) or row[2].strip().casefold() in ('nan','null','none','na','n/a') for row in rows):
        raise ProtocolError('predictions must contain three nonmissing cells per row')
    pairs = [(row[0],row[1]) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ProtocolError('duplicate user_id/text_id prediction pair')
    expected = {(row[0],row[1]) for row in context['test_annotations']}
    if set(pairs) != expected:
        raise ProtocolError(f'predictions do not exactly cover the selected test split: {len(expected-set(pairs))} missing, {len(set(pairs)-expected)} unexpected pairs')
    allowed = set(context['label_mapping'].values())
    # -1 is an explicit existing PersEval decoder parse-failure prediction, not a missing cell.
    if any(row[2] not in allowed and row[2] != '-1' for row in rows):
        raise ProtocolError('prediction label is absent from the dataset label mapping')
    rows.sort(key=lambda row:(row[0],row[1]))
    out = io.StringIO(newline='')
    csv.writer(out,lineterminator='\n').writerows([COLUMNS,*rows])
    return out.getvalue().encode('utf-8')


def evaluate(predictions: bytes, context, *, positive_label=None):
    import pandas as pd
    from ..evaluation import Evaluator
    table = pd.read_csv(io.BytesIO(predictions),dtype=str,keep_default_na=False)
    # Existing reports iterate sets; their captured output is frozen after evaluation.
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator = Evaluator(table,evaluator_split(context),context['label'])
        global_ = evaluator.global_metrics()
        user = evaluator.annotator_level_metrics()
        text = evaluator.text_level_metrics()
        traits = evaluator.trait_level_metrics()
    compact = {'global_f1':float(global_['macro avg']['f1-score']),
               'user_f1':float(user['macro avg']['f1-score']),
               'text_f1':float(text['macro avg']['f1-score'])}
    codes = context['label_codes']
    if len(codes) == 2:
        positive_label = str(positive_label) if positive_label is not None else ('1' if set(codes)=={'0','1'} else None)
        if positive_label is not None and positive_label not in codes:
            raise ProtocolError('positive_label is not one of the dataset label codes')
        task_score = {'kind':'positive-class-f1', 'positive_label':positive_label,
                      'value':float(global_.get(positive_label,{}).get('f1-score',0.0)) if positive_label is not None else None}
    else:
        task_score = {'kind':'macro-f1', 'positive_label':None,'value':compact['global_f1']}
    metrics = {'schema_version':'1.0','scoring_rule':'perseval.macro-f1.v1',
               'prediction_count':len(table),'compact':compact}
    reports = plain({'format':'perseval-evaluation-v1','scoring_rule':metrics['scoring_rule'],
        'global':global_,'user_average':user,'text_average':text,'trait_average':traits,
        'per_user':evaluator.annotator_level_metrics_dic,'per_text':evaluator.text_level_metrics_dic,
        'per_trait':evaluator.trait_level_metrics_dic,'task_score':task_score})
    return metrics,reports
