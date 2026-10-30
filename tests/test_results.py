from __future__ import annotations

import copy
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from perseval import config
from perseval.data import PerspectivistDataset, PerspectivistSplit, User
from perseval.results import LocalRun
from perseval.results.bundle import read_archive, validate_bundle, validate_files
from perseval.results.context import capture_split, make_context, validate_context
from perseval.results.contract import Contract, ProtocolError, canonical_json, checksum_bytes, digest, read_json
from perseval.results.environment import capture_environment
from perseval.results.run import REPORT_PATH

FIXTURES = Path(__file__).parent/'fixtures/result-bundle-1.0'
INDEX = json.loads((FIXTURES/'index.json').read_text())


@pytest.fixture
def dataset():
    ds = PerspectivistDataset()
    ds.name,ds.label='BREXIT','hs'
    ds.named,ds.extended,ds.user_adaptation=False,False,False
    ds.labels={'hs':{0,1}}
    ds.dataset=[{'user':'01','text':'t1','hs':0},{'user':'02','text':'t2','hs':1}]
    ds.result_dataset_revision='synthetic-v1'
    ds._result_split_definition={'protocol':'perseval.get_splits.v1','parameters':{'test_fraction':0.2},'seed':17,'baseline':True}
    ds.training_set=PerspectivistSplit('train')
    ds.adaptation_set=PerspectivistSplit('adaptation')
    ds.test_set=PerspectivistSplit('test')
    ds.test_set.annotation=[{'user':'01','text':'t1','label':{'hs':0}},{'user':'02','text':'t2','label':{'hs':1}}]
    ds.test_set.users={'01':User('01'),'02':User('02')}
    ds.test_set.users['01'].traits={'group':['a']}
    ds.test_set.users['02'].traits={'group':['b']}
    ds.test_set.texts={'t1':{'text':'first'},'t2':{'text':'second'}}
    return ds


@pytest.fixture
def predictions(tmp_path):
    path=tmp_path/'predictions.csv'
    path.write_bytes(b'user_id,text_id,predictions\n02,t2,1\n01,t1,0\n')
    return path


@pytest.fixture
def make_run(tmp_path,dataset,predictions,monkeypatch):
    monkeypatch.setattr(config,'output_root',tmp_path/'outputs')
    def factory(**kwargs):
        return LocalRun.record(kwargs.pop('dataset',dataset),kwargs.pop('predictions',predictions),
            model_identifier='synthetic-model',model_family='encoder',
            method=kwargs.pop('method','synthetic-test'),representation='none',seed=42,
            started_at='2026-01-01T00:00:00Z',finished_at='2026-01-01T00:01:00Z',**kwargs)
    return factory


def exported(run,tmp_path,**kwargs):
    path=run.export_result(tmp_path/'exports',**kwargs)
    files,docs=validate_bundle(path,Contract())
    return path,files,docs


@pytest.mark.parametrize('filename',INDEX['valid_bundles'])
def test_authoritative_valid_vectors(filename):
    validate_bundle(FIXTURES/filename,Contract())


@pytest.mark.parametrize('case',INDEX['invalid_bundles'],ids=lambda case:case['file'])
def test_authoritative_invalid_vectors(case):
    with pytest.raises(ProtocolError,match=case['expected_error']):
        validate_bundle(FIXTURES/case['file'],Contract())


def test_vendored_schema_inventory_and_honest_source_record():
    contract=Contract()
    path=Path(importlib.import_module('perseval.results.contract').__file__).parent/'schemas/1.0'
    assert checksum_bytes({p.name:p.read_bytes() for p in path.glob('*.schema.json')})==(path/'SHA256SUMS').read_bytes()
    assert digest((path/'SHA256SUMS').read_bytes())==contract.schema_digest
    assert contract.vendor_source['source_repository']=='PersEval-paper-reproduction'
    revision=contract.vendor_source['source_revision']
    if revision is None:
        assert contract.vendor_source['source_state']=='uncommitted-no-HEAD'
    else:
        import re
        assert re.fullmatch(r'[0-9a-f]{40}',revision), 'release vendoring must pin a genuine Git revision'


def test_minimal_export_uses_evaluator_and_preserves_original(make_run,tmp_path,predictions):
    original=predictions.read_bytes()
    run=make_run()
    assert not list(tmp_path.rglob('*.tar.gz'))
    path,files,docs=exported(run,tmp_path)
    assert path.name==f'{run.run_id}.perseval-result.tar.gz'
    assert docs['metrics']['compact']=={'global_f1':1.0,'user_f1':1.0,'text_f1':1.0}
    assert docs['run']['configuration']['split']['seed']==17
    assert docs['run']['seed']==42
    assert docs['run']['configuration']['hyperparameters']['inference']['model_family']=='encoder'
    assert files['predictions.csv']==b'user_id,text_id,predictions\n01,t1,0\n02,t2,1\n'
    assert predictions.read_bytes()==original
    reports=read_json(files[REPORT_PATH])
    assert all(key in reports for key in ['global','user_average','text_average','trait_average','per_user','per_text','per_trait'])
    assert reports['task_score']=={'kind':'positive-class-f1','positive_label':'1','value':1.0}


def test_stable_identity_deterministic_reexport_after_reload(make_run,tmp_path,monkeypatch):
    run=make_run()
    path,_,_=exported(run,tmp_path)
    first=path.read_bytes()
    monkeypatch.setattr('perseval.results.run.capture_environment',lambda:(_ for _ in ()).throw(AssertionError('must not recapture environment')))
    again=LocalRun(run.path)
    second=again.export_result(tmp_path/'elsewhere')
    assert again.run_id==run.run_id
    assert second.read_bytes()==first


def test_separate_execution_new_run_id_same_configuration(make_run,tmp_path):
    first=make_run();second=make_run()
    assert first.run_id!=second.run_id
    _,_,a=exported(first,tmp_path)
    _,_,b=exported(second,tmp_path)
    assert a['run']['configuration_id']==b['run']['configuration_id']
    assert a['manifest']['payload_digest']!=b['manifest']['payload_digest']


def test_changed_finalized_predictions_fail(make_run,tmp_path):
    run=make_run();exported(run,tmp_path)
    (run.path/'predictions.csv').write_bytes(b'user_id,text_id,predictions\n01,t1,1\n02,t2,1\n')
    with pytest.raises(ProtocolError,match='changed local snapshot'):
        run.export_result(tmp_path/'new-output')
    assert not (tmp_path/'new-output').exists()
    with pytest.raises(ProtocolError,match='immutable'):
        run.replace_predictions(run.path/'predictions.csv')


def test_mutable_draft_requires_explicit_correction(make_run,tmp_path,predictions):
    run=make_run()
    run.update_metadata(method='corrected method',notes='A draft note')
    predictions.write_bytes(b'user_id,text_id,predictions\n01,t1,1\n02,t2,1\n')
    run.replace_predictions(predictions)
    _,files,docs=exported(run,tmp_path)
    assert docs['run']['configuration']['method']=='corrected method'
    assert docs['run']['provenance']['notes']=='A draft note'
    assert read_json(files[REPORT_PATH])['task_score']['value']==pytest.approx(2/3)
    with pytest.raises(ProtocolError,match='immutable'):
        run.update_metadata(notes='changed')


def test_legacy_source_can_change_without_overwriting_prior_run(make_run,tmp_path,predictions):
    run=make_run();path,_,_=exported(run,tmp_path);before=path.read_bytes()
    predictions.write_text('another prediction invocation replaced this legacy filename')
    assert run.export_result(tmp_path/'exports').read_bytes()==before


def test_binary_positive_f1_differs_from_global_macro(make_run,tmp_path,predictions):
    predictions.write_bytes(b'user_id,text_id,predictions\n01,t1,1\n02,t2,1\n')
    _,files,docs=exported(make_run(),tmp_path)
    assert docs['metrics']['compact']['global_f1']==pytest.approx(1/3)
    assert read_json(files[REPORT_PATH])['task_score']['value']==pytest.approx(2/3)


def test_multiclass_task_score_is_macro(make_run,tmp_path,dataset,predictions):
    dataset.name,dataset.label='DICES','Q2_harmful_content_overall'
    dataset.labels={dataset.label:{0,1,2}}
    for row,value in zip(dataset.test_set.annotation,[0,2]):row['label']={dataset.label:value}
    predictions.write_bytes(b'user_id,text_id,predictions\n01,t1,0\n02,t2,2\n')
    _,files,docs=exported(make_run(),tmp_path)
    assert read_json(files[REPORT_PATH])['task_score']=={'kind':'macro-f1','positive_label':None,'value':1.0}
    assert docs['metrics']['scoring_rule']=='perseval.macro-f1.v1'


def test_parse_failure_is_explicit_scored_label(make_run,tmp_path,predictions):
    predictions.write_bytes(b'user_id,text_id,predictions\n01,t1,-1\n02,t2,1\n')
    _,files,docs=exported(make_run(),tmp_path)
    assert docs['run']['provenance']['label_mapping']['__parse_failure__']=='-1'
    assert '-1' in read_json(files[REPORT_PATH])['global']


@pytest.mark.parametrize('content,message',[
    (b'user_id,text_id,predictions,extra\n01,t1,0,x\n02,t2,1,x\n','exactly'),
    (b'user_id,text_id,predictions\n01,t1,0\n01,t1,1\n','duplicate'),
    (b'user_id,text_id,predictions\n01,t1,0\n','exactly cover'),
    (b'user_id,text_id,predictions\n01,t1,0\n03,t2,1\n','exactly cover'),
    (b'user_id,text_id,predictions\n01,t1,\n02,t2,1\n','nonmissing'),
    (b'user_id,text_id,predictions\n01,t1,NaN\n02,t2,1\n','nonmissing'),
    (b'user_id,text_id,predictions\n01,t1,9\n02,t2,1\n','label mapping'),
])
def test_prediction_validation_before_recording(content,message,make_run,predictions,tmp_path):
    predictions.write_bytes(content)
    with pytest.raises(ProtocolError,match=message):make_run()
    assert not list(tmp_path.rglob('descriptor.json'))


def test_unknown_dataset_label_and_invalid_task_settings(dataset):
    for key,value in [('name','not-a-dataset'),('label','not-a-label'),('named',None),('user_adaptation','upload')]:
        other=copy.deepcopy(dataset);setattr(other,key,value)
        with pytest.raises(ProtocolError):make_context(other)
    context=make_context(dataset);context['settings'].update(task='baseline',adaptation='train')
    with pytest.raises(ProtocolError,match='baseline'):validate_context(context)


def test_context_detects_test_fingerprint_tampering(dataset):
    context=make_context(dataset);context['test_annotations'][0][2]='1'
    with pytest.raises(ProtocolError,match='fingerprint'):validate_context(context)


def test_split_seed_captured_at_construction_not_export(dataset,monkeypatch):
    class Dummy:
        name='BREXIT'
        @capture_split
        def get_splits(self,extended,user_adaptation,named,baseline=False):
            return 'existing-return'
    fake=Dummy();monkeypatch.setattr(config,'seed',123)
    assert fake.get_splits(False,False,False,True)=='existing-return'
    monkeypatch.setattr(config,'seed',999)
    assert fake._result_split_definition['seed']==123
    assert fake._result_split_definition['parameters']==config.dataset_specific_splits['BREXIT']


def test_offline_context_save_never_downloads(dataset,tmp_path,monkeypatch):
    monkeypatch.setattr('perseval.data.download',Mock(side_effect=AssertionError('must not download')))
    path=dataset.save_result_context(tmp_path/'context.json')
    assert validate_context(read_json(path.read_bytes()))['dataset']['name']=='BREXIT'


def test_optional_artifacts_explicit_and_frozen(make_run,tmp_path):
    directory=tmp_path/'diagnostics';directory.mkdir();(directory/'plot.txt').write_text('selected')
    (directory/'nested').mkdir();(directory/'nested/table.csv').write_text('x,y\n1,2\n')
    excluded=tmp_path/'secret-prompt.txt';excluded.write_text('not selected')
    run=make_run()
    summary=run.inspect(artifacts={'diagnostics':directory})
    assert not run.finalized
    assert 'artifacts/diagnostics/plot.txt' in summary['included_artifacts']
    assert summary['estimated_artifact_bytes']>len('selected')
    path,files,_=exported(run,tmp_path,artifacts={'diagnostics':directory})
    assert all(not name.startswith('/') and '..' not in name.split('/') for name in files)
    assert not any('secret' in name for name in files)
    before=path.read_bytes();shutil_module=importlib.import_module('shutil');shutil_module.rmtree(directory)
    assert run.export_result(tmp_path/'exports2').read_bytes()==before
    with pytest.raises(ProtocolError,match='frozen'):
        run.export_result(tmp_path/'exports3',artifacts={})


@pytest.mark.parametrize('alias',['../escape','/absolute','nested/name','.hidden'])
def test_artifact_alias_traversal_rejected(alias,make_run,tmp_path):
    file=tmp_path/'selected.txt';file.write_text('x');run=make_run()
    with pytest.raises(ProtocolError):run.export_result(tmp_path/'exports',artifacts={alias:file})
    assert not run.finalized
    assert not list(tmp_path.rglob('*.tar.gz'))


def test_escaping_artifact_symlink_rejected(make_run,tmp_path):
    outside=tmp_path/'outside';outside.write_text('private')
    selected=tmp_path/'selected';selected.mkdir();(selected/'link').symlink_to(outside)
    run=make_run()
    with pytest.raises(ProtocolError,match='symlink'):run.export_result(tmp_path/'exports',artifacts={'diagnostics':selected})
    assert not run.finalized


def test_checksums_cover_all_files(make_run,tmp_path):
    _,files,_=exported(make_run(),tmp_path)
    assert files['checksums.sha256']==checksum_bytes({k:v for k,v in files.items() if k!='checksums.sha256'})
    altered={**files,'predictions.csv':b'changed'}
    with pytest.raises(ProtocolError,match='checksums'):validate_files(altered,Contract())


def test_atomic_output_failure_no_partial_archive(make_run,tmp_path,monkeypatch):
    import perseval.results.io as io_module
    original=io_module.os.replace
    def failing_replace(source,destination):
        if str(destination).endswith('.perseval-result.tar.gz'):raise OSError('simulated disk failure')
        return original(source,destination)
    run=make_run();monkeypatch.setattr(io_module.os,'replace',failing_replace)
    with pytest.raises(OSError,match='disk failure'):run.export_result(tmp_path/'exports')
    assert not run.finalized
    assert not list((tmp_path/'exports').glob('*'))
    assert not list(run.path.glob('.finalize-*'))
    assert not (run.path/'.result.lock').exists()


def test_environment_snapshot_persisted_sorted_nullable(make_run,tmp_path,monkeypatch):
    run=make_run();before=(run.path/'environment.json').read_bytes()
    monkeypatch.setattr('perseval.results.run.capture_environment',Mock(side_effect=AssertionError('no recapture')))
    _,files,_=exported(run,tmp_path)
    assert files['environment.json']==before
    environment=read_json(before)
    assert list(environment['packages'])==sorted(environment['packages'])
    assert 'python' in environment and 'platform' in environment and 'machine' in environment
    details=json.loads(environment['notes'])
    assert set(details)=={'pytorch','cuda_version','cuda_available','gpus'}


def test_inspect_is_read_only_and_missing_provenance_visible(make_run,tmp_path,dataset):
    dataset.result_dataset_revision=None
    run=make_run();before={p.name:p.read_bytes() for p in run.path.iterdir()}
    info=run.inspect()
    assert info['state']=='draft' and 'dataset revision' in info['completeness']['missing']
    assert info['metrics']['compact']['global_f1']==1.0
    assert info['predictions']['count']==2
    assert info['estimated_artifact_bytes']>0
    assert before=={p.name:p.read_bytes() for p in run.path.iterdir()}
    assert not list(tmp_path.rglob('*.tar.gz'))


def test_missing_method_can_be_supplied_before_export(make_run,tmp_path):
    run=make_run(method=None)
    assert 'incomplete' in run.inspect()['completeness']['errors'][0]
    with pytest.raises(ProtocolError,match='method'):run.export_result(tmp_path/'exports')
    run.update_metadata(method='documented-custom-method')
    exported(run,tmp_path)


def test_no_results_package_import():
    root=Path(__file__).parents[1]
    for path in (root/'perseval').rglob('*.py'):
        text=path.read_text()
        assert 'import perseval_results' not in text
        assert 'from perseval_results' not in text
        assert 'sys.path' not in text


@pytest.fixture
def encoder(dataset,monkeypatch,tmp_path):
    import perseval.baselines.encoder as module
    monkeypatch.setattr(module.AutoTokenizer,'from_pretrained',Mock(return_value=NS()))
    model=NS(config=NS(num_labels=2,_commit_hash='a'*40))
    monkeypatch.setattr(module.AutoModelForSequenceClassification,'from_pretrained',Mock(return_value=model))
    monkeypatch.setattr(config,'prediction_dir',str(tmp_path/'legacy-predictions'))
    monkeypatch.setattr(config,'output_root',tmp_path/'outputs')
    instance=module.PerspectivistEncoder('synthetic-model',dataset,'hs')
    monkeypatch.setattr(instance,'_PerspectivistEncoder__generate_data',lambda split:([{},{}],[('01','t1'),('02','t2')]))
    trainer=NS(args=NS(seed=42,learning_rate=0.001,num_train_epochs=2),model=model,
               predict=Mock(return_value=NS(predictions=np.array([[1,0],[0,1]]))))
    return instance,trainer


def test_standard_model_integration_and_return_compatibility(encoder,tmp_path):
    model,trainer=encoder
    assert model.predict(trainer) is None
    run=model.last_run
    assert isinstance(run,LocalRun)
    assert not run.finalized and not list(tmp_path.rglob('*.tar.gz'))
    legacy=next((tmp_path/'legacy-predictions').glob('*.csv'))
    assert list(pd.read_csv(legacy).columns)==['user_id','text_id','predictions']
    assert run.inspect()['metrics']['compact']['global_f1']==1.0
    path,_,docs=exported(run,tmp_path)
    assert docs['run']['configuration']['hyperparameters']['training']['learning_rate']==0.001
    assert docs['run']['configuration']['model']['revision']=='a'*40
    prior=run.run_id
    model.predict(trainer)
    assert model.last_run.run_id!=prior
    assert run.export_result(tmp_path/'reexport').read_bytes()==path.read_bytes()


def test_model_recording_opt_out_keeps_existing_behavior(encoder,tmp_path):
    model,trainer=encoder
    assert model.predict(trainer,record_run=False) is None
    assert model.last_run is None and not (tmp_path/'outputs').exists()


def test_model_metadata_arguments_avoid_manual_duplication(encoder,tmp_path):
    model,trainer=encoder
    model.predict(trainer,result_metadata={'method':'my-adaptation','representation':'my-features','notes':'experiment note'})
    _,_,docs=exported(model.last_run,tmp_path)
    assert docs['run']['configuration']['method']=='my-adaptation'
    assert docs['run']['configuration']['representation']=='my-features'
    assert docs['run']['provenance']['notes']=='experiment note'


def test_recording_failure_does_not_return_a_stale_run(encoder,tmp_path,monkeypatch):
    model,trainer=encoder;model.predict(trainer);assert model.last_run is not None
    monkeypatch.setattr(model.persp_dataset,'label','unknown')
    with pytest.warns(RuntimeWarning,match='record'):
        assert model.predict(trainer,result_metadata={'unknown-field':'x'}) is None
    assert model.last_run is None and model.last_run_error
    assert list((tmp_path/'legacy-predictions').glob('*.csv'))


def test_evaluation_alone_does_not_export(dataset,predictions,tmp_path):
    from perseval.evaluation import Evaluator
    e=Evaluator(pd.read_csv(predictions,dtype={'user_id':str,'text_id':str}),dataset.test_set,'hs')
    e.global_metrics()
    assert not list(tmp_path.rglob('*.tar.gz'))
    assert not list(tmp_path.rglob('descriptor.json'))


def test_python_module_entrypoint_does_not_expose_export_cli():
    result=subprocess.run([sys.executable,'-m','perseval','result','export','--help'],text=True,capture_output=True,cwd=Path(__file__).parents[1])
    assert result.returncode!=0
    assert 'No module named perseval.__main__' in result.stderr


def test_training_alone_has_no_run_or_portable_export(encoder,tmp_path,monkeypatch):
    import perseval.baselines.encoder as module
    model,_=encoder
    model.training_split.annotation=copy.deepcopy(model.test_split.annotation)
    monkeypatch.setattr(model,'_PerspectivistEncoder__add_special_tokens_to_tokenizer',lambda:None)
    monkeypatch.setattr(module,'TrainingArguments',lambda **kwargs:NS(**kwargs))
    trainer=Mock()
    monkeypatch.setattr(module,'CustomTrainer',Mock(return_value=trainer))
    monkeypatch.setattr(module.torch.cuda,'get_device_name',lambda index:'synthetic CPU fixture')
    monkeypatch.setitem(config.model_config,'synthetic-model',copy.deepcopy(config.model_config['roberta-base']))
    assert model.train() is trainer
    trainer.train.assert_called_once()
    assert model.last_run is None
    assert not list(tmp_path.rglob('*.tar.gz'))
    assert not list(tmp_path.rglob('descriptor.json'))


def test_llm_preserves_raw_files_and_records_only_logical_columns(dataset,tmp_path,monkeypatch):
    import perseval.baselines.llm as module
    def fake_pipeline(messages,**kwargs):
        label='not hate speech' if 'first' in messages[0]['content'] else 'hate speech'
        return [{'generated_text':[{'role':'assistant','content':'{'+label+'}'}]}]
    fake_pipeline.model=NS(config=NS(_commit_hash='b'*40))
    monkeypatch.setattr(module,'pipeline',lambda *args,**kwargs:fake_pipeline)
    monkeypatch.setattr(config,'prediction_dir_llama',str(tmp_path/'raw-llm'))
    monkeypatch.setattr(config,'output_root',tmp_path/'outputs')
    model=module.PerspectivistLLM('meta-llama/Meta-Llama-3.1-8B-Instruct',dataset,'hs')
    assert model.predict() is None
    assert model.last_run is not None and not list(tmp_path.rglob('*.tar.gz'))
    raw=next((tmp_path/'raw-llm').glob('*.csv'))
    assert list(pd.read_csv(raw).columns)==['user_id','text_id','predictions','thoughts','prompt']
    first=model.last_run
    _,files,docs=exported(first,tmp_path)
    assert docs['run']['seed'] is None
    assert docs['metrics']['compact']['global_f1']==1.0
    assert set(files)=={'manifest.json','run.json','metrics.json','predictions.csv','environment.json','checksums.sha256',REPORT_PATH}
    model.predict()
    assert model.last_run.run_id!=first.run_id
    # Legacy appending is retained, while each run owns only this invocation's rows.
    assert model.last_run.inspect()['predictions']['count']==2


def test_llm_does_not_silently_choose_or_ensemble_perspectives(dataset,tmp_path,monkeypatch):
    import perseval.baselines.llm as module
    dataset.named=True
    for user in dataset.test_set.users.values():user.traits['another']=['z']
    monkeypatch.setattr(module,'pipeline',lambda *args,**kwargs:lambda *args,**kwargs:[{'generated_text':[{'content':'{hate speech}'}]}])
    monkeypatch.setattr(config,'prediction_dir_llama',str(tmp_path/'raw-llm'))
    monkeypatch.setattr(config,'output_root',tmp_path/'outputs')
    model=module.PerspectivistLLM('meta-llama/Meta-Llama-3.1-8B-Instruct',dataset,'hs')
    assert model.predict() is None
    assert model.last_run is None
    assert set(model.last_runs)=={'group','another'}
    assert len({run.run_id for run in model.last_runs.values()})==2
    assert not list(tmp_path.rglob('*.tar.gz'))


def test_lamp_preserves_raw_api_and_records_normalized_predictions(dataset,tmp_path,monkeypatch):
    import perseval.baselines.lamp as module
    class Tensor:
        shape=(1,1)
        def __len__(self):return 1
        def __getitem__(self,key):return self
        def to(self,*args):return self
        def unsqueeze(self,*args):return self
    dataset.user_adaptation='test'
    loaded=NS(config=NS(_commit_hash='c'*40),generate=Mock(return_value=Tensor()))
    tokenizer=NS(pad_token=None,eos_token='eos',decode=lambda *args,**kwargs:'hate speech')
    monkeypatch.setattr(module.AutoTokenizer,'from_pretrained',Mock(return_value=tokenizer))
    monkeypatch.setattr(module.AutoModelForCausalLM,'from_pretrained',Mock(return_value=loaded))
    monkeypatch.setattr(config,'output_root',tmp_path/'outputs')
    monkeypatch.setattr(config,'data_lamp_dir',str(tmp_path/'lamp-input'))
    input_dir=Path(config.data_lamp_dir);input_dir.mkdir()
    (input_dir/'BREXIT_False_merged.json').write_text(json.dumps({
        '01':[{'id':'t1','input':'first','profile':[],'hs':0}],
        '02':[{'id':'t2','input':'second','profile':[],'hs':1}]}))
    model=module.PerspectivistLaMP('meta-llama/Meta-Llama-3.1-8B-Instruct',dataset,'hs',False,'BREXIT')
    monkeypatch.setattr(model,'create_prompt_generator',lambda *args:lambda *args:'synthetic prompt')
    monkeypatch.setattr(model,'create_preprocessor',lambda:lambda profiles:{'input_ids':Tensor(),'attention_mask':Tensor()})
    assert model.evaluate_dataset() is None
    assert model.last_run is not None
    assert list(pd.read_csv(input_dir/'output/BREXIT_False.csv').columns)==['user_id','id','predictions']
    _,files,docs=exported(model.last_run,tmp_path)
    assert docs['run']['configuration']['method']=='ipa-lamp'
    assert docs['run']['configuration']['adaptation']=='test'
    assert list(csv.reader(files['predictions.csv'].decode().splitlines()))[0]==['user_id','text_id','predictions']
    assert not any('prompt' in name for name in files)


def test_fixture_snapshot_inventory():
    listing=checksum_bytes({p.relative_to(FIXTURES).as_posix():p.read_bytes() for p in FIXTURES.rglob('*') if p.is_file()})
    assert listing==FIXTURES.with_name('result-bundle-1.0.SHA256SUMS').read_bytes()
    assert Contract().vendor_source['conformance_digest']==digest(listing)


def test_opaque_na_identifier_is_preserved(dataset,make_run,predictions,tmp_path):
    dataset.test_set.annotation[0]['user']='NA'
    dataset.test_set.users['NA']=dataset.test_set.users.pop('01');dataset.test_set.users['NA'].id='NA'
    predictions.write_bytes(b'user_id,text_id,predictions\nNA,t1,0\n02,t2,1\n')
    _,files,_=exported(make_run(),tmp_path)
    assert b'NA,t1,0\n' in files['predictions.csv']


def test_numeric_float_identifiers_preserve_legacy_csv_values(dataset,make_run,predictions,tmp_path):
    dataset.test_set.annotation[0]['user']=1.0
    dataset.test_set.users[1.0]=dataset.test_set.users.pop('01');dataset.test_set.users[1.0].id=1.0
    predictions.write_bytes(b'user_id,text_id,predictions\n1.0,t1,0\n02,t2,1\n')
    _,files,_=exported(make_run(),tmp_path)
    assert b'1.0,t1,0\n' in files['predictions.csv']


def test_model_retains_selected_dataset_task_and_split(encoder,dataset,tmp_path):
    model,trainer=encoder
    original_fingerprint=make_context(model.persp_dataset)['split']['test_fingerprint']
    dataset.named=True;dataset.user_adaptation='train';dataset.extended=True
    dataset._result_split_definition['seed']=999
    dataset.test_set=PerspectivistSplit('test')
    assert model.predict(trainer) is None
    _,_,docs=exported(model.last_run,tmp_path)
    config_=docs['run']['configuration']
    assert config_['task']=='baseline' and config_['adaptation']=='none' and config_['extended'] is False
    assert config_['split']['seed']==17
    assert config_['split']['test_fingerprint']==original_fingerprint


def test_input_logical_column_order_is_canonicalized_without_source_edits(make_run,predictions,tmp_path):
    raw=b'predictions,text_id,user_id\n0,t1,01\n1,t2,02\n'
    predictions.write_bytes(raw)
    _,files,_=exported(make_run(),tmp_path)
    assert files['predictions.csv']==b'user_id,text_id,predictions\n01,t1,0\n02,t2,1\n'
    assert predictions.read_bytes()==raw


def test_real_producer_conformance_vector_is_local_and_pinned():
    root=Path(__file__).parent/'fixtures'
    fixture=root/'producer-v1.perseval-result.tar.gz'
    provenance=json.loads((root/'producer-v1.provenance.json').read_bytes())
    assert provenance['synthetic'] is True
    assert digest(fixture.read_bytes())==provenance['archive_sha256']
    assert provenance['schema_digest']==Contract().schema_digest
    _,docs=validate_bundle(fixture,Contract())
    assert docs['metrics']['compact']['global_f1']==1.0


def test_unsupported_contract_version_is_actionable():
    files,docs=validate_bundle(FIXTURES/'valid/minimal.perseval-result.tar.gz',Contract())
    docs['manifest']['schema_version']='999.0'
    with pytest.raises(ProtocolError,match='unsupported bundle schema version.*compatible'):
        Contract().validate('manifest',docs['manifest'])


def test_schema_drift_requires_exact_offline_vendor_restore(tmp_path,monkeypatch):
    import shutil
    import perseval.results.contract as module
    source=Path(module.__file__).parent/'schemas'
    shutil.copytree(source,tmp_path/'schemas')
    path=tmp_path/'schemas/1.0/run.schema.json';original=path.read_bytes()
    monkeypatch.setattr(module,'__file__',str(tmp_path/'contract.py'))
    for changed in (original+b'\n',b'{invalid JSON'):
        path.write_bytes(changed)
        with pytest.raises(ProtocolError,match='schema drift.*Restore exact'):
            module.Contract()
