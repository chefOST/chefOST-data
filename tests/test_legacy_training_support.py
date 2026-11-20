"""Centralized legacy training preserves checkpoint selection without running jobs."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'training_support' / 'vidosc_legacy'


class LegacyTrainingSupportTests(unittest.TestCase):
    def test_source_bootstrap_requires_a_model_checkout(self):
        spec = importlib.util.spec_from_file_location('legacy_bootstrap_fixture', SUPPORT/'bootstrap.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'CHEFOST_VIDOSC_SOURCE'):
                module.configure()
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'VidOSC'
            model.mkdir()
            (model/'model.py').write_text('# fixture model location\n')
            with patch.dict(os.environ, {'CHEFOST_VIDOSC_SOURCE': folder}), patch.object(sys, 'path', list(sys.path)):
                self.assertEqual(module.configure(), Path(folder).resolve())
                self.assertEqual(sys.path[0], str(model.resolve()))

    def load_driver(self, filename, temporary):
        calls = {'checkpoints': [], 'fit': [], 'validate': [], 'test': []}
        class Task:
            metric_name_list = ['val_avg_prec_1', 'val_avg_action_prec_1']
            ckpt_save_format = 'model-{epoch:02d}-{val_avg_prec_1:.4f}-{val_avg_action_prec_1:.4f}'
            def __init__(self, args): pass
            @classmethod
            def load_from_checkpoint(cls, checkpoint_path, args): return cls(args)
        class Trainer:
            def __init__(self, **kwargs): calls['trainer'] = kwargs
            def fit(self, task): calls['fit'].append(task)
            def validate(self, task):
                calls['validate'].append(task)
                return [{'val_avg_prec_1': .5}]
            def test(self, task):
                calls['test'].append(task)
                return [{'test_avg_prec_1': .25}]
        def checkpoint(**kwargs):
            calls['checkpoints'].append(kwargs)
            return kwargs
        pl = ModuleType('pytorch_lightning'); pl.Trainer = Trainer
        loggers = ModuleType('pytorch_lightning.loggers')
        loggers.TensorBoardLogger = lambda **kwargs: kwargs
        loggers.WandbLogger = lambda **kwargs: kwargs
        callbacks = ModuleType('pytorch_lightning.callbacks'); callbacks.ModelCheckpoint = checkpoint
        task_name = 'task_changeit' if filename == 'train_changeit.py' else 'task'
        task = ModuleType('training_support.vidosc_legacy.'+task_name); task.FrameCls = Task
        modules = {'pytorch_lightning': pl, 'pytorch_lightning.loggers': loggers,
                   'pytorch_lightning.callbacks': callbacks, task.__name__: task}
        name = 'training_support.vidosc_legacy.'+Path(filename).stem
        spec = importlib.util.spec_from_file_location(name, SUPPORT/filename)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules): spec.loader.exec_module(module)
        module.args = SimpleNamespace(log_dir=temporary, log_name='fixture', gpus=1, n_epochs=2,
                                      ckpt='', accelerator='cpu', wandb_project=None, novel_obj=True)
        return module, calls

    def test_training_preserves_validation_metric_checkpoint_selection(self):
        for filename in ('train.py', 'train_changeit.py'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                module, calls = self.load_driver(filename, folder)
                module.main()
                self.assertEqual([x['monitor'] for x in calls['checkpoints']],
                                 ['val_avg_prec_1', 'val_avg_action_prec_1'])
                self.assertTrue(all(x['mode']=='max' and x['save_top_k']==1 for x in calls['checkpoints']))
                self.assertEqual(len(calls['fit']), 1)
                self.assertEqual(calls['validate'], [])

    def test_checkpoint_validation_entrypoint_is_centralized_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            module, calls = self.load_driver('train_changeit.py', folder)
            module.args.ckpt = 'existing-checkpoint.ckpt'
            module.main()
            self.assertEqual(calls['fit'], [])
            self.assertEqual(len(calls['validate']), 1)
            self.assertEqual(len(calls['test']), 1)
            path = Path(folder)/'changeit/fixture/eval_metrics.json'
            self.assertEqual(json.loads(path.read_text()), {'val_avg_prec_1': .5, 'test_avg_prec_1': .25})


if __name__ == '__main__':
    unittest.main()
