"""CPU regression checks: no Isaac Gym import, CUDA call or experiment launch."""
import ast
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]


class Wrapper:
    def __init__(self, env): self.env = env
    def __getattr__(self, key): return getattr(self.env, key)
    def step(self, action): return self.env.step(action)


def collector_class():
    tree = ast.parse((ROOT / 'per_env_diag.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = {'np': np, 'th': th, 'WrapperEnv': Wrapper}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'per_env_diag.py', 'exec'), scope)
    return scope['PerEnvDiag']


class DelayedResetEnv:
    num_env = 2
    device = 'cpu'
    def __init__(self):
        self.tensors = {'root': th.zeros(4, 13)}
        self.task = SimpleNamespace(goal=th.zeros(2, 7))
        self.scene = SimpleNamespace(cur_ids=th.tensor([0, 1]), cur_radii=th.ones(2),
            table_dims=th.ones(2, 3), table_pos=th.zeros(2, 3),
            _valid_poses=th.zeros(2, 1, 7), stable_poses=th.zeros(2, 1, 7),
            cfg=SimpleNamespace(num_obj_per_env=1), _per_env_offsets=th.zeros(2, dtype=th.long))
        self.robot = SimpleNamespace(indices=th.tensor([2, 3]))
        self.age = th.zeros(2, dtype=th.long)
        self.pending = th.zeros(2, dtype=th.bool)
        self.generation = th.zeros(2, dtype=th.long)
    def reset_indexed(self, indices=None):
        ids = th.arange(2) if indices is None else th.as_tensor(indices)
        self.age[ids] = 0
        self.pending[ids] = False
        self.generation[ids] += 1
        self.tensors['root'][ids, 0] = self.generation[ids].float() * 10 + ids
        self.task.goal[ids, 0] = self.generation[ids].float() * 100 + ids
        return 'reset-result'
    def step(self, action):
        self.reset_indexed(th.nonzero(self.pending).flatten())
        self.age += 1
        self.tensors['root'][:2, 0] += 1
        done = self.age >= th.tensor([1, 3])
        self.pending = done.clone()
        return self.tensors['root'].clone(), th.zeros(2), done, {'success':done.clone(), 'timeout':th.zeros(2,dtype=th.bool)}


def validator():
    spec = importlib.util.spec_from_file_location('validate_cell', ROOT/'analysis/s5_validate_cell.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.base = DelayedResetEnv()
        self.diag = collector_class()(Wrapper(self.base))
        self.assertEqual(self.base.reset_indexed(), 'reset-result')
    def test_exact_start_and_goal_after_delayed_and_partial_resets(self):
        for _ in range(4): self.diag.step(None)
        env = np.concatenate(self.diag.rec['env'])
        start = np.concatenate(self.diag.rec['start'])[:, 0]
        goal = np.concatenate(self.diag.rec['goal'])[:, 0]
        lengths = np.concatenate(self.diag.rec['len'])
        np.testing.assert_array_equal(start[env == 0], [10,20,30,40])
        np.testing.assert_array_equal(goal[env == 0], [100,200,300,400])
        np.testing.assert_array_equal(start[env == 1], [11])
        np.testing.assert_array_equal(goal[env == 1], [101])
        np.testing.assert_array_equal(lengths[env == 1], [3])
        self.assertFalse(self.diag.errors)
    def test_observer_does_not_change_rollout(self):
        plain = DelayedResetEnv(); plain.reset_indexed()
        for _ in range(8):
            left, right = plain.step(None), self.diag.step(None)
            for i in range(3): th.testing.assert_close(left[i],right[i])
            for key in left[3]: th.testing.assert_close(left[3][key],right[3][key])
    def test_empty_and_no_completed_envs(self):
        self.base.pending.zero_()
        self.base.age[:] = -10
        self.diag.step(None)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'diag.npz'; self.diag.dump(path)
            with np.load(path,allow_pickle=False) as d:
                self.assertEqual(d['start'].shape,(0,3))
                self.assertEqual(len(d['pending_env']),2)
    def test_missing_initial_reset_is_not_silent(self):
        base=DelayedResetEnv(); diag=collector_class()(base)
        diag.step(None)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError): diag.dump(Path(td)/'diag.npz')
    def test_restores_hook(self):
        self.diag.close_diagnostics()
        self.assertEqual(self.base.reset_indexed,self.diag._reset_original)
    def test_dump_is_snapshot_not_live_view(self):
        self.diag.step(None)
        self.base.tensors['root'].fill_(999)
        self.assertEqual(self.diag.rec['start'][0][0,0],10)


class CellValidationTests(unittest.TestCase):
    def fixture(self, directory):
        path=Path(directory)/'cell.log'
        counts=dict(term_done=2,term_success=1,term_immediate_le3=1,
            frac_success=.5,frac_oob=.5,frac_timeout=0.,frac_immediate=.5)
        path.write_text('TAXONOMY '+repr(counts)+"\n{'valid/balanced_suc_rate': 0.5}\n")
        np.savez(path.with_name('diag.npz'),schema_version=2,errors=np.array([],dtype=str),
            env=[0,1],len=[1,4],suc=[True,False],timeout=[False,False],
            start=np.zeros((2,3)),goal=np.ones((2,3)))
        return path
    def test_valid_cell(self):
        with tempfile.TemporaryDirectory() as td:
            path=self.fixture(td); validator().append_row(path,'tag,0.1,0',Path(td)/'out.csv','OK')
            self.assertTrue((Path(td)/'out.csv').read_text().endswith(',OK\n'))
    def test_failed_process_cannot_become_ok(self):
        with tempfile.TemporaryDirectory() as td:
            path=self.fixture(td)
            with self.assertRaises(ValueError): validator().append_row(path,'tag,.1,0',Path(td)/'out.csv','FAILED')
            self.assertTrue((Path(td)/'out.csv').read_text().endswith(',FAILED\n'))
    def test_missing_or_corrupt_dump_rejected(self):
        for contents in [None,b'not an npz']:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as td:
                path=self.fixture(td); dump=path.with_name('diag.npz'); dump.unlink()
                if contents: dump.write_bytes(contents)
                with self.assertRaises((ValueError,OSError)): validator().validate(path)
    def test_nonfinite_start_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=self.fixture(td)
            with np.load(path.with_name('diag.npz')) as d: arrays=dict(d)
            arrays['start'][0,0]=np.nan
            np.savez(path.with_name('diag.npz'),**arrays)
            with self.assertRaisesRegex(ValueError,'nonfinite'): validator().validate(path)
    def test_taxonomy_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=self.fixture(td); path.write_text(path.read_text().replace("'term_done': 2","'term_done': 3"))
            with self.assertRaisesRegex(ValueError,'disagree'): validator().validate(path)
    def test_collector_errors_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=self.fixture(td)
            with np.load(path.with_name('diag.npz')) as d: arrays=dict(d)
            arrays['errors']=np.array(['failed reset observation'])
            np.savez(path.with_name('diag.npz'),**arrays)
            with self.assertRaisesRegex(ValueError,'diagnostic errors'): validator().validate(path)


class LaunchLockTests(unittest.TestCase):
    def test_pending_gpu_entries_refuse_without_permission(self):
        import os
        env=dict(os.environ,DYWA_GPU_APPROVED='0')
        for name in [p.name for p in ROOT.glob('*.sh') if p.name not in ('gpu_permission.sh','s5_common.sh')]:
            with self.subTest(name=name):
                out=subprocess.run(['bash',str(ROOT/name)],env=env,capture_output=True,text=True)
                self.assertEqual(out.returncode,64,out.stderr)
                self.assertIn('GPU execution is locked',out.stderr)
    def test_generator_reproduces_committed_entry(self):
        import sys
        before=(ROOT/'valid_ppo_arm_diag.py').read_bytes()
        subprocess.run([sys.executable,str(ROOT/'patch_diag.py')],check=True,capture_output=True)
        self.assertEqual(before,(ROOT/'valid_ppo_arm_diag.py').read_bytes())


class QueueFailureTests(unittest.TestCase):
    def run_stub_queue(self, stages):
        # All child entries are test doubles: no GPU command exists in this fixture.
        import os
        td=tempfile.TemporaryDirectory();root=Path(td.name)
        (root/'gpu_s5_queue.sh').write_bytes((ROOT/'gpu_s5_queue.sh').read_bytes())
        (root/'gpu_permission.sh').write_text('return 0\n')
        (root/'gpu_s5_diag.sh').write_text('exit 7\n')
        (root/'sizeinv_v2.sh').write_text('touch "'+str(root/'eval_ran')+'"; exit 3\n')
        ckpt=root/'ckpt';ckpt.mkdir();(ckpt/'last.ckpt').touch()
        env=dict(os.environ,STAGES=stages,FINETUNED_CKPT=str(ckpt))
        run=subprocess.run(['bash',str(root/'gpu_s5_queue.sh')],env=env,capture_output=True,text=True)
        return td,root,run
    def test_failed_diagnostic_stops_remaining_queue(self):
        td,root,run=self.run_stub_queue('1 2')
        try:
            self.assertNotEqual(run.returncode,0)
            self.assertFalse((root/'eval_ran').exists())
        finally: td.cleanup()
    def test_failed_final_eval_does_not_print_done(self):
        td,root,run=self.run_stub_queue('5')
        try:
            self.assertNotEqual(run.returncode,0)
            self.assertNotIn('queue DONE',run.stdout)
            self.assertTrue((root/'eval_ran').exists())
        finally: td.cleanup()


class GeometryGateTests(unittest.TestCase):
    def test_saved_configuration_rejects_unsupported_large_radius(self):
        import sys
        p=subprocess.run([sys.executable,str(ROOT/'analysis/s5_geometry_gate.py'),'--radius','.30'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('BLOCKED',p.stderr)
    def test_small_radius_passes_support_certificate(self):
        import sys
        p=subprocess.run([sys.executable,str(ROOT/'analysis/s5_geometry_gate.py'),'--radius','.10'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_geometry_inputs_must_be_positive(self):
        import sys
        p=subprocess.run([sys.executable,str(ROOT/'analysis/s5_geometry_gate.py'),'--radius','-.10'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
    def test_radius_grid_generator_preserves_existing_config(self):
        import sys,yaml
        before=(ROOT/'bimanual_scales_radius_grid.yaml').read_bytes()
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'grid.yaml'
            subprocess.run([sys.executable,str(ROOT/'analysis/make_bimanual_radius_grid.py'),'--output',str(target)],check=True,capture_output=True)
            self.assertEqual(yaml.safe_load(before),yaml.safe_load(target.read_bytes()))
        self.assertEqual(before,(ROOT/'bimanual_scales_radius_grid.yaml').read_bytes())


if __name__ == '__main__': unittest.main()
