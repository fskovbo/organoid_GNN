"""Exercise the notebook's candidate selection and ground-truth mesh callback."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def notebook_namespace():
    notebook = json.loads((ROOT/'experiments/data_quality/cohort_review.ipynb').read_text())
    source = ''.join(notebook['cells'][9]['source'])
    definitions = [node for node in ast.parse(source).body
                   if isinstance(node,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
    namespace = dict(np=np,pd=pd,display=Mock())
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'cohort widget','exec'),namespace)
    return namespace


class CohortReviewTests(unittest.TestCase):
    def test_rank_sphericity_controls_and_ground_truth_projection(self):
        ns = notebook_namespace()
        ranked = pd.DataFrame(dict(mse_rank=[1,2,3],mse=[9.,4.,1.],sphericity=[.2,.8,np.nan]),
                              index=['worst','middle','missing'])
        cells = pd.DataFrame(dict(organoid_str=['middle','middle'],node=[1,0],y_true=[20.,10.],y_pred=[18.,12.]))
        raw = {'middle':SimpleNamespace(x=np.zeros((2,2)))}
        ns['project_node_quantities_to_mesh'] = Mock(return_value={'mesh':'fixture'})
        ns['plot_projected_true_vs_pred'] = Mock(return_value='rendered truth/prediction')
        with contextlib.redirect_stdout(io.StringIO()):
            browser, controls = ns['build_candidate_browser'](ranked,cells,raw)
            self.assertEqual(controls['organoid'].value,'worst')
            controls['sphericity'].value=(.8,.8)
            self.assertEqual(controls['organoid'].value,'middle')
            self.assertIn('#2',controls['organoid'].options[0][0])
            controls['show_mesh'].click()
            call = ns['project_node_quantities_to_mesh'].call_args
            self.assertIs(call.args[0],raw['middle'])
            np.testing.assert_array_equal(call.kwargs['node_true'],[10.,20.])
            np.testing.assert_array_equal(call.kwargs['node_pred'],[12.,18.])
            self.assertEqual(ns['plot_projected_true_vs_pred'].call_args.kwargs['true_title'],'Ground truth curvature')
            controls['rank'].value=(1,1)
            self.assertIsNone(controls['organoid'].value)
            self.assertTrue(controls['show_mesh'].disabled)
            self.assertIn('No organoids match',controls['status'].value)
            controls['include_missing'].value=True
            controls['rank'].value=(3,3)
            self.assertEqual(controls['organoid'].value,'missing')
            # Missing mesh metadata is surfaced for this candidate, not replaced.
            controls['show_mesh'].click()
            self.assertEqual(controls['organoid'].value,'missing')
            self.assertEqual(ns['project_node_quantities_to_mesh'].call_count,1)
            browser.close()
            for control in controls.values():control.close()

    def test_rank_ranges_and_missing_sphericity(self):
        ns = notebook_namespace()
        ranked=pd.DataFrame(dict(mse_rank=[1,2],mse=[2.,1.],sphericity=[np.nan,np.nan]),index=['a','b'])
        self.assertTrue(ns['review_candidates'](ranked,(1,2),(0.,1.)).empty)
        self.assertEqual(list(ns['review_candidates'](ranked,(2,2),(0.,1.),True).index),['b'])
        with self.assertRaisesRegex(ValueError,'lower bounds'):
            ns['review_candidates'](ranked,(2,1),(0.,1.))
        with contextlib.redirect_stdout(io.StringIO()):
            browser,controls=ns['build_candidate_browser'](ranked,pd.DataFrame(),{},include_missing=False)
            self.assertTrue(controls['show_mesh'].disabled)
            controls['include_missing'].value=True
            self.assertEqual(controls['organoid'].value,'a')
            browser.close()
            for control in controls.values():control.close()

    def test_model_depth_and_fold_are_explicit(self):
        notebook=json.loads((ROOT/'experiments/data_quality/cohort_review.ipynb').read_text())
        source=''.join(notebook['cells'][5]['source'])
        source=source[source.index('# Saved model selection:'):source.index('# Inference resources')]
        records=pd.DataFrame([dict(key=f'd{d}_f{f}',name='gin',depth=d,fold=f,seed=42,hidden_dim=8)
                              for d in [0,2] for f in [0,1]])
        ns=dict(run=SimpleNamespace(records=records,legacy=None),MODEL_DEPTH=2,MODEL_FOLD=1,
                MODEL_NAME='gin',MODEL_SEED=None,HIDDEN_DIM=None,MODEL_KEY=None)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source,ns)
            self.assertEqual(ns['SELECTED_MODEL_KEY'],'d2_f1')
            self.assertIsNone(ns['MODEL_KEY'])
            ns['MODEL_FOLD']=0
            exec(source,ns)
            self.assertEqual(ns['SELECTED_MODEL_KEY'],'d2_f0')
            ns['MODEL_DEPTH']=99
            with self.assertRaisesRegex(ValueError,'exactly one'):
                exec(source,ns)


if __name__=='__main__':unittest.main()
