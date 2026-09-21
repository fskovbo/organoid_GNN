"""Exercise notebook-owned interval selection and deferred mesh rendering."""
import ast
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def browser_namespace():
    notebook = json.loads((ROOT/'experiments/data_quality/marker_complexity.ipynb').read_text())
    ns = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            definitions = [node for node in ast.parse(''.join(cell['source'])).body
                           if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), 'complexity widget', 'exec'), ns)
    ns['display'] = Mock()
    return ns


class MarkerComplexityTests(unittest.TestCase):
    def setUp(self):
        self.ns = browser_namespace()
        self.table = pd.DataFrame(dict(sphericity=[.3,.8,.8,np.nan],diversity=[.4,.7,.2,.7]),
                                  index=['low','high','other','missing'])

    def test_inclusive_joint_intervals_and_invalid_values(self):
        select = self.ns['complexity_candidates']
        self.assertEqual(list(select(self.table,(.8,.8),(.7,.7)).index),['high'])
        self.assertEqual(list(select(self.table,(0.,1.),(0.,1.)).index),['high','low','other'])
        for bounds in [(1.,0.), (np.nan,1.), (0.,)]:
            with self.assertRaises(ValueError):select(self.table,bounds,(0.,1.))

    def test_controls_defer_render_and_handle_empty_ranges(self):
        ns = self.ns
        graphs = {key:object() for key in self.table.index}
        ns['project_marker_categories_for_graph'] = Mock(return_value={'mesh':'mesh','mesh_categories':['A']})
        figure = Mock()
        figure.to_json.return_value = json.dumps({'data': [{'type': 'mesh3d'}], 'layout': {}})
        ns['plot_categorical_mesh'] = Mock(return_value=figure)
        with contextlib.redirect_stdout(io.StringIO()):
            browser,controls = ns['build_complexity_browser'](self.table,graphs,['A'],{'A':'red','none':'gray'})
            self.assertEqual(controls['organoid'].value,'high')
            self.assertEqual(controls['preview'].value.count('<table'), 1)
            self.assertIn('<th>high</th>', controls['preview'].value)
            ns['display'].assert_not_called()
            ns['project_marker_categories_for_graph'].assert_not_called()
            controls['sphericity'].value=(.3,.3)
            controls['diversity'].value=(.4,.4)
            self.assertEqual(controls['organoid'].value,'low')
            self.assertEqual(controls['preview'].value.count('<table'), 1)
            self.assertIn('<th>low</th>', controls['preview'].value)
            self.assertNotIn('<th>high</th>', controls['preview'].value)
            controls['show_mesh'].click()
            self.assertIs(ns['project_marker_categories_for_graph'].call_args.args[0],graphs['low'])
            self.assertIn('diversity=0.400',ns['plot_categorical_mesh'].call_args.kwargs['title'])
            # One MIME output per click, independent of default Plotly renderers.
            self.assertEqual(len(controls['mesh_output'].outputs), 1)
            self.assertEqual(set(controls['mesh_output'].outputs[0]['data']),
                             {'application/vnd.plotly.v1+json'})
            self.assertFalse(any(call.args[0] is figure for call in ns['display'].call_args_list))
            controls['show_mesh'].click()
            self.assertEqual(len(controls['mesh_output'].outputs), 1)
            controls['diversity'].value=(.8,1.)
            self.assertIsNone(controls['organoid'].value)
            self.assertEqual(controls['preview'].value, '')
            self.assertTrue(controls['show_mesh'].disabled)
            self.assertIn('No organoids match',controls['status'].value)
            controls['diversity'].value=(0.,1.)
            self.assertEqual(controls['preview'].value.count('<table'), 1)
            ns['project_marker_categories_for_graph'].side_effect=FileNotFoundError('missing mesh')
            controls['show_mesh'].click()
            self.assertEqual(controls['organoid'].value,'low')
            self.assertEqual(ns['plot_categorical_mesh'].call_count,2)
            self.assertEqual(controls['mesh_output'].outputs, ())
            browser.close()
            for control in controls.values():control.close()


if __name__ == '__main__':unittest.main()
