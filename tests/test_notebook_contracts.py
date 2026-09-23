"""Cheap structural checks; do not train models or touch scientific outputs."""
import ast
import json
import os
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class NotebookContractTests(unittest.TestCase):
    def test_notebook_imports_and_dataset_setup_are_not_repeated(self):
        for path in (ROOT / 'experiments').rglob('*.ipynb'):
            with self.subTest(notebook=str(path.relative_to(ROOT))):
                notebook = nbformat.read(path, as_version=4)
                imports = set()
                paths = set()
                for cell in notebook.cells:
                    if cell.cell_type != 'code':
                        continue
                    for node in ast.parse(cell.source).body:
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            for alias in node.names:
                                key = (type(node).__name__, getattr(node, 'module', None),
                                       getattr(node, 'level', 0), alias.name, alias.asname)
                                self.assertNotIn(key, imports, f'Repeated import: {ast.unparse(node)}')
                                imports.add(key)
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id in {
                                    'PROJECT_ROOT', 'ROOT', 'DATA_ROOT', 'DATA_DIR', 'data_dir'
                                }:
                                    self.assertNotIn(target.id, paths, f'Repeated setup: {target.id}')
                                    paths.add(target.id)

    def test_layout_introductions_syntax_and_analysis_training_boundary(self):
        self.assertFalse(list((ROOT / 'experiments').glob('*.ipynb')))
        forbidden = {'train', 'train_mask_model', 'train_masking_experiment', 'train_or_load_model',
                     'train_model_grid_for_variant_packs', 'prepare_fold_graphs', 'prepare_training_graphs'}
        for path in (ROOT / 'experiments').rglob('*.ipynb'):
            with self.subTest(notebook=str(path.relative_to(ROOT))):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                self.assertEqual(notebook.cells[0].cell_type, 'markdown')
                self.assertIn('Role:', notebook.cells[0].source)
                for cell in notebook.cells:
                    if cell.cell_type != 'code':
                        continue
                    tree = ast.parse(cell.source)
                    if 'training' not in path.parts and 'archive' not in path.parts:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                                self.assertNotIn(node.func.id, forbidden)

    def test_consolidated_training_and_no_compatibility_shims(self):
        expected = {'gin_depth_training.ipynb',
                    'fate_masking_training.ipynb', 'graph_controls_training.ipynb',
                    'lineage_removal_training.ipynb'}
        self.assertEqual({p.name for p in (ROOT/'experiments/training').glob('*.ipynb')}, expected)
        self.assertEqual(
            {p.name for p in (ROOT/'experiments/ablation').glob('*.ipynb')},
            {'total_analysis.ipynb', 'size_dependent_ablation.ipynb',
             'ablation_comparison.ipynb', 'sampling_diagnostics.ipynb', 'size_ablation_viewer.ipynb'},
        )
        for obsolete in ('analysis', 'archive', 'figures'):
            self.assertFalse((ROOT/'experiments'/obsolete).exists())
        self.assertFalse(list((ROOT/'legacy').rglob('*.ipynb')))
        for path in (ROOT/'src/analysis').glob('*.py'):
            self.assertNotIn('sys.modules[__name__]', path.read_text())
        for path in (ROOT/'experiments').rglob('*.ipynb'):
            notebook = nbformat.read(path, as_version=4)
            self.assertGreaterEqual(len(notebook.cells[0].source.split()), 45, str(path))
            source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
            self.assertNotIn('globals().update(', source)

    def test_ablation_coverage_settings_are_mirrored(self):
        keys = ['SAMPLING_SCHEME','CENTER_SAMPLE_SIZE','CENTER_MIN_MARKER_COUNT',
                'CENTER_MIN_PAIR_COUNT','SAMPLING_SEED','CENTER_APPLY_NON_OVERLAP',
                'CENTER_NON_OVERLAP_GROUP_COLUMNS','CENTERS_PER_ORGANOID','HEATMAP_DISPLAY_MIN_CASES']
        settings = []
        for name in ['total_analysis','sampling_diagnostics','ablation_comparison']:
            notebook = nbformat.read(ROOT/f'experiments/ablation/{name}.ipynb',as_version=4)
            namespace = {}
            exec(notebook.cells[1].source,namespace)
            exec(notebook.cells[3].source,namespace)
            settings.append({key:namespace[key] for key in keys})
        self.assertEqual(settings[0],settings[1])
        self.assertEqual(settings[0],settings[2])
        self.assertEqual(settings[0]['SAMPLING_SCHEME'],'coverage')
        self.assertEqual(settings[0]['CENTER_MIN_PAIR_COUNT'],25)
        self.assertEqual(settings[0]['HEATMAP_DISPLAY_MIN_CASES'],20)
        notebook = nbformat.read(ROOT/'experiments/ablation/size_dependent_ablation.ipynb',as_version=4)
        namespace = {}
        exec(notebook.cells[1].source,namespace)
        exec(notebook.cells[3].source,namespace)
        self.assertEqual(namespace['SAMPLING_SCHEME'],'size_stratified')
        self.assertEqual(namespace['SAMPLING_OPTIONS'],dict(cases_per_pair_bin=25,max_cases_per_organoid=2))
        self.assertNotIn('CENTER_SAMPLE_SIZE',namespace)
        self.assertNotIn('CENTERS_PER_ORGANOID',namespace)

    def test_total_ablation_has_no_sweep_configuration_or_execution(self):
        notebook = nbformat.read(ROOT/'experiments/ablation/total_analysis.ipynb',as_version=4)
        source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
        self.assertNotIn('N_DEPENDENT_REPLACEMENT', source)
        self.assertNotIn('SWEEP_COUNTS', source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ('evaluate_fate_edit', 'replacement_distribution'):
                    self.assertNotIn('count', [kw.arg for kw in node.keywords])

    def test_bootstrap_works_from_every_notebook_directory(self):
        before = Path.cwd()
        try:
            for path in (ROOT / 'experiments').rglob('*.ipynb'):
                notebook = json.loads(path.read_text())
                os.chdir(path.parent)
                namespace = {}
                exec(''.join(notebook['cells'][1]['source']), namespace)
                self.assertEqual(namespace['PROJECT_ROOT'], ROOT)
        finally:
            os.chdir(before)

    def test_migration_targets_exist(self):
        mapping = json.loads((ROOT / 'docs/notebook_migration.json').read_text())
        for targets in mapping.values():
            for target in targets:
                self.assertTrue((ROOT / 'experiments' / target).is_file(), target)

    def test_analysis_settings_precede_definitions_and_loading(self):
        for path in (ROOT/'experiments').rglob('*.ipynb'):
            if 'training' in path.parts:
                continue
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                self.assertTrue(notebook.cells[2].source.startswith('## Analysis settings'))
                namespace = {}
                exec(notebook.cells[1].source, namespace)
                # Settings must be inspectable without loading files/checkpoints,
                # creating output folders, or executing plotting definitions.
                tree = ast.parse(notebook.cells[3].source)
                self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(tree)))
                exec(notebook.cells[3].source, namespace)


if __name__ == '__main__':
    unittest.main()
