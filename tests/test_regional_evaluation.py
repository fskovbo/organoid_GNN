"""Source-region alignment and equal-organoid regional validation metrics."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.analysis.spatial.regions import graph_regions
from src.analysis.metrics.evaluation import regional_mse, organoid_mse_summary
from src.plotting.depth_scan import plot_regional_mse

ROOT = Path(__file__).resolve().parents[1]


class RegionalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.d = np.array([[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.,1.1,1.2,1.3,1.5]])
        self.graph = SimpleNamespace(organoid_str='org',x=np.ones((14,2)),
                                     meta={'segmentation_path':str(self.root/'segmentation.npz')})
        self.axis = np.linspace(.2,1.8,81)
        self.profile = 1+.5*(self.axis-1)**2
        self.write_sources()

    def write_sources(self, distances=None, profiles=None, mesh_distances=None):
        distances = self.d if distances is None else distances
        profiles = np.array([self.profile]) if profiles is None else profiles
        np.savez(self.root/'org.npz',x=self.graph.x,d_crypts_graph=distances,proj_vertex_ids=np.arange(14))
        np.savez(self.root/'segmentation.npz',d_crypts=distances if mesh_distances is None else mesh_distances,
                 d_discretized=self.axis,circumference_crypts=profiles)

    def test_regions_boundaries_and_nearest_crypt(self):
        labels = graph_regions(self.graph,self.root)
        self.assertEqual(labels.region.tolist(),['crypt']*7+['neck']*5+['villus']*2)
        self.assertTrue((labels.profile_class=='local_minimum').all())
        # A closer unsupported crypt must not be bypassed in favor of a supported one.
        distances = np.concatenate([self.d,self.d+.1])
        self.write_sources(distances,np.array([self.axis+1,self.profile]))
        labels = graph_regions(self.graph,self.root,neck_config={'minimum_crypt_cells':0})
        self.assertTrue((labels.region=='unqualified_crypt').all())
        self.assertTrue((labels.crypt_id==0).all())

    def test_missing_no_crypt_low_support_and_bad_alignment(self):
        labels = graph_regions(self.graph,self.root,neck_config={'minimum_crypt_cells':11})
        self.assertTrue((labels.region=='unqualified_crypt').all())
        self.assertTrue((labels.profile_class=='low_cell_support').all())
        self.write_sources(np.empty((0,14)),np.empty((0,81)))
        self.assertTrue((graph_regions(self.graph,self.root).region=='no_detected_crypt').all())
        self.write_sources(mesh_distances=self.d+.2)
        with self.assertRaisesRegex(ValueError,'alignment mismatch'):
            graph_regions(self.graph,self.root)
        (self.root/'segmentation.npz').unlink()
        missing = graph_regions(self.graph,self.root)
        self.assertTrue((missing.region=='annotation_unavailable').all())
        self.assertTrue((missing.annotation_status=='missing_segmentation_file').all())
        self.write_sources(distances=np.where(np.arange(14)[None,:]==3,np.nan,self.d))
        self.assertEqual(graph_regions(self.graph,self.root).region.iloc[3],'annotation_unavailable')
        (self.root/'org.npz').unlink()
        self.assertTrue((graph_regions(self.graph,self.root).annotation_status=='missing_graph_source').all())

    def test_regional_mse_uses_node_keys_and_equal_organoid_weights(self):
        cells = pd.DataFrame(dict(organoid_str=['a','a','a','b'],node=[0,1,2,0],
            squared_error=[1.,9.,16.,25.],baseline_squared_error=[4.,4.,4.,36.]))
        labels = cells[['organoid_str','node']].assign(region=['crypt','crypt','neck','crypt']).iloc[::-1]
        scores = regional_mse(cells,labels)
        crypt = scores[scores.region=='crypt'].assign(depth=2,name='gin',seed=1)
        self.assertEqual(crypt.mse.tolist(),[5.,25.])
        repeated = pd.concat([crypt,crypt.assign(seed=2)])
        summary = organoid_mse_summary(repeated,'depth').iloc[0]
        self.assertEqual(summary['count'],2)
        self.assertEqual(summary['mean'],15.)
        self.assertAlmostEqual(summary['sem'],10.)
        with self.assertRaisesRegex(ValueError,'Missing region'):
            regional_mse(cells,labels.iloc[1:])
        with self.assertRaisesRegex(ValueError,'unique'):
            regional_mse(cells,pd.concat([labels,labels]))
        fig = plot_regional_mse(repeated)
        self.assertEqual(fig.axes[0].lines[0].get_xdata().tolist(),[2])
        self.assertEqual(fig.axes[0].lines[0].get_ydata().tolist(),[15.])
        self.assertEqual(len(fig.axes[1].lines),0)
        plt.close(fig)

    def test_training_notebooks_own_evaluation_and_old_notebook_removed(self):
        self.assertFalse((ROOT/'experiments/neighborhoods/region_predictability.ipynb').exists())
        for path in (ROOT/'experiments/training').glob('*.ipynb'):
            notebook = json.loads(path.read_text())
            source = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
            self.assertIn('REGION_SETTINGS = dict(',source)
            self.assertIn('regional_run.records.to_dict',source)
            self.assertIn('graph_regions(raw_graph, region_data_dir',source)
            self.assertIn("region_output / 'node_regions.csv.gz'",source)
            self.assertNotIn('exclusive_v2_profile_necks',source)


if __name__=='__main__':unittest.main()
