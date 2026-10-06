import unittest
import numpy as np
import networkx as nx
from src.data.marker_restoration import restore_marker_channels


class RestorationTests(unittest.TestCase):
    def graph(self):
        g=nx.Graph();g.graph['marker_names']=['Agr2','Mucin 2','Glucagon']
        for i,v in [(0,[1,1,0]),(3,[0,0,1]),(9,[0,0,0])]:g.add_node(i,markers_bin=v,proj_vertex=i+100)
        g.add_edges_from([(0,3),(3,9)])
        return g

    def test_reordered_pruned_cells_and_reapplied_rules(self):
        g=self.graph();x=np.array([[0],[1]],dtype=np.float32);before=x.copy()
        y,names,available=restore_marker_channels(x,['Agr2'],dict(kept_node_ids=[3,0],proj_vertex_ids=[103,100]),['Mucin 2','Glucagon'],edges=[[1,0]],graph=g)
        np.testing.assert_array_equal(y,[[0,0,1],[0,1,0]])
        np.testing.assert_array_equal(x,before)
        self.assertEqual(g.nodes[0]['markers_bin'],[1,1,0])
        self.assertTrue(all(available.values()))
        self.assertEqual(names,['Agr2','Mucin 2','Glucagon'])

    def test_missing_channel_is_not_measured_negative(self):
        g=nx.Graph();g.graph['marker_names']=['Agr2'];g.add_node(0,markers_bin=[0])
        y,_,available=restore_marker_channels([[0]],['Agr2'],dict(kept_node_ids=[0]),['Mucin 2','Glucagon'],graph=g)
        np.testing.assert_array_equal(y,[[0,0,0]])
        self.assertEqual(available,{'Mucin 2':False,'Glucagon':False})

    def test_mapping_and_marker_mismatch_fail(self):
        with self.assertRaisesRegex(ValueError,'cell IDs'):
            restore_marker_channels([[1]],['Agr2'],dict(kept_node_ids=[99]),['Mucin 2'],graph=self.graph())
        with self.assertRaisesRegex(ValueError,'Existing marker'):
            restore_marker_channels([[0]],['Agr2'],dict(kept_node_ids=[0]),['Mucin 2'],graph=self.graph())
        with self.assertRaisesRegex(ValueError,'edges disagree'):
            restore_marker_channels([[1],[0]],['Agr2'],dict(kept_node_ids=[0,3]),['Mucin 2'],edges=[],graph=self.graph())


if __name__=='__main__':unittest.main()
