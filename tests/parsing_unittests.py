# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 14:02:59 2022

@author: pyprg
"""
import unittest
import src.graphparser.parsing as parsing

class Parsing(unittest.TestCase):
    
    def test_1_node(self):
        d = 'node'
        ex = [('node', 'node', (), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")
    
    def test_2_nodes_1_edge(self):
        d = '0 1'
        ex = [
            ('node', '0', ('1',), {}), 
            ('edge', ('0', '1'), {}), 
            ('node', '1', ('0',), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")
    
    def test_2_nodes(self):
        d = """
            A
            
            B"""
        ex = [
            ('node', 'A', (), {}), 
            ('node', 'B', (), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")
    
    def test_2_nodes_valueerror(self):
        d = """
            A
            B"""
        with self.assertRaises(ValueError):
            [*parsing.parse(d)]
    
    def test_2_nodes_att_valueerror(self):
        d = """
            A
                  att=value"
            B"""
        with self.assertRaises(ValueError):
            [*parsing.parse(d)]
    
    def test_2_nodes_comment_valueerror(self):
        d = ("A           \n"
             "# comment   \n"
             "B           ")
        with self.assertRaises(ValueError):
            [*parsing.parse(d)]
    
    def test_2_nodes_no_edge(self):
        d0 = '0_ 1'
        d1 = '0 _1'
        d2 = '0_ _1'
        ex = [
            ('node', '0', (), {}), 
            ('node', '1', (), {})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d2)], ex, f"expected {ex}")
    
    def test_3_nodes_2_edges(self):
        d = '0 1 2'
        ex = [
            ('node', '0', ('1',), {}), 
            ('edge', ('0', '1'), {}), 
            ('node', '1', ('0', '2'), {}),
            ('edge', ('1', '2'), {}), 
            ('node', '2', ('1',), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")

    def test_blank(self):
        d = '^°"§$%&/()=?`´\}][{+~@€*;,:.-<>|'
        ex = []
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")

    def test_comment(self):
        d = '# this is a comment'
        ex = []
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")

    def test_ignored_att(self):
        d = 'a=42'
        ex = []
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")
    
    def test_node_ignored_att_above(self):
        d = \
        """
              att=42
        mynode
        """
        ex = [('node', 'mynode', (), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")
    
    def test_node_att_above(self):
        d0 =  """
                att=42
                mynode"""
        d1 = """
                 att=42
                mynode"""
        d2 = """
                     att=42
                mynode"""
        ex = [('node', 'mynode', (), {'att': '42'})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d2)], ex, f"expected {ex}")
    
    def test_node_ignored_att_below(self):
        d = \
        """
            mynode
                  att=42
        """
        ex = [('node', 'mynode', (), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")
    
    def test_node_att_below(self):
        d0 = """
            mynode
            att=42e-6+27j"""
        d1 = """
            mynode
              att=42e-6+27j"""
        d2 = """
            mynode
                 att=42e-6+27j"""
        ex = [('node', 'mynode', (), {'att': '42e-6+27j'})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d2)], ex, f"expected {ex}")
    
    def test_edge_att(self):
        d0 =  """
              att=b
            a==b"""
        d1 = """
            a==b
              att=b"""
        ex = [('node', 'a', ('b',), {}),
              ('edge', ('a', 'b'), {'att': 'b'}),
              ('node', 'b', ('a',), {})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")
    
    def test_node_and_edge_att(self):
        d0 =  """
             g=h
            a==b
            c=d
               e=f"""
        d1 = """
            c=d
               e=f
            a==b
             g=h"""
        ex = [('node', 'a', ('b',), {'c':'d'}),
              ('edge', ('a', 'b'), {'g': 'h'}),
              ('node', 'b', ('a',), {'e':'f'})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")

    def test_underscore(self):
        d0 = '_'
        d1 = '__'
        ex = [('node', '', (), {})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")

    def test_underscore_underscore(self):
        d0 = '_ _'
        d1 = '__ __'
        ex = [('node', '', (), {}), ('node', '', (), {})]
        self.assertEqual([*parsing.parse(d0)], ex, f"expected {ex}")
        self.assertEqual([*parsing.parse(d1)], ex, f"expected {ex}")

    def test_3_underscores(self):
        d = '___'
        ex = [('node', '_', (), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")

    def test_3_underscores_3_underscores(self):
        d = '___ ___'
        ex = [('node', '_', (), {}), ('node', '_', (), {})]
        self.assertEqual([*parsing.parse(d)], ex, f"expected {ex}")

if __name__ == '__main__':
    unittest.main()
    