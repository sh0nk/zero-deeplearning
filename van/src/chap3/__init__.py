from . import activation_functions

def suite():
  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(activation_functions))
  return suite

