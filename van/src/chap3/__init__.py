def suite():
  from . import activation_functions
  from . import network

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(activation_functions))
  suite.addTest(doctest.DocTestSuite(network))
  return suite

