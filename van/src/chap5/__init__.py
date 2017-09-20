def suite():
  from . import layer_native

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(layer_native))
  return suite

