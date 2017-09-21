def suite():
  from . import layer_native
  from . import two_layer_net

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(layer_native))
  suite.addTest(doctest.DocTestSuite(two_layer_net))
  return suite

