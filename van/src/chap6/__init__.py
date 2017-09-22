def suite():
  from . import optimizers
  from . import multi_layer_net
  from . import dropout

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(optimizers))
  suite.addTest(doctest.DocTestSuite(multi_layer_net))
  suite.addTest(doctest.DocTestSuite(dropout))
  return suite

