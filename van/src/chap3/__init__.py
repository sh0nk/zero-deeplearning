def suite():
  from . import activation_functions
  from . import network
  from . import neuralnet_mnist

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(activation_functions))
  suite.addTest(doctest.DocTestSuite(network))
  suite.addTest(doctest.DocTestSuite(neuralnet_mnist))
  return suite

