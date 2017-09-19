def suite():
  from . import loss_functions
  from . import numerical

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(loss_functions))
  suite.addTest(doctest.DocTestSuite(numerical))
  return suite

