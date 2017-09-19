def suite():
  from . import loss_functions

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(loss_functions))
  return suite

