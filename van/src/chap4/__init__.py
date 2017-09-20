def suite():
  from . import loss_functions
  from . import numerical
  from . import simplenet
  from . import gradient

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(loss_functions))
  suite.addTest(doctest.DocTestSuite(numerical))
  suite.addTest(doctest.DocTestSuite(simplenet))
  suite.addTest(doctest.DocTestSuite(gradient))
  return suite

