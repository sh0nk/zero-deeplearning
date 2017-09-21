def suite():
  from . import optimizers

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(optimizers))
  return suite

