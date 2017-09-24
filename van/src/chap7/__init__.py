def suite():
  from . import layers

  import doctest
  import unittest
  suite = unittest.TestSuite()
  suite.addTest(doctest.DocTestSuite(layers))
  return suite

