import unittest
import chap2

def all_suite():
  suite = unittest.TestSuite()
  suite.addTests(chap2.suite())
  #suite.addTest(doctest.TestSuite(xxxxxx))
  return suite

