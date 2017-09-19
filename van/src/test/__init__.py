import unittest
import chap2
import chap3
import chap4

def all_suite():

  suite = unittest.TestSuite()
  suite.addTests(chap2.suite())
  suite.addTests(chap3.suite())
  suite.addTests(chap4.suite())
  #suite.addTest(doctest.TestSuite(xxxxxx))
  return suite

