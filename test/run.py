import unittest
import test_criterion

suite =unittest.TestSuite()

suite.addTest(unittest.makeSuite(test_criterion.my_test))

runner=unittest.TextTestRunner()
runner.run(suite)  
