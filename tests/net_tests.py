from dopamine.basics import *
from mathtools.fit import *
from nose.tools import assert_equals, assert_almost_equals, assert_raises
from nose import with_setup
from numpy.testing import assert_array_almost_equal_nulp, assert_array_equal
import numpy as np


# SETUP -----------------------------------------------------

def setup():
    pass

def teardown():
    pass


# BEGIN TESTS ------------------------------------------------------

def create_gaussian_density_test(self):
    gauss = DiagGaussian(5)
    assert_equal(gauss.D, 5)


def diag_loglikelihood
