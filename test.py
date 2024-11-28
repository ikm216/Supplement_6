import numpy as np

def test_should_return_specified_shape_array():
    shape = (10,10)
    mean = 5
    stan_dev = 2
    shape_arr = shape_array(shape, mean, stan_dev)

    assert shape_arr.shape == shape
    assert abs(np.mean(shape_arr) - mean) < 0.5
    assert abs(np.stan_dev(shape_arr) - stan_dev) < 0.5

