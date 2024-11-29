import numpy as np

def shape_array(shape, mean, stan_dev):
    return np.full(shape, mean) + stan_dev

def test_should_return_specified_shape_array():
    shape = (10,10)
    mean = 5
    stan_dev = 2
    shape_arr = shape_array(shape, mean, stan_dev)

    assert shape_arr.shape == shape
    assert np.all(shape_arr == mean + stan_dev)

