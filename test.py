import numpy as np

def shape_array(shape, mean, stan_dev):
    """
    Creates an array filled with specified value of mean and standard deviation.

    Args:
        shape: Shape of the array.
        mean: Value to fill the array with.
        stan_dev: Standard deviation.

    Returns:
        Array filled with values following mean and simulated deviation.
    """
    return np.full(shape, mean) + stan_dev #np.full: creates the shape array with the specified value



def test_should_return_specified_shape_array():
    shape = (10,10)
    mean = 5
    stan_dev = 2
    shape_arr = shape_array(shape, mean, stan_dev)

    assert shape_arr.shape == shape
    assert np.all(shape_arr == mean + stan_dev) #np.all: tests all values in the array evaulate to true with a specified axis 

def test_should_return_negative1_4_from_cramers_rule():
    coe_matrix = np.array([[5, 3], [3, -5]])
    conts = np.array([7, -23])

    correct_answer = np.array([-1, 4])
    answer = cramers_rule(coe_matrix, conts)

    assert np.allclose(answer, correct_answer)
    
