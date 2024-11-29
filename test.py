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

def cramers_rule(coe_matrix, conts):
    """
    Solves a system of linear equations using Cramer's Rule formula.

    Args:
        coe_matrix: Coefficient matrix.
        conts: Constants.

    Returns:
        The answer of the system of linear equations using Cramer's Rule formula.
    """
    determinant = np.linalg.det(coe_matrix) #np.linalg.det: gets the determinant of coefficients 

    num_vals = coe_matrix.shape[1]
    answer = []

    for i in range(num_vals):
        # Calculating the answer using the cramers rule
        temp = coe_matrix.copy()
        temp[:, i] = conts
        answer.append(np.linalg.det(temp) / determinant)
    
    return answer

def shape_array_even_odd(shape, even, odd):
    even_arr = np.full(shape, even)
    odd_arr = np.full(shape, odd)

    mask = np.indices(shape).sum(axis=0) % 2 == 0

    arr2 = np.where(mask, even_arr, odd_arr)

    even_index = np.argwhere(mask)
    odd_index = np.argwhere(~mask)

    return arr2, even_index, odd_index

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
    # System equation
    # 5x + 3y = 7
    # 3x - 5y = -23

    correct_answer = np.array([-1, 4])
    # x = -1, y = 4

    answer = cramers_rule(coe_matrix, conts)

    assert np.allclose(answer, correct_answer)

def test_should_return_indexes_even_odd_numbers_separate_lists_newly_generated_array():
    shape = (10,10)
    even = 2
    odd = 1

    arr, even_index, odd_index = shape_array_even_odd(shape, even, odd)

    assert arr.shape == shape
    assert all(arr[tuple(idx)] == even for idx in even_index)
    assert all(arr[tuple(idx)] == odd for idx in odd_index)
    for idx in even_index:
        assert arr[tuple(idx)] % 2 == 0
    for idx in odd_index:
        assert arr[tuple(idx)] % 2 != 0
    
