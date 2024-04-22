import numpy as np
from aot.cluster import load_sparse, save_sparse
from scipy.sparse import coo_array


def test_save_sparse(tmp_path):
    # Create some sparse arrays for testing
    arr1 = coo_array([[1, 0, 2], [0, 3, 0]])
    arr2 = coo_array([[4, 0, 5], [0, 6, 0]])

    tmp_file = tmp_path / "test_sparse_arrays.npz"
    # Save the sparse arrays to a file
    save_sparse([arr1, arr2], tmp_file)

    # Load the saved arrays
    loaded_arrays = load_sparse(tmp_file)
    assert np.allclose(loaded_arrays[0].toarray(), arr1.toarray())
    assert np.allclose(loaded_arrays[1].toarray(), arr2.toarray())
