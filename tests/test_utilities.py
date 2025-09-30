import numpy as np
import pytest

from aot_analysis.utilities import hypersphere_center_of_mass


class TestHypersphereCenterOfMass:
    """Test the hypersphere_center_of_mass function."""

    def test_three_uniform_particles_no_pbc(self):
        """Test COM calculation for three uniformly separated particles without PBC crossing."""
        # Box dimensions
        box_dimensions = np.array([10.0, 10.0, 10.0])

        # Separation distance
        a = 1.0

        # Test displacement that keeps all particles well within the box
        displacement = np.array([5.0, 5.0, 5.0])

        # Create three uniformly separated particles: [-a, 0, a] + displacement
        positions = np.array([[-a, 0, a], [0, 0, 0], [a, 0, -a]]) + displacement

        # Calculate center of mass
        com = hypersphere_center_of_mass(positions, box_dimensions)

        # The COM should be at the displacement (since the particles are symmetric around origin)
        expected_com = displacement

        np.testing.assert_allclose(com, expected_com, atol=1e-10)

    def test_three_uniform_particles_with_pbc_crossing(self):
        """Test COM calculation when particles cross periodic boundaries."""
        # Box dimensions
        box_dimensions = np.array([10.0, 10.0, 10.0])

        # Separation distance
        a = 1.5

        # Displacement that causes particles to cross boundaries
        displacement = np.array([9.5, 1.0, 8.5])

        # Create three uniformly separated particles: [-a, 0, a] + displacement
        base_positions = np.array([[-a, 0, a], [0, 0, 0], [a, 0, -a]])

        positions = base_positions + displacement

        # Apply periodic boundary conditions to positions that are outside the box
        positions = np.mod(positions, box_dimensions)

        # Calculate center of mass
        com = hypersphere_center_of_mass(positions, box_dimensions)

        # The expected COM should still be at the displacement (wrapped to box)
        expected_com = np.mod(displacement, box_dimensions)

        np.testing.assert_allclose(com, expected_com, atol=1e-8)

    def test_three_uniform_particles_multiple_displacements(self):
        """Test COM calculation for multiple random displacements."""
        # Box dimensions
        box_dimensions = np.array([10.0, 10.0, 10.0])

        # Separation distance
        a = 1.0

        # Base particle configuration: [-a, 0, a] in x, all at origin in y, [a, 0, -a] in z
        base_positions = np.array([[-a, 0, a], [0, 0, 0], [a, 0, -a]])

        # Test with multiple random displacements
        np.random.seed(42)  # For reproducible tests
        for _ in range(10):
            displacement = np.random.uniform(0, 10, 3)

            positions = base_positions + displacement
            positions = np.mod(positions, box_dimensions)

            com = hypersphere_center_of_mass(positions, box_dimensions)
            expected_com = np.mod(displacement, box_dimensions)

            np.testing.assert_allclose(com, expected_com, atol=1e-8)

    def test_three_uniform_particles_with_masses(self):
        """Test COM calculation with non-uniform masses."""
        # Box dimensions
        box_dimensions = np.array([10.0, 10.0, 10.0])

        # Separation distance
        a = 1.0

        # Test displacement
        displacement = np.array([5.0, 5.0, 5.0])

        # Create three uniformly separated particles
        base_positions = np.array([[-a, 0, a], [0, 0, 0], [a, 0, -a]])

        positions = base_positions + displacement

        # Test with equal masses (should give same result as no masses)
        masses = np.array([1.0, 1.0, 1.0])
        com_equal_masses = hypersphere_center_of_mass(positions, box_dimensions, masses)
        com_no_masses = hypersphere_center_of_mass(positions, box_dimensions)

        np.testing.assert_allclose(com_equal_masses, com_no_masses, atol=1e-10)

        # The COM should still be at the displacement for equal masses
        np.testing.assert_allclose(com_equal_masses, displacement, atol=1e-10)

    def test_three_uniform_particles_different_separations(self):
        """Test COM calculation with different separation distances."""
        # Box dimensions
        box_dimensions = np.array([20.0, 20.0, 20.0])

        # Test displacement
        displacement = np.array([10.0, 10.0, 10.0])

        # Test with different separation distances
        separations = [0.5, 1.0, 2.0, 3.0, 5.0]

        for a in separations:
            base_positions = np.array([[-a, 0, a], [0, 0, 0], [a, 0, -a]])

            positions = base_positions + displacement

            com = hypersphere_center_of_mass(positions, box_dimensions)

            # The COM should always be at the displacement regardless of separation
            np.testing.assert_allclose(com, displacement, atol=1e-10)

    def test_three_uniform_particles_edge_cases_pbc(self):
        """Test specific edge cases where particles are near box boundaries."""
        # Box dimensions
        box_dimensions = np.array([10.0, 10.0, 10.0])

        # Separation distance
        a = 0.5

        # Case 1: Displacement puts one particle exactly on the boundary
        displacement = np.array([0.5, 5.0, 9.5])

        base_positions = np.array([[-a, 0, a], [0, 0, 0], [a, 0, -a]])

        positions = base_positions + displacement
        positions = np.mod(positions, box_dimensions)

        com = hypersphere_center_of_mass(positions, box_dimensions)
        expected_com = np.mod(displacement, box_dimensions)

        np.testing.assert_allclose(com, expected_com, atol=1e-8)

        # Case 2: Displacement causes wrapping in all dimensions
        displacement = np.array([9.8, 9.9, 9.7])

        positions = base_positions + displacement
        positions = np.mod(positions, box_dimensions)

        com = hypersphere_center_of_mass(positions, box_dimensions)
        expected_com = np.mod(displacement, box_dimensions)

        np.testing.assert_allclose(com, expected_com, atol=1e-8)

    def test_single_particle(self):
        """Test COM calculation for a single particle."""
        box_dimensions = np.array([10.0, 10.0, 10.0])
        position = np.array([[5.0, 3.0, 7.0]])

        com = hypersphere_center_of_mass(position, box_dimensions)

        np.testing.assert_allclose(com, position[0], atol=1e-10)

    def test_two_particles_symmetric(self):
        """Test COM calculation for two symmetric particles."""
        box_dimensions = np.array([10.0, 10.0, 10.0])
        center = np.array([5.0, 5.0, 5.0])
        offset = np.array([1.0, 0.0, 0.0])

        positions = np.array([center - offset, center + offset])

        com = hypersphere_center_of_mass(positions, box_dimensions)

        np.testing.assert_allclose(com, center, atol=1e-10)

    def test_two_particles_across_boundary(self):
        """Test COM calculation for two particles across periodic boundary."""
        box_dimensions = np.array([10.0, 10.0, 10.0])

        # Place particles at x=0.5 and x=9.5 (should have COM at x=0 or x=10, both equivalent)
        positions = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]])

        com = hypersphere_center_of_mass(positions, box_dimensions)

        # The COM should be at either 0 or 10 in x (both equivalent due to PBC)
        # and exactly at 5 in y and z
        expected_x = 0.0  # or could be 10.0 due to PBC
        np.testing.assert_allclose(com[1:], [5.0, 5.0], atol=1e-10)
        # Allow for both 0 and 10 (equivalent under PBC)
        assert abs(com[0] - expected_x) < 1e-8 or abs(com[0] - 10.0) < 1e-8
