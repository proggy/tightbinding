# To Do

## Define

- Implement V-J Model (class _VJ)


## sc

- supercell itself may have (anti)periodic boundary conditions (until now,
  just lattices have boundary conditions)
- improve Lattice.ndindex2index (increase performance even more by using a
  lookup table)
- add special algorithms for special cases (e.g., Lattice.scnnmat)
- add automatic performance checks (compare times for finding the
  connections to the setting of matrix elements, random and constant)
- rename "dim" to "ndim", be consistent with numpy.ndarray"""
- SuperCell.add_triang(): bvects should be chosen to get lattice with equilateral triangles
- UnitCell.add_hop(): Should check if both entities are of the same site. If so, hand task off to Site.add_hop().
- Site.add_hop(): What will happen if hop is random and iso is True?
- scnnmat(): improve description of working principle
- SparseLattice.coords(): implement the method
- SparseLattice.ndindex2index: The look-up table could better be calculated once, and not be re-calculated for every vector, as it is universal.


## sc.dist

- define even more mathematical operators and functions (e.g. log etc.)
- define even more more probability distributions
- scale the triangular distribution to make it comparable to the others
  (see the paper by Thouless...)
- allow other values for the c parameter of the triangular distribution
  (asymmetric distributions)"""
- triang(): scale the distribution to make it comparable to the others
      (see the paper by Thouless...)
- triang(): allow other values for the c parameter (asymmetric distributions)
- a copy counter should be offered in all distribution classes (how many copies of the same random values have been delivered)


## sc.pos

- PosRule(): is there anything extra to be done? Or is it just the same as a distribution object?
- spheres(): allow spheres to overlap boundaries (with periodic boundary conditions)?


## sc.misc

- types(): should be enabled for any iterable, also for set and frozenset