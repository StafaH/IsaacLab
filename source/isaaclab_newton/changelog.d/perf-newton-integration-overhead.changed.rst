* Reduced Newton stepping overhead by avoiding redundant body-force copies when
  no actuator callbacks could modify the forces before the first solver substep.
* Combined articulation joint and body acceleration history updates into one
  kernel while preserving their sampling intervals and public body ordering.
