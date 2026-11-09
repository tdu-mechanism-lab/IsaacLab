Added
^^^^^

* Added the ``IsaacContrib-Velocity-Rough-UnitreeA1DogNoheadPaper`` task: the ``DogNoheadLimits`` task with the joint
  limits and the default pose of the headed dog.

Fixed
^^^^^

* Fixed the per-joint weights of the ``feet_torque`` penalty of the dog and horse tasks, which named ``*_foot_joint``
  instead of ``*_calf2_joint``, so that the ``calf2`` torques were weighted like the hip torques (1.0 instead of 0.1).
