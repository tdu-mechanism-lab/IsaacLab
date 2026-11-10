Added
^^^^^

* Added the ``IsaacContrib-Velocity-Rough-UnitreeA1DogNoheadArm`` and
  ``IsaacContrib-Velocity-Rough-UnitreeA1DogNoheadPaperArm`` tasks: the ``DogNoheadLimits`` and ``DogNoheadPaper``
  tasks with a joint armature of 0.01 kg m^2. Without it, the explicit PD control of the light ``calf`` and ``calf2``
  links is numerically unstable and the joints oscillate at the velocity limit while the robot stands.
