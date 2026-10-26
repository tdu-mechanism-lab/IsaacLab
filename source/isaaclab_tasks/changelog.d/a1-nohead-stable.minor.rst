Added
^^^^^

* Added the ``IsaacContrib-Velocity-Rough-UnitreeA1DogNoheadStable`` and ``IsaacContrib-Velocity-Rough-UnitreeA1HorseNoheadStable`` tasks. They guard the headless dog and horse against simulation blow-ups (a 21 rad/s joint velocity limit, a ``physics_diverged`` termination, clipped observations and actions, and sanitized step rewards) and use the terminations, rewards, base-mass randomization and dog pose of the 3-link training run.
