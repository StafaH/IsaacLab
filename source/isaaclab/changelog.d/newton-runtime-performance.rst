Fixed
^^^^^

* Reduced camera synchronization overhead by avoiding device mask readbacks when the full camera batch was already known to require an update.
* Removed device-to-host synchronization from velocity-command heading and standing-mask updates.
* Reused the environment reset selection for termination bookkeeping instead of compacting the same device mask twice.
* Cached resolved body selectors on the scene device and used fixed-shape joint masks in common reward reductions to avoid per-step host/device synchronization.
* Resolved local and remote neural-actuator checkpoints consistently during Newton schema authoring.
