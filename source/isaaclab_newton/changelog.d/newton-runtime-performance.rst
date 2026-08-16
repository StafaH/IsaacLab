Fixed
^^^^^

* Reduced Newton MJWarp and renderer runtime overhead by removing duplicate solver reconciliation and scene-state refreshes, and by skipping reset/FK reconciliation when no state was authored.
* Initialized actuator velocity-limit metadata when using Newton-native actuators.
* Added a backend-scoped Newton actuator setting for multi-backend presets.
