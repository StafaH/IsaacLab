Added
^^^^^

* Added :func:`~isaaclab.managers.observation_output_owned` for observation implementations that
  transfer independent outputs to their callers. The observation pipeline used this guarantee
  to avoid redundant copies automatically; task configurations required no copy setting.

Changed
^^^^^^^

* Made observation clipping and scaling allocate only when processing borrowed storage, and
  returned independent snapshots for single-term groups, dictionary outputs, and history.
* Created ``image_features`` normalization statistics once instead of on every inference call.
