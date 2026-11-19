* Added :data:`~isaaclab_experimental.mdp2.mdp.image_rgb_channel_first`, a camera's ``uint8`` image of shape
  ``(3, height, width)``, and concatenated groups of one term of another type than ``float32``, which are that term's
  buffer with its shape and type, as MDP1 keeps the term of a group of one term: such a group reaches RSL-RL as an
  image.
