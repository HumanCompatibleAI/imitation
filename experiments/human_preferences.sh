python -m imitation.scripts.train_preference_comparisons \
  with \
  pendulum \
  human_preferences \
  total_comparisons=5000 \
  total_timesteps=1000000 \
  gatherer_kwargs.pref_collect_address=127.0.0.1:8000 \
  gatherer_kwargs.video_output_dir=../pref-collect/videofiles \
  gatherer_kwargs.wait_for_user=True \
  common.post_wrappers_kwargs.RenderImageInfoWrapper.scale_factor=0.5 \
  common.post_wrappers_kwargs.RenderImageInfoWrapper.use_file_cache=True \