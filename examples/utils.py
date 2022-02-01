def render_a_trajectory_and_print_reward(env, policy):
    obs = env.reset()
    done = False
    cumulative_reward = 0
    env.render()
    while not done:
        action, _ = policy.predict(obs)
        obs, reward, done, _ = env.step(action)
        cumulative_reward += reward
        env.render()
    env.close()
    print(f"Cumulative Reward: {cumulative_reward}")