import gym

from RayEnvWrapper import WrapperRayVecEnv

def make_and_seed(seed: int) -> gym.Env:
    env = gym.make('CartPole-v0')
    env.seed(seed)
    return env

if __name__ == '__main__':
    vec_env = WrapperRayVecEnv(make_and_seed, 4, 2)
    print(vec_env.observation_space)
    print(vec_env.action_space)
    print(vec_env.reset())
    print(vec_env.step([vec_env.action_space.sample() for _ in range(8)]))
