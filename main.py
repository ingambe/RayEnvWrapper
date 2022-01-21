import gym

from RayEnvWrapper import WrapperRayVecEnv

if __name__ == '__main__':
    vec_env = WrapperRayVecEnv(lambda x: gym.make('CartPole-v0'), 8, 1)
    print(vec_env.reset())
    print(vec_env.step([0 for _ in range(8)]))
