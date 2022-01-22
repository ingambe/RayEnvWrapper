import gym
from RayEnvWrapper import WrapperRayVecEnv

def make_and_seed(seed: int) -> gym.Env:
    env = gym.make('CartPole-v0')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    return env

if __name__ == '__main__':
    number_of_workers = 4  # Usually, this is set to the number of CPUs in your machine
    envs_per_worker = 2

    vec_env = WrapperRayVecEnv(make_and_seed, number_of_workers, envs_per_worker)
    # RESET all environment
    vec_env.reset()
    print(vec_env.step([vec_env.action_space.sample() for _ in range(number_of_workers * envs_per_worker)]))