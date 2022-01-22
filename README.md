# Ray Vector Environment Wrapper

You would like to use Ray to vectorize your environment but you don't want to use RLLib ?  
You came to the right place !

This package allows you to parallelize your environment using [Ray](https://github.com/ray-project/ray)  
Not only does it allows to run environments in parallel, but it also permits to run multiple sequential environments on each worker  
For example, you can run 80 workers in parallel, each running 10 sequential environments for a total of 80 * 10 environments  
This can be useful if your environment is fast and simply running 1 environment per worker leads to too much communication overhead between workers

## Installation

````zsh
pip install RayEnvWrapper
````

If something went wrong, it most certainly is because of [Ray](https://github.com/ray-project/ray)  
For example, you might have issue installing Ray on Apple Silicon (i.e., M1) laptop. See [Ray's documentation](https://docs.ray.io/en/latest/installation.html#apple-silicon-support) for a simple fix  
**At the moment Ray does not support Python 3.10**.
This package has been tested with Python 3.9.

## How does it work?

You first need to define a function that seed and return your environment:

Here is an example for CartPole:
````python
import gym

def make_and_seed(seed: int) -> gym.Env:
    env = gym.make('CartPole-v0')
    env = gym.wrappers.RecordEpisodeStatistics(env) # you can put extra wrapper to your original environment
    env.seed(seed)
    return env
````

**Note**: If you don't want to seed your environment, simply return it without using the seed, but the function you define needs to take a number as an input

Then, call the wrapper to create and wrap all the vectorized environment:

````python
from RayEnvWrapper import WrapperRayVecEnv

number_of_workers = 4 # Usually, this is set to the number of CPUs in your machine
envs_per_worker = 2

vec_env = WrapperRayVecEnv(make_and_seed, number_of_workers, envs_per_worker)
````

You can then use your environment.
All the output for each of the environments are stacked in a numpy array

**Reset:**

````python
vec_env.reset()
````
Output

````python
[[ 0.03073904  0.00145001 -0.03088818 -0.03131252]
 [ 0.03073904  0.00145001 -0.03088818 -0.03131252]
 [ 0.02281231 -0.02475473  0.02306162  0.02072129]
 [ 0.02281231 -0.02475473  0.02306162  0.02072129]
 [-0.03742824 -0.02316945  0.0148571   0.0296055 ]
 [-0.03742824 -0.02316945  0.0148571   0.0296055 ]
 [-0.0224773   0.04186813 -0.01038048  0.03759079]
 [-0.0224773   0.04186813 -0.01038048  0.03759079]]
````

The i-th entry represent the initial observation of the i-th environment  
**Note**: As environments are vectorized, you don't need explicitly to reset the environment at the end of the episode, **it is done automatically**
However, you need to do it once at the beginning

**Take a random action:**

````python
vec_env.step([vec_env.action_space.sample() for _ in range(number_of_workers * envs_per_worker)])
````

Notice how the actions are passed. We pass an array containing an action for each of the environments  
Thus, the array is of size `number_of_workers * envs_per_worker` (i.e., the total number of environments)

Output

````python
(array([[ 0.03076804, -0.19321568, -0.03151444,  0.25146705],
       [ 0.03076804, -0.19321568, -0.03151444,  0.25146705],
       [ 0.02231721, -0.22019969,  0.02347605,  0.3205903 ],
       [ 0.02231721, -0.22019969,  0.02347605,  0.3205903 ],
       [-0.03789163, -0.21850128,  0.01544921,  0.32693872],
       [-0.03789163, -0.21850128,  0.01544921,  0.32693872],
       [-0.02163994, -0.15310344, -0.00962866,  0.3269806 ],
       [-0.02163994, -0.15310344, -0.00962866,  0.3269806 ]],
      dtype=float32), 
 array([1., 1., 1., 1., 1., 1., 1., 1.], dtype=float32), 
 array([False, False, False, False, False, False, False, False]), 
 [{}, {}, {}, {}, {}, {}, {}, {}])
````

As usual, the `step` method returns a tuple, except that here both the observation, reward, dones and infos are concatenated  
In this specific example, we have 2 environments per worker.  
Index 0 and 1 are environments from worker 1; index 1 and 2 are environments from worker 2, etc.

# License

Apache License 2.0