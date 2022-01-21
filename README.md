# Ray Vector Environment Wrapper

You would like to use Ray to vectorize your environment but you don't want to use RLLib ?  
You came to the right place !

This package allows you to parallelize your environment using [Ray](https://github.com/ray-project/ray)  
Not only it allows to run environment in parallel, but it also permit to run multiple sequential environment on each worker  
For example, you can run 80 workers in parallel, each running 10 sequential environment for a total of 80 * 10 environment  
This can be usefull if you environment is fast and simply running 1 environment per worker leads to too much communication overhead between workers

## How does it works?
