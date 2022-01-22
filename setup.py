from setuptools import setup, find_packages

setup(name='RayEnvWrapper',
      version='1.0.0',
      author="Pierre Tassel",
      author_email="pierre.tassel@aau.at",
      description="OpenAi's gym environment wrapper to vectorize them with Ray",
      url="https://github.com/ingambe/RayEnvWrapper",
      packages=find_packages(),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.8',
      install_requires=['gym', 'ray', 'numpy', 'ray[rllib]', 'stable-baselines3'],
      include_package_data=True
)