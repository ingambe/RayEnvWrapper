from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='RayEnvWrapper',
      version='1.0.1',
      author="Pierre Tassel",
      author_email="pierre.tassel@aau.at",
      description="OpenAi's gym environment wrapper to vectorize them with Ray",
      long_description=long_description,
      long_description_content_type="text/markdown",
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