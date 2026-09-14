from setuptools import setup
import os
from setuptools import dist

required = [
    "numpy",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "matplotlib>=3.5.0",
    "psutil",
    "futures",
    #"flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
    "black==22.8.0",
    "flake8==5.0.4",
    "datasets"
]


setup(
    name='es_reasoning',
    version='0.0.1',
    description='Python code to tune LLMs with ES and RL.',
    author='',
    packages=['es_reasoning'],
    install_requires=required,
)
