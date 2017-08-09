from setuptools import setup

setup(
    name='deepmotion',
    version='0.1',
    install_requires=[
        "numpy",
        "pandas",
        "dlib",
        "scipy",
        "tensorflow",
        "h5py",
        "keras",
        "bayesian-optimization",
        "matplotlib",
        "sklearn",
        
    ],
    description="Emotion Recognition Artificial Intelligence based on Deep Learning and Spatio-Temporal Features.",
    url='https://github.com/dshahrokhian/deepmotion',
    packages=['deepmotion', 'deepmotion/dataloader', 'deepmotion/c3d', 'deepmotion/frontalization']
)