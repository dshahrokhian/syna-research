from setuptools import setup

setup(
    name='syna',
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
    url='https://github.com/dshahrokhian/syna',
    packages=['syna', 'syna/dataloader', 'syna/c3d', 'syna/frontalization']
)