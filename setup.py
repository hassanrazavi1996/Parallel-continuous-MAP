from setuptools import setup

setup(
    name="cmap",
    version="0.0.1",
    author="Hassan Razavi",
    author_email="hassan.razavi@aalto.fi",
    description="Continuous MAP estimation ",
    install_requires=[
        "jax>=0.4.27",
        "jaxlib>=0.4.27",
        "matplotlib",
        "pandas",
    ],
    zip_safe=False,
)