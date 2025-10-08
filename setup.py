from setuptools import setup

setup(
    name="cmap",
    version="0.0.1",
    author="Hassan Razavi",
    author_email="hassan.razavi@aalto.fi",
    description="",
    install_requires=["jax[cuda12]", "matplotlib"],
    zip_safe=False,
)
