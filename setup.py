from setuptools import setup

setup_requires = []

install_requires = ["numpy", "matplotlib", "scipy"]

setup(
    name="square",
    version="0.0.1",
    description="experimental",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    package_data={"square": ["py.typed"]},
)
