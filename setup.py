from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES=['audio2numpy==0.1.2','librosa==0.6.3', 'numba==0.48','scipy==1.4.1', 'tensorflow>=2.2', 'keras==2.4.3']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Generacion de musica"
)