from setuptools import find_packages
from setuptools import setup
'',
REQUIRED_PACKAGES=['numpy==1.19.2','requests==2.21.0','six==1.15.0','google-cloud-storage==1.34.0','audio2numpy==0.1.2','librosa==0.6.3', 'numba==0.48','scipy==1.4.1', 'tensorflow>=2.2', 'keras==2.4.3', 'matplotlib==3.3.3']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="Generacion de musica"
)