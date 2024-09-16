import os
from setuptools import setup, find_packages
from distutils.core import Extension
from pathlib import Path

MINIMAL_DESCRIPTION = '''RGC-Quant: Retinal Ganglion Cell (RGC) automatic counting in 3D confocal images using Deep Learning'''

def get_requires():
    """Read requirements.txt."""
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(requirements_file, "r") as f:
            requirements = f.read()
        return list(filter(lambda x: x != "", requirements.split()))
    except FileNotFoundError:
        return []

def read_description():
    """Read README.md and CHANGELOG.md."""
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path) as r:
            description = "\n" + r.read()
        return description
    return MINIMAL_DESCRIPTION

setup(
    name="ElecPhys",
    version="0.0.1",
    author='Amin Alam, Arsalan Firoozi',
    description='RGC Quantification',
    long_description=read_description(),
    long_description_content_type='text/markdown',
    install_requires=get_requires(),
    python_requires='>=3.8',
    license='MIT',
    url='https://github.com/AminAlam/RGC-Quant',
    keywords=['RGC', 'Quantification', 'Deep Learning', '3D', 'Confocal', 'Microscopy', 'Retina', 'Neuroscience', 'Image Processing', 'Image Analysis', 'Image Segmentation', 'Deep learning', 'PyTorch', '3D U-Net'],
    entry_points={
        'console_scripts': [
            'rgc_quant=rgc_quant.main:main',
        ],
    },
    packages=['rgc_quant'],
    package_data={'rgc_quant': ['*.pth', 'rgc_quant/models/*.m']},
    include_package_data=True,
)