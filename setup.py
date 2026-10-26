from setuptools import setup

setup(
    name='qvis',
    version="0.0.1",
    packages=[],
    instal_requires=[],
    author='Fabricio Barth & Robson França',
    author_email='fabriciojb@insper.edu.br',
    description='Q-Vis: Visualization of Tabular Reinforcement Learning Algorithms',
    long_description=open("ABOUT.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Insper/qvis",
    license='MIT',
    data_files=[('dist', ['ABOUT.md'])]
)