from setuptools import setup

setup(
    name='keggpathwaygraphs',
    version='1.0.0',
    packages=['keggpathwaygraphs'],
    install_requires=['biopython'],
    url='https://github.com/mdparkes/keggpathwaygraphs',
    license='MIT',
    author='Michael Parkes',
    author_email='',
    description='Python scripts for scraping KEGG BRITE orthology KGML data and constructing graph objects from it'
)
