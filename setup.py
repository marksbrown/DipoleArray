from distutils.core import setup

setup(
    name='phased_scatterer_array',
    version='0.1',
    packages=['', 'dipolearray'],
    url='',
    license='CRAPL',
    author='Mark S. Brown',
    author_email='contact@markbrown.io',
    description='The differential cross section for a series of dipoles induced within a periodic array of scatterers is calculated',
    requires=['numpy', 'nose']
)
