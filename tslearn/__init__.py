import os

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
__version__ = "0.1.16.2"


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    import pyximport
    pyximport.install()
