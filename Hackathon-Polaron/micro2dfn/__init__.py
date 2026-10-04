"""micro2dfn — BSE cross-section -> structural markers -> PyBaMM DFN.

Standalone package (no imports from polaron_qc). Pipeline:
    BSE TIFF -> flat-field -> shared multi-Otsu -> watershed +
    Si/bright-fine object classification -> stereological markers
    -> per-batch bands -> PyBaMM DFN sweep -> consequence indicators.
"""
__version__ = "0.1.0"
