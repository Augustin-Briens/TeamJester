"""recon3d — optional 2D->3D reconstruction & transport branch.

Independent of polaron_qc and micro2dfn. Reuses micro2dfn's frozen
threshold segmentation so the phase definition is identical.
"""
"""
Phases: 0 = pore, 1 = bulk (graphite+binder), 2 = bright (Si-family,
unclassified — the "all bright counts" bracket interpretation).
"""
