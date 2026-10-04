# NOTES - silicon particle size analysis

## Image inventory
- Batch_1: 7 BSE images; Batch_2: 7; Batch_3: 17. Matches the earlier run's
  count of 7 / 7 / 17 - no mismatch.
- Some locations store the third detector image as `*_SE.tif` instead of
  `*_ETD.tif` (Batch_2: rxax5ozo; Batch_3: utfgcjfa, vc2whyaq, x77cy643).
  BSE counts are unaffected; every location still has three files.

## Assumptions
- Bright BSE particles are ASSUMED silicon-based; grey bulk ASSUMED graphite
  - from image appearance only, no compositional (e.g. EDS) confirmation.
  This assumption applies to every output.
- Pixel size is taken from the TIFF resolution tags: ~25.0 nm per pixel in
  every BSE image (range 24.999-25.001 nm). No scale bar is burned into the
  images and no other instrument metadata exists.
- Segmented objects may be agglomerates or embedded composite particles,
  not primary crystallites.
- 2D section diameters understate true 3D size (stereological bias).

## Method notes
- No information banner found in any image (`banner_rows_cropped`=0
  everywhere).
- Thresholding: most images show no separate bright peak in the histogram
  (the bright signal is a shoulder/tail of the bulk peak), so they use the
  3-class multi-Otsu fallback - see `method` column in the Per_image sheet.
- The 'large particle' cut-off is the pooled D90 across all images
  (1.57 um): a RELATIVE limit. The paper gives no physical cut-off
  applicable at micron scale (its particles are 3-27 nm; the ~150 nm
  rupture limit it cites is below our minimum detectable object).

## Outlier images
- Batch_1 images `4ih2ggld` and `5n1q8atc` returned ~2-4x the particle
  count of other images at visibly lower contrast. Inspection of their
  overlays shows the threshold catching marginal bright material - bright
  flake interiors, edge rims and fragmented rim segments - in addition to
  genuine bright particles. Their counts are inflated and their size
  distributions skewed small; the segmentation is NOT considered fully
  valid for particle statistics on these two images. All batch statistics
  and tests are reported twice: `all_images` and
  `excluding_outlier_images` (see the `scope` column).

## Statistics
- The IMAGE is the independent sample (n = 7, 7, 17), not the particle.
  Batch_3's larger image count gives pairwise tests involving it more
  power than the Batch_1-Batch_2 comparison.
- Bootstrap CIs (2000 resamples) resample images with replacement and
  recompute the pooled-particle quantile each time.

## Known issues / what failed
- Reading the TIFFs requires the `imagecodecs` package (LZW decode); it is
  a dependency of tifffile, not an extra analysis library.
- Overlays are saved at 50% resolution (~3500 px wide) to keep file sizes
  reasonable; outlines are dilated to 3 px before downscaling so they stay
  visible. Boundary accuracy is better checked on the full-size label data
  if needed.
- No image was excluded entirely; border-touching particles are flagged
  (`touches_border`) and excluded from size statistics.
