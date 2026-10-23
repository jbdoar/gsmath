# Reference data notices

The MIT license in `LICENSE` applies to the gsmath software. The reference
datasets identified below retain their own terms and attribution.

## CIE colour-matching functions

The files

- `src/gsmath/reference/data/color_matching_functions/cie1931_2deg_xyzbar.csv`
- `src/gsmath/reference/data/color_matching_functions/cie1964_10deg_xyzbar.csv`

are adapted from datasets published by the International Commission on
Illumination (CIE). Adaptations are limited to CSV headings and numeric
serialization for use by this package.

- International Commission on Illumination (CIE), *Colour-matching functions
  of CIE 1931 standard colorimetric observer* (2019),
  <https://doi.org/10.25039/CIE.DS.xvudnb9b>.
- International Commission on Illumination (CIE), *CIE 1964 colour-matching
  functions, 10 degree observer* (2019),
  <https://doi.org/10.25039/CIE.DS.sqksu2n5>.

The CIE metadata licenses both datasets under the Creative Commons
Attribution-ShareAlike 4.0 International license:
<https://creativecommons.org/licenses/by-sa/4.0/>.

## CIE standard illuminants

The files under `src/gsmath/reference/data/illuminants/` contain relative
spectral distributions for CIE standard illuminants A, B, C, D50, D55, D65,
D75, E, and F1 through F12. Published distributions are linearly aligned to a
1 nm grid, normalized to a peak value of one, and zero-padded outside their
published support. Illuminant E is the locally generated equal-energy
distribution.

The corresponding CIE datasets are:

- Illuminant A, DOI <https://doi.org/10.25039/CIE.DS.8jsxjrsn>.
- Illuminant C, DOI <https://doi.org/10.25039/CIE.DS.mjdd2enu>.
- Illuminant D50, DOI <https://doi.org/10.25039/CIE.DS.etgmuqt5>.
- Illuminant D55, DOI <https://doi.org/10.25039/CIE.DS.qewfb3kp>.
- Illuminant D65, DOI <https://doi.org/10.25039/CIE.DS.hjfjmt59>.
- Illuminant D75, DOI <https://doi.org/10.25039/CIE.DS.9fvcmrk4>.
- Fluorescent illuminants F1 through F12, DOI
  <https://doi.org/10.25039/CIE.DS.54hy6srn>.

The metadata accompanying those CIE datasets licenses them under the Creative
Commons Attribution-ShareAlike 4.0 International license:
<https://creativecommons.org/licenses/by-sa/4.0/>. The scientific definitions
are also specified by:

- International Commission on Illumination (CIE), *Colorimetry, 4th Edition*,
  CIE 015:2018, <https://doi.org/10.25039/TR.015.2018>.
- ISO/CIE 11664-2:2022, *Colorimetry — Part 2: CIE standard illuminants*,
  <https://doi.org/10.25039/IS.ISO_CIE.11664-2.2022>.

The legacy illuminant B tabulation was checked against the `SDS_ILLUMINANTS`
data in [Colour 0.4.7](https://github.com/colour-science/colour/tree/v0.4.7),
distributed by the Colour Developers under the BSD 3-Clause license. A copy
of that license is in `LICENSES/Colour-BSD-3-Clause.txt`.

## Robertson correlated colour temperature

`src/gsmath/reference/data/robertson_table.csv` contains the four columns used
by the Robertson correlated-colour-temperature calculation. It is adapted
from `DATA_ISOTEMPERATURE_LINES_ROBERTSON1968` in Colour 0.4.7, copyright 2013
Colour Developers and distributed under the BSD 3-Clause license. A copy of
that license is in `LICENSES/Colour-BSD-3-Clause.txt`. The adapted file removes
unused columns and retains Colour's corrected values at 125 and 325 reciprocal
megakelvin.

The method is attributed to:

- A. R. Robertson, “Computation of Correlated Color Temperature and
  Distribution Temperature,” *Journal of the Optical Society of America*,
  vol. 58, no. 11, p. 1528 (1968),
  <https://doi.org/10.1364/JOSA.58.001528>.

Colour documents the tabulation against Wyszecki and Stiles, *Color Science:
Concepts and Methods, Quantitative Data and Formulae*, 2nd edition, table
1(3.11), p. 228. These citations are provided for attribution and
reproducibility; they do not place either publication under the gsmath MIT
license.
