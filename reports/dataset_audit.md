# Dataset audit

Reference polygons after GEE filters: 10144; with features: 10144

- rows with missing S1 seasonal: 0 (not used to filter; see S1 experiment)
- rows with missing AE: 0
- rows with missing S2-annual: 0
- rows with missing S2-seasonal: 0
- common valid intersection: 10144 of 10144

Dropped rows by country/class:

none

AE polygon-mean vector norm: median 0.991, 1st pct 0.951, 99th pct 0.999
S2_nobs: median 74.7, min 19.0
S2q1_nobs: median 12.0, min 1.0
S2q2_nobs: median 17.7, min 2.4
S2q3_nobs: median 26.0, min 4.0
S2q4_nobs: median 16.0, min 3.0
S2-annual reflectance range: 0.0119 to 0.6627
10 m pixels per polygon: min 3, 1st pct 6, median 43, max 77
pixels with >=1 clear S2 observation per polygon: min 3, median 43; share of polygons fully clear 1.000
quarter 1: min area fraction with >=1 clear S2 observation 1.000; S1 observations per cell, ascending min 11.9 median 15, descending min 12.4 median 23
quarter 2: min area fraction with >=1 clear S2 observation 1.000; S1 observations per cell, ascending min 7.0 median 16, descending min 11.0 median 16
quarter 3: min area fraction with >=1 clear S2 observation 1.000; S1 observations per cell, ascending min 7.0 median 27, descending min 13.0 median 25
quarter 4: min area fraction with >=1 clear S2 observation 1.000; S1 observations per cell, ascending min 6.0 median 32, descending min 14.0 median 32
Duplicate point_id: 0

Final samples by country and class:

nuts0       EL    ES
label               
Bare land   56   593
Cropland   511  3044
Grassland  545  1542
Shrubland  205  1281
Woodland   468  1899

Survey months (ES+EL):

survey_date
4      484
5     2178
6     2789
7     2866
8     1214
9      481
10     132