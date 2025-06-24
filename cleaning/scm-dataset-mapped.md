| Dataset Column                         | Required Column        | Notes                                      |
| -------------------------------------- | ---------------------- | ------------------------------------------ |
| `Cement(kg/m3)`                        | `cement_opc`           | ✔ Direct                                   |
| `FA (kg/m3)`                           | `scm_flyash`           | ✔ Direct                                   |
| `GGBFS (kg/m3)`                        | `scm_ggbs`             | ✔ Direct                                   |
| (missing)                              | `silica_sand`          | ✖ Not available → fill as 0                |
| `Fine aggregate(kg/m3)`                | `locally_avail_sand`   | ✔ Direct                                   |
| `Water(kg/m3)` + `Cement(kg/m3)`       | `w_b`                  | ➕ compute: `Water / (Cement + FA + GGBFS)` |
| `SP (kg/m3)`                           | `hrwr_b`               | ➕ compute: `SP / (Cement + FA + GGBFS)`    |
| (missing)                              | `perc_of_fibre`        | ✖ Not available → fill as 0                |
| (missing)                              | `aspect_ratio`         | ✖ Not available → fill as 0                |
| `Splitting tensile strength (MPa)`     | `tensile_strength`     | ✔ Direct                                   |
| `Cement+Water+Coarse+Fine+FA+SF+GGBFS` | `density`              | ➕ approximate total mass                   |
| `Elastic modulus (GPa)`                | `youngs_modulus`       | ✔ Direct                                   |
| `Elongation (%)`                       | `elongation`           | ✖ Not available → fill as 0                |
| `Cylinder compressive strength (MPa)`  | `compressive_strength` | ✔ Direct                                   |