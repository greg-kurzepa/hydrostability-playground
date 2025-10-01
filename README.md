# hydrostability-playground

Simulaion of hydrostability of 2D shapes using three methods:
- minimum potential energy
- GZ righting arm
- catastrophe theory (buoyancy loci & their evolutes)

The technical report is `Hydrostatic_Stability_Report.pdf`.

Here is an example from the technical report, where the hydrostability of a buoyant anchor is analysed using GZ curves:
<img width="1026" height="662" alt="image" src="https://github.com/user-attachments/assets/ace9d5a5-23a3-4a99-becc-7c362ed93087" />
<img width="1164" height="598" alt="image" src="https://github.com/user-attachments/assets/81a553db-f101-4cc8-ad48-29a7dc7b0459" />

Instances of `Shape` class are defined as an arrangement of instances of the `Triangle` class. Each triangle is defined by the coordinates of its three corners.

The majority of the calculations are in `polygons.py`. The Pygame interface is in `display.py`. The other python files are for plotting or tests.
