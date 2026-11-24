# Historical parity record

The refactored implementation was compared against the historical RSRS and
RLST fork using the same 5,041-degree-of-freedom perturbed BIEGrid matrix and
the same 602 sample rows.

Both implementations produced:

- tree levels `[4, 3, 2]`;
- cycle counts `[175, 64, 16]`;
- batch counts `[10, 13, 9]`;
- 255 elimination cycles and 256 final active degrees of freedom.

The first cycle selected the same skeleton and redundant index sets. When the
refactored projected panel was restricted to the historical row count, its
first interpolation matrix agreed with the historical matrix to approximately
`1.1e-7` in relative Frobenius norm, and all 175 finest-level cycles selected
the same skeleton sets.

The intentional production difference is projection-row handling:

- historical RSRS retained only `sample_rows - near_columns` rows after
  projection (26 of 107 rows in the first cycle);
- the refactored implementation retains all projected rows.

For the fixed BIEGrid diagnostic vector, the historical backend reported a
forward relative error of `1.213101903222e-3` and an inverse residual of
`1.436558487451e-3`. The all-row refactored implementation reported
`1.331841531164e-3` and `1.568654925401e-3`, respectively. Restricting only
the refactored diagnostic to the historical row count gave `1.298221778956e-3`
and `1.531692977346e-3`; later differences are floating-point/order
propagation through the hierarchy and extraction stages.

The all-row policy is covered by a unit test and is the production default.
QR and SVD orthonormal nullification remain available when stronger numerical
robustness is preferred over the cheaper projection method.
