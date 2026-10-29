# PeacoQC example file

`111_Comp_Trans.fcs` is the compensated and logicle-transformed example file
shipped with PeacoQC (https://github.com/saeyslab/PeacoQC, `inst/extdata/`),
GPL-3, by Annelies Emmaneel and the Saeys lab. Sidescatter is GPL-3.0-or-later,
so it can be redistributed here.

`test_cleaning.py` checks that Sidescatter removes exactly the events PeacoQC
removes from it. The reference (events 1-1000 and 9501-9617, 11.6148%) came
from running PeacoQC's own R functions (MakeBreaks, DeterminePeaksAllChannels,
MADOutlierMethod, RemoveShortRegions) on channels 1, 3, 5-14, 18 and 21 with
MAD 6, the vignette's channels and default settings.
