Data Information
================
All molecule SMILES data is from [ZINC15 database](https://zinc15.docking.org).

Positive set (Drug)
- fda: FDA Approved drugs, per DrugBank
- world-not-fda: Drugs approved, but not by the FDA

Negative set (NonDrug-Like)
- ZINC: the data used as negative set in the research of Beker et al.
[Beker, W., Wołos, A., Szymkuć, S. et al. Minimal-uncertainty prediction of general drug-likeness based on Bayesian neural networks. Nat Mach Intell 2, 457–465 (2020). https://doi.org/10.1038/s42256-020-0209-y](https://doi.org/10.1038/s42256-020-0209-y)

We only use RdKit-readable smiles data.

For the negative set ZINC, the molecules with tanimoto similarity to the positive set less than 0.8 were used.

* * *
Datafile
- train.csv: world-not-fda(2,833 molecules) + ZINC(2,833 molecules)
- test.csv: fda(1,489 molecules) + ZINC(1,489 molecules)
