MnemonicDiscriminationIndex.jl
======
Package for our new index of mnemonic discrimination

Installation
------------
To install this package to you environment, run
```julia-repl
julia> ]add MnemonicDiscriminationIndex.jl
```
in the Julia REPL. To prevent breakage, add a compat entry to the version you install in your environment's `Project.toml` as the API might be improved over time.

Basic Usage
-----------

For a workflow similar to the paper using the logistic5, lets assume we have our MST data.
```julia-repl
julia> using MnemonicDiscriminationIndex

julia> old_or_new = [0,0,0,1,0,1,1,1]
8-element Vector{Int64}:
 0
 0
 0
 1
 0
 1
 1
 1

julia> dissimilarity = 0:(1/7):1
0.0:0.14285714285714285:1.0
```

Then, we fit our data to the logistic5 curve (using a reproducible rng) and save the parameters:
```julia-repl
julia> using StableRNGs

julia> logistic5_results = fit_logistic5(dissimilarity, old_or_new; rng=StableRNG(123));
```

Now, we have an MDIResult object with all the information needed:
```julia-repl
julia> logistic5_results.auc
0.44583208733243396

julia> logistic5_results.Δ
0.9390919680940668

julia> logistic5_results.λ
0.5252519428557438
```

To fit to a different function than `logistic5`, check out the docstrings for `fit_model` by calling `?fit_model` in the REPL.

Paper
-----
Code for the paper can be found [here](https://github.com/cpsylab/New-MD-Measure-Code/).

Release v0.1.0 of this repository is the version of the code used in the paper.
