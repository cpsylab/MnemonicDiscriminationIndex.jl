MnemonicDiscriminationIndex.jl
======
Package for our new index of mnemonic discrimination

Installation
------------
To install this package to you environment, run
```julia
]add https://github.com/cpsylab/MnemonicDiscriminationIndex.jl#v0.2.0
```
in the Julia REPL. The `"#v0.2.0"` part is not necessary, but it'll prevent your code from breaking if there are ever any breaking changes to the package.

Basic Usage
-----------

For a workflow similar to the paper using the logistic5, lets assume we have our MST data.
```julia
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

julia> distance = 0:(1/7):1
0.0:0.14285714285714285:1.0
```

Then, we fit our data to the logistic5 curve (using a reproducible rng) and save the parameters:
```julia
julia> using StableRNGs

julia> logistic5_params = fit_logistic5(distance, old_or_new; rng=StableRNG(123)).param
5-element Vector{Float64}:
 0.0
 2.311177095326278
 0.9236730069589213
 1.0
 3.546302574580187
```

Now, we're ready to calculate the auc and the Δ and λ indices:
```julia
julia> auc = get_auc(logistic5_params)
MnemonicDiscriminationIndex.AUC{Float64}(0.44583208733243396, 0.0, 0.9390919680940668, (0, 1))

julia> mdis = get_MD_indices(auc)
MnemonicDiscriminationIndex.MDIndices{Float64}(0.9390919680940668, 0.5252519428557438)

julia> mdis.Δ
0.9390919680940668

julia> mdis.λ
0.5252519428557438
```

To fit to a different function than `logistic5`, check out the docstrings for `fit_model` and `get_auc` by calling `?fit_model`, and `?get_auc` in the REPL.

Paper
-----
Code for the paper can be found [here](https://github.com/cpsylab/New-MD-Measure-Code/).

Release v0.1.0 of this repository is the version of the code used in the paper.
