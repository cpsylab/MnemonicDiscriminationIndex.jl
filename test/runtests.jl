using MnemonicDiscriminationIndex
using StableRNGs
using Test

const rtol = 0.001

@testset "MnemonicDiscriminationIndex.jl" begin
    # Define a well-known function for testing
    @. quadratic(x, params) = params[1] * x^2 + params[2] * x + params[3]

    quad_dissimilarities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    quad_responses = [0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405, 0.5]

    quad_res =
        fit_model(quadratic, quad_dissimilarities, quad_responses, () -> [0.6, 0.1, 0.1])
    @test all(
        isapprox.(
            quad_res.params,
            [0.5000000000000945, -9.991904006257049e-14, 1.7089768102995267e-14];
            rtol,
        ),
    )

    @test isapprox(quad_res.auc, 0.33333333333334636; rtol)
    @test isapprox(quad_res.startval, 1.7091570000361224e-14; rtol)
    @test isapprox(quad_res.endval, 0.5000000000000117; rtol)

    @test isapprox(quad_res.Δ, 0.49999999999999456; rtol)
    @test isapprox(quad_res.λ, 0.33333333333330006; rtol)

    @testset "logistic5" begin
        @testset "logistic5" begin
            # Number
            numoutput = logistic5(0.5, [0.05, 5, 0.5, 0.95, 1])
            @test isapprox(numoutput, 0.5; rtol)

            # Vectors
            vecoutput1 = logistic5([0.5], [0.02, 5, 0.6, 0.9, 13])
            @test all(isapprox.(vecoutput1, [0.889104390448767]; rtol))
            vecoutput2 = logistic5(
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [0.05, 5, 0.5, 0.95, 1],
            )
            @test all(
                isapprox.(
                    vecoutput2,
                    [
                        0.050000000000000044,
                        0.050287907869481896,
                        0.05912258473234089,
                        0.11493467933491697,
                        0.27212581344902387,
                        0.5,
                        0.6919961471424639,
                        0.8088952438290186,
                        0.8716421029170033,
                        0.9047640492810499,
                        0.9227272727272727,
                    ];
                    rtol,
                ),
            )

            # Matrix
            matoutput =
                logistic5([0 0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8 0.9], [0.05, 5, 0.5, 0.95, 1])
            @test all(
                isapprox.(
                    matoutput,
                    [
                        0.050000000000000044 0.050287907869481896 0.05912258473234089 0.11493467933491697 0.27212581344902387
                        0.5 0.6919961471424639 0.8088952438290186 0.8716421029170033 0.9047640492810499
                    ];
                    rtol,
                ),
            )
        end

        @testset "fit_logistic5" begin
            logistic5_dissimilarities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            logistic5_responses = [
                0.050000000000000044,
                0.050287907869481896,
                0.05912258473234089,
                0.11493467933491697,
                0.27212581344902387,
                0.5,
                0.6919961471424639,
                0.8088952438290186,
                0.8716421029170033,
                0.9047640492810499,
                0.9227272727272727,
            ]

            logistic5_res = fit_logistic5(
                logistic5_dissimilarities,
                logistic5_responses;
                rng=StableRNG(123),
            )
            @test all(
                isapprox.(
                    logistic5_res.params,
                    [
                        0.050000000000001696,
                        5.000000000000355,
                        0.49999999999996286,
                        0.950000000000014,
                        0.9999999999997183,
                    ];
                    rtol,
                ),
            )

            @test isapprox(logistic5_res.auc, 0.44682346066959855; rtol)
            @test isapprox(logistic5_res.startval, 0.05000000000000171; rtol)
            @test isapprox(logistic5_res.endval, 0.922727272727276; rtol)

            @test isapprox(logistic5_res.Δ, 0.8727272727272742; rtol)
            @test isapprox(logistic5_res.λ, 0.48801478464941916; rtol)
        end
    end
end
