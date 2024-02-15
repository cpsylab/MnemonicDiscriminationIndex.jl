using MnemonicDiscriminationIndex
using StableRNGs
using Test

const rtol = 0.001

@testset "MnemonicDiscriminationIndex.jl" begin
    # Define a well-known function for testing
    @. quadratic(x, params) = params[1] * x^2 + params[2] * x + params[3]

    quad_data_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    quad_data_y = [0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405, 0.5]

    quad_params =
        fit_model(quadratic, quad_data_x, quad_data_y, () -> [0.6, 0.1, 0.1]).param
    @test all(
        isapprox.(
            quad_params,
            [0.5000000000000945, -9.991904006257049e-14, 1.7089768102995267e-14];
            rtol,
        ),
    )

    quad_auc = get_auc(quadratic, quad_params)
    @test isapprox(quad_auc.auc, 0.33333333333334636; rtol)
    @test isapprox(quad_auc.startval, 1.7091570000361224e-14; rtol)
    @test isapprox(quad_auc.endval, 0.5000000000000117; rtol)

    quad_mdis = get_MD_indices(quad_auc)
    @test isapprox(quad_mdis.Δ, 0.49999999999999456; rtol)
    @test isapprox(quad_mdis.λ, 0.33333333333330006; rtol)

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
            logistic5_data_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            logistic5_data_y = [
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

            logistic5_params =
                fit_logistic5(logistic5_data_x, logistic5_data_y; rng=StableRNG(123)).param
            @test all(
                isapprox.(
                    logistic5_params,
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

            auc = get_auc(logistic5_params)
            @test isapprox(auc.auc, 0.44682346066959855; rtol)
            @test isapprox(auc.startval, 0.05000000000000171; rtol)
            @test isapprox(auc.endval, 0.922727272727276; rtol)

            mdis = get_MD_indices(auc)
            @test isapprox(mdis.Δ, 0.8727272727272742; rtol)
            @test isapprox(mdis.λ, 0.48801478464941916; rtol)
        end
    end
end
