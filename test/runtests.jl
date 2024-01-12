using MDI
using StableRNGs
using Test

@testset "MDI.jl" begin

    # Define a well-known function for testing
    @. quadratic(x, params) = params[1]*x^2 + params[2]*x + params[3]

    quad_data_x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    quad_data_y = [0,0.005,0.02,0.045,0.08,0.125,0.18,0.245,0.32,0.405,0.5]

    quad_params = fit_model(quadratic, quad_data_x, quad_data_y, () -> [0.6, 0.1, 0.1]).param
    @test quad_params ≈ [0.5, 0, 0]

    quad_auc = get_auc(quadratic, quad_params)
    @test quad_auc.auc ≈ 0.33333333333334636
    @test quad_auc.startval ≈ 1.7091570000361224e-14
    @test quad_auc.endval ≈ 0.5000000000000117

    quad_mdis = get_MD_indices(quad_auc)
    @test quad_mdis.Δ ≈ 0.49999999999999456
    @test quad_mdis.λ ≈ 0.6666666666666999

    @testset "logistic5" begin
        @testset "logistic5" begin
            # Number
            numoutput = logistic5(0.5, [0.05,5,0.5,0.95,1])
            @test numoutput ≈ 0.5

            # Vectors
            vecoutput1 = logistic5([0.5], [0.02,5,0.6,0.9,13])
            @test vecoutput1 == [0.889104390448767]
            vecoutput2 = logistic5([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], [0.05,5,0.5,0.95,1])
            @test vecoutput2 ≈ [0.050000000000000044, 0.050287907869481896, 0.05912258473234089,
                                0.11493467933491697, 0.27212581344902387, 0.5, 0.6919961471424639,
                                0.8088952438290186, 0.8716421029170033, 0.9047640492810499, 0.9227272727272727]

            # Matrix
            matoutput = logistic5([0 0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8 0.9], [0.05,5,0.5,0.95,1])
            @test matoutput ≈ [0.050000000000000044 0.050287907869481896 0.05912258473234089 0.11493467933491697 0.27212581344902387;
                               0.5                  0.6919961471424639   0.8088952438290186  0.8716421029170033  0.9047640492810499]
        end

        @testset "fit_logistic5" begin
            logistic5_data_x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
            logistic5_data_y = [0.050000000000000044, 0.050287907869481896, 0.05912258473234089,
                                0.11493467933491697, 0.27212581344902387, 0.5, 0.6919961471424639,
                                0.8088952438290186, 0.8716421029170033, 0.9047640492810499, 0.9227272727272727]

            logistic5_params = fit_logistic5(logistic5_data_x, logistic5_data_y; rng=StableRNG(123)).param
            @test logistic5_params ≈ [0.050000000000001696, 5.000000000000355, 0.49999999999996286, 0.950000000000014, 0.9999999999997183]

            auc = get_auc(logistic5_params)
            @test auc.auc ≈ 0.44682346066959855
            @test auc.startval ≈ 0.05000000000000171
            @test auc.endval ≈ 0.922727272727276

            mdis = get_MD_indices(auc)
            @test mdis.Δ ≈ 0.8727272727272644
            @test mdis.λ ≈ 0.5119852153505808
        end
    end
end
