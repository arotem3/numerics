cmake_minimum_required(VERSION 3.10)
project(numerics)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)

include_directories(include)

add_library(numerics SHARED
        include/numerics.hpp
        include/numerics/data_science.hpp
        include/numerics/derivatives.hpp
        include/numerics/integrals.hpp
        include/numerics/interpolation.hpp
        include/numerics/ode.hpp
        include/numerics/optimization.hpp
        include/numerics/utility.hpp

        src/data_science/kd_tree.cpp
        src/data_science/bin_data.cpp
        src/data_science/kmeans.cpp
        src/data_science/polyfeat.cpp
        src/data_science/lasso.cpp
        src/data_science/ridge.cpp
        src/data_science/splines.cpp
        src/data_science/bw.cpp
        src/data_science/kernel_smooth.cpp
        src/data_science/kde.cpp
        src/data_science/logistic_regression.cpp
        src/data_science/pairwise_kernel.cpp

        src/derivatives/approx_jacobian.cpp
        src/derivatives/deriv.cpp
        src/derivatives/grad.cpp
        src/derivatives/spectral_deriv.cpp

        src/integrals/chebyshev_integral.cpp
        src/integrals/lobatto_integral.cpp
        src/integrals/simpson_integral.cpp

        src/interpolation/cubic_interp.cpp
        src/interpolation/hspline_interp.cpp
        src/interpolation/lagrange_interp.cpp
        src/interpolation/sinc_interp.cpp
        src/interpolation/polynomials.cpp

        src/ode/am2.cpp
        src/ode/bvp_k.cpp
        src/ode/bvp_cheb.cpp
        src/ode/bvpIIIa.cpp
        src/ode/cheb.cpp
        src/ode/diffmat.cpp
        src/ode/ivp.cpp
        src/ode/rk4.cpp
        src/ode/rk5i.cpp
        src/ode/rk45.cpp
        src/ode/rk34i.cpp
        src/ode/poisson2d.cpp

        src/optimization/quasi_newton.cpp
        src/optimization/gradient_optimizer.cpp
        src/optimization/bfgs.cpp
        src/optimization/broyd.cpp
        src/optimization/pcg.cpp
        src/optimization/gmres.cpp
        src/optimization/fzero.cpp
        src/optimization/fmin.cpp
        src/optimization/lbfgs.cpp
        src/optimization/lmlsqr.cpp
        src/optimization/mgd.cpp
        src/optimization/mix_fpi.cpp
        src/optimization/nelder_mead.cpp
        src/optimization/newton.cpp
        src/optimization/trust_newton.cpp
        src/optimization/simplex.cpp
        src/optimization/wolfe_step.cpp
        
        src/utility/meshgrid.cpp
        src/utility/sample_from.cpp

        src/neural_network/layer.cpp
        src/neural_network/optimize.cpp
        src/neural_network/model.cpp
)

if(OpenMP_FOUND)
	target_link_libraries(numerics PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS numerics DESTINATION lib)
install(FILES include/numerics.hpp DESTINATION include)
install(DIRECTORY include/numerics DESTINATION include)