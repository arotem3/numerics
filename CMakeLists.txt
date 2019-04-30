cmake_minimum_required(VERSION 3.10)
project(numerics)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_library(numerics SHARED
        include/numerics.hpp
        include/ODE.hpp
        include/statistics.hpp
        src/integration/integrate.cpp
        src/integration/mcIntegrate.cpp
        src/interpolation/cinterp.cpp
        src/interpolation/interp.cpp
        src/interpolation/polyInterp.cpp
        src/interpolation/sincInterp.cpp
        src/data_sci/kmeans.cpp
        src/data_sci/splines.cpp
        src/data_sci/kernel_smooth.cpp
        src/data_sci/k_fold.cpp
        src/data_sci/regularizer.cpp
        src/nonlinear/adj_gd.cpp
        src/nonlinear/bfgs.cpp
        src/nonlinear/broyd.cpp
        src/nonlinear/cgd.cpp
        src/nonlinear/fpi.cpp
        src/nonlinear/fzero.cpp
        src/nonlinear/genetic.cpp
        src/nonlinear/line_min.cpp
        src/nonlinear/lmlsqr.cpp
        src/nonlinear/minimize_unc.cpp
        src/nonlinear/momentum.cpp
        src/nonlinear/newton.cpp
        src/nonlinear/nlcgd.cpp
        src/nonlinear/sgd.cpp
        src/nonlinear/simplex.cpp
        src/nonlinear/wolfe_step.cpp
        src/ODEs/am1.cpp
        src/ODEs/am2.cpp
        src/ODEs/bdf23.cpp
        src/ODEs/cheb.cpp
        src/ODEs/diffmat.cpp
        src/ODEs/ivp_events.cpp
        src/ODEs/IVP_solve.cpp
        src/ODEs/linear_bvp.cpp
        src/ODEs/bvp.cpp
        src/ODEs/poisson.cpp
        src/ODEs/rk4.cpp
        src/ODEs/rk45.cpp
        src/ODEs/rk5i.cpp
        src/statistics/chi_sq.cpp
        src/statistics/distributions.cpp
        src/statistics/functions.cpp
        src/statistics/tests.cpp
        src/utility/cyc_queue.cpp
        src/utility/eval.cpp
        src/utility/meshgrid.cpp
        src/utility/sample_from.cpp
        src/finite_dif.cpp
        )

install(TARGETS numerics DESTINATION lib)
install(FILES include/numerics.hpp include/ODE.hpp include/statistics.hpp DESTINATION include)