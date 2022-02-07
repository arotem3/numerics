#include <iostream>
#include <armadillo>
#include <valarray>
#include "numerics/ode/trbdf2.hpp"

using numerics::ode::ivpOpts;
using numerics::ode::trbdf2;
using namespace std::complex_literals;

int main()
{
    int n_passed = 0, n_failed = 0;
    { // double
        auto f = [](double t, double y) -> double {return -std::sin(2*M_PI*t)*y;};
        double y0 = 1.0;
        auto yexact = [y0](double t) -> double {return y0*std::exp((std::cos(2*M_PI*t)-1)/(2*M_PI));};
        std::vector<double> tspan = {0.0, 1.0};
        ivpOpts<double> opts;
        opts.dense_output = true;

        auto sol = trbdf2(f, tspan, y0, opts);

        double emax = 0;
        for (int i=0; i < sol.t.size(); ++i)
            emax = std::max(emax, std::abs(sol.y[i] - yexact(sol.t[i])));

        if (emax > 1e-2) {
            std::cout << "trbdf2 failed double precision accuracy test.\n";
            ++n_failed;
        } else
            ++n_passed;
        
        std::vector<double> x(100);
        for (int i=0; i < 100; ++i)
            x[i] = i*0.01;

        auto y = sol(x); // interpolate

        emax = 0;
        for (int i=0; i < 100; ++i)
            emax = std::max(emax, std::abs(y[i] - yexact(x[i])));

        if (emax > 1e-2) {
            std::cout << "trbdf2 failed double precision interpolation test.\n";
            ++n_failed;
        } else
            ++n_passed;
    }

    { // complex double
        typedef std::complex<double> cx;
        auto f = [](double t, cx y) -> cx {return -y;};
        cx y0 = std::sqrt(1.0i);
        auto yexact = [y0](double t) -> cx {return y0*std::exp(-t);};

        std::vector<double> tspan = {0.0, 1.0};
        ivpOpts<double> opts;
        opts.dense_output = true;

        auto sol = trbdf2(f, tspan, y0, opts);

        double emax = 0;
        for (int i=0; i < sol.t.size(); ++i)
            emax = std::max(emax, std::abs(sol.y[i] - yexact(sol.t[i])));

        if (emax > 1e-2) {
            std::cout << "trbdf2 failed complex double precision accuracy test.\n";
            ++n_failed;
        } else
            ++n_passed;

        bool A = false, B = false;
        try {
            cx y = sol(-1.0);
        } catch (const std::exception& e) {
            A = true;
        }
        try {
            cx y = sol(2.0);
        } catch(const std::exception& e) {
            B = true;
        }

        if (A and B)
            ++n_passed;
        else {
            std::cout << "trbdf2 failed safe interpolation check\n";
            ++n_failed;
        }

        cx y = sol(M_1_PI);
        emax = std::abs(y - yexact(M_1_PI));

        if (emax > 1e-2) {
            std::cout << "trbdf2 failed complex interpolation accuracy test\n";
            ++n_failed;
        } else
            ++n_passed;
    }

    { // armadillo fvec
        const arma::fmat A = {{-500.5f, -499.5f},{-499.5f, -500.5f}}; // moderately stiff, lambda = -1, -1000
        auto f = [&A](float t, const arma::fvec& y) {return A*y;};
        arma::fvec y0 = {1.0, 0.0};

        arma::cx_fvec e; arma::cx_fmat R;
        arma::eig_gen(e, R, A);
        auto yexact = [&e,&R,&y0](float t) -> arma::fvec {return arma::real(R*arma::diagmat(arma::exp(t*e))*arma::inv(R)*y0);};

        std::vector<float> tspan = {0.0, 1.0, 2*M_PI};

        auto sol = trbdf2(f, tspan, y0);

        float emax = 0;
        for (int i=0; i < sol.t.size(); ++i)
            emax = std::max(emax, arma::norm(sol.y[i] - yexact(sol.t[i])));

        if (emax > 1e-2) {
            std::cout << "trbdf2 failed armadillo vec single precision accuracy test.\n";
            ++n_failed;
        } else
            ++n_passed;

        bool check = false;
        for (float s : sol.t)
        {
            if (std::abs(s-1) == 0) {
                check = true;
                break;
            }
        }

        if (not check) {
            std::cout << "trbdf2 failed to compute solution at requested points in tspan.\n";
            ++n_failed;
        } else
            ++n_passed;

    }

    { // valarray double
        typedef std::valarray<double> arr;
        auto f = [](double t, arr y) -> arr {return {-500.5*y[0]-499.5*y[1], -499.5*y[0]-500.5*y[1]};};
        arr y0 = {1.0, 0.0};
        auto yexact = [](double t) -> arr {double u = std::exp(-1000*t); double v = std::exp(-t); return {0.5*u+0.5*v, 0.5*u-0.5*v};};

        std::vector<double> tspan = {0.0, M_PI};
        
        auto sol = trbdf2(f, tspan, y0);

        double emax = 0;
        for (int i=0; i < sol.t.size(); ++i)
            emax = std::max(emax, std::abs(yexact(sol.t[i]) - sol.y[i]).sum());

        if (emax > 1e-2) {
            std::cout << "trbdf2 failed valarray double precision accuracy test.\n";
            ++n_failed;
        } else
            ++n_passed;
    }

    std::cout << "trbdf2 passed " << n_passed << " / " << n_passed + n_failed << " tests.\n";

    return 0;
}