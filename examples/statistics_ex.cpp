#include "../statistics/statistics.hpp"
#include <fstream>

using namespace statistics;

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "normalCDF(0)     = " << normalCDF(0)    << "\terror: " << std::abs(0.5 - normalCDF(0) )           << std::endl
              << "normalCDF(1.0)   = " << normalCDF(1.0)  << "\terror: " << std::abs(0.8413 - normalCDF(1.0) )      << std::endl
              << "normalCDF(-2.1)  = " << normalCDF(-2.1) << "\terror: " << std::abs(0.01786 - normalCDF(-2.1) )    << std::endl
              << "normalQ(0.3)     = " << normalQ(0.3)    << "\terror: " << std::abs(-0.5244 - normalQ(0.3) )       << std::endl
              << "normalQ(0.84)    = " << normalQ(0.84)   << "\terror: " << std::abs(0.99446 - normalQ(0.84) )      << std::endl
              << "normalQ(0.5)     = " << normalQ(0.5)    << "\terror: " << std::abs(0 - normalQ(0.5) )             << std::endl;
    stats x(231.9, 2.19, 66);
    std::cout << z_test(x,232) << std::endl << std::endl;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "tCDF(1, 3)      = " << tCDF(1, 3)     << "\terror: " << std::abs(0.8045 - tCDF(1,3) )         << std::endl
              << "tCDF(-0.5, 6.4) = " << tCDF(-0.5,6.4) << "\terror: " << std::abs(0.3169 - tCDF(-0.5,6.4) )    << std::endl
              << "tCDF(2.1, 21.1) = " << tCDF(2.1,21.1) << "\terror: " << std::abs(0.976 - tCDF(2.1,21.1) )     << std::endl
              << "tQ(0.3, 10)     = " << tQ(0.3,10)     << "\terror: " << std::abs(-0.5415 - tQ(0.3,10) )       << std::endl
              << "tQ(0.95, 3.2)   = " << tQ(0.95,3.2)   << "\terror: " << std::abs(2.2946 - tQ(0.95,3.2) )      << std::endl
              << "tQ(0.6, 16.1)   = " << tQ(0.6, 16.1)  << "\terror: " << std::abs(0.2576 - tQ(0.6, 16.1) )     << std::endl;

    std::vector<double> x1 = {2.88, 2.85, 1.84,  1.6,  0.8, 0.89, 2.03,  1.9};
    std::vector<double> x2 = {2.59, 2.47, 1.58, 1.56, 0.78, 0.66, 1.87, 1.71};
    std::cout << t_test(x1, 2.0, hypothesis::GREATER, 0.95);
    std::cout << t_test(x1, x2, 0, hypothesis::NEQ, 0.95, true);
    std::cout << t_test(x1, x2);

    std::cout << p_test(84, 200, 0.5);

    std::ifstream samp1("examples/sample1.txt");
    std::ifstream samp2("examples/sample2.txt");

    arma::vec u1(25, arma::fill::zeros);
    arma::vec u2(20, arma::fill::zeros);

    for (int i(0); i < 25; ++i) {
        double x,y;
        samp1 >> x;
        if (i < 20) samp2 >> y;

        u1(i) = x;
        if (i < 20) u2(i) = y;
    }

    samp1.close();
    samp2.close();

    std::cout << std::setprecision(5) << "permuations test. p = " << perm_test(u1,u2) << std::endl;
    return 0;
}