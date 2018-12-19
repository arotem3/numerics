#include "statistics.hpp"

//--- normal Distribution ---//
double statistics::normalPDF(double x) {
    return M_1_SQRT2PI * std::exp(-x*x/2.0);
}

double statistics::normalCDF(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2l);
}

double statistics::normalQ(double p) {
    if (p == 0.5) return 0;
    // (1) determine good guess using empirical rule
    double guess;
    if (p < 0.16) guess = -2;
    else if (p < 0.5) guess = -1;
    else if (p < 0.975) guess = 1;
    else guess = 2;
    // (2) solve root finding problem: normalCDF(x) - p = 0
    auto f = [p](double x) -> double {return normalCDF(x) - p;};
    return numerics::newton(f, normalPDF, guess);
}

//--- student's t distribution ---//
double statistics::tPDF(double x, double df) {
    if (df < 1) { // error
        std::cerr << "tPDF() error: t-distribution is undefined for df < 1" << std::endl;
        return NAN;
    }

    double G1 = std::tgamma((df + 1)/2.0);
    double G2 = std::tgamma(df/2.0);
    double sqrt_df = std::sqrt(df * M_PIl);
    double t = G1/(sqrt_df * G2) * std::pow(1 + x*x/df, -(df+1)/2.0);
    return t;
}

double statistics::tCDF(double x, double df) {
    if (x == 0) return 0.5;
    auto f = [df](double t) -> double {return tPDF(t,df);};
    if (x < 0) return 0.5 - numerics::gIntegrate(f, x, 0, 1e-10);
    else return 0.5 + numerics::gIntegrate(f, 0, x, 1e-10);
}

double statistics::tQ(double p, double df) {
    double guess;
    if (p == 0.5) return 0;
    else if (p < 0.5) guess = -1;
    else guess = 1;

    auto g  = [p,df](double t) -> double {return tCDF(t,df) - p;};
    auto dg = [df](double t) -> double {return tPDF(t,df);};
    return numerics::newton(g, dg, guess);
}

//--- Chi Squared distribution ---//
double statistics::chiPDF(double x, double df) {
    if (x < 0) { // invalid input
        std::cerr << "chiPDF error(): the chi squared distribution has no support for x < 0. Returning 0 instead." << std::endl;
        return 0;
    }
    double k = df/2;
    double p = std::pow(2.0,k)*tgamma(k);
    p = 1/p;
    p *= std::pow(x, k-1) * std::exp(-x/2);
    return p;
}

double statistics::chiCDF(double x, double df) {
    if (x < 0) { // invalid input
        std::cerr << "chiCDF() error: the chi squared distribution has no support for x < 0. returning 0 instead." << std::endl;
        return 0;
    }
    auto f = [df](double t) -> double {return chiPDF(t,df);};
    return numerics::gIntegrate(f, 0, x, 1e-10);
}

double statistics::chiQ(double p, double df) {
    if (p < 0 || p > 1) {
        std::cerr << "chiQ() error: probabilities must be bounded between 0 and 1. p input was: " << p << ". returning 0 instead." << std::endl;
        return 0;
    }
    double guess; double k = (df-2 < 0) ? (0) : (df - 2);
    if (p < 0.5) {
        guess = 0.5*k;
    } else {
        guess = 1.5*k;
    }

    auto g = [p,df](double x) -> double {return chiCDF(x,df) - p;};
    auto dg = [df](double x) -> double {return chiPDF(x,df);};
    return numerics::newton(g,dg,guess);
}

//--- generic quantile function ---//
std::function<double(double,double)> statistics::quantile(std::function<double(double)> cdf, std::function<double(double)> pdf) {
    std::function<double(double,double)> Q = [cdf,pdf](double p, double guess) -> double {
        auto f = [cdf,p](double x) -> double {return cdf(x) - p;};
        return numerics::newton(f, pdf, guess);
    };
    return Q;
}