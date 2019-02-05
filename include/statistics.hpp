#include "numerics.hpp"

#define M_1_SQRT2PI 0.3989422804014326779399460599343818684758586311649346576L

using namespace arma;

/* Copyright 2019 Amit Rotem

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

namespace statistics {
    //--- enumerator ----------//
        enum hypothesis {
            NEQ,
            LESS,
            GREATER,
            HOMOGENEITY,
            INDEPENDENCE,
            GOF
        };

        enum test {
            z1,
            z2,
            z2_paired,
            t1,
            t2,
            t2_paired
        };
    //--- output structs ------//
        class stats {
            public:
            double x_bar;
            double x_sd;
            double n;
            stats() {
                x_bar = 0;
                x_sd  = 1;
                n     = 1;
            }
            stats(double m, double s, double size) {
                x_bar = m;
                x_sd  = s;
                n     = size;
            }
        };

        class mean_test {
            public:
            double score;
            double df;
            double p;
            double test_mu;
            std::pair<double,double> conf_interval;
            double conf_level;
            enum hypothesis H1;
            enum test T;
            stats S;
            mean_test() {
                score = 0;
                df = 0;
                p = 0;
                test_mu = 0;
                conf_interval = {0,0};
                conf_level = 0;
                H1 = NEQ;
                T = z1;
            }
        };

        class prop_test {
            public:
            double score;
            double df;
            double p;
            double test_p0;
            std::pair<double,double> conf_interval;
            double conf_level;
            enum hypothesis H1;
            double p1;
            double p1_sd;
            prop_test() {
                score = 1;
                df = 1;
                p = 1;
                test_p0 = 1;
                conf_interval = {1,1};
                conf_level = 1;
                H1 = NEQ;
            }
        };

        class category_test {
            public:
            double X2;
            double df;
            double p;
            hypothesis H1;
            category_test() {
                X2 = 0;
                df = 0;
                p = 0;
                H1 = hypothesis::GOF;
            }
        };

        std::ostream& operator<<(std::ostream &out, const mean_test &x);
        void LaTeX_print(std::ostream &out, const mean_test &x);
        std::ostream& operator<<(std::ostream &out, const prop_test &x);
        void LaTeX_print(std::ostream &out, const prop_test &x);
        std::ostream& operator<<(std::ostream &out, const category_test &x);
        void LaTeX_print(std::ostream &out, const category_test &x);
    //--- basic probability ---//
        double normalPDF(double);   // normal pdf
        double normalCDF(double);   // normal cdf
        double normalQ(double);     // normal quantile function

        double tPDF(double, double);    // student's t pdf
        double tCDF(double, double);    // student's t cdf
        double tQ(double, double);      // student's t quantile function

        double chiPDF(double, double);  // chi squared pdf
        double chiCDF(double, double);  // chi squared cdf
        double chiQ(double, double);    // chi squared quantile function

        std::function<double(double,double)> quantile(const numerics::dfunc&, const numerics::dfunc&); // quantile function of any distribution
    //--- basic statistics ----//
        template<class X>
        double mean(X &x) {
            double x_bar = 0;
            size_t n = x.size();
            for (size_t i(0); i < n; ++i) {
                x_bar += x.at(i);
            }
            x_bar /= n;
            return x_bar;
        }

        template<class X>
        double var(X &x) {
            double x_var = 0;
            double x_bar = mean<X>(x);
            size_t n = x.size();
            for (size_t i(0); i < n; ++i) {
                x_var += std::pow(x.at(i) - x_bar, 2);
            }
            x_var /= n-1;
            return x_var;
        }

        template<class X>
        double median(X &x) {
            std::sort(x.begin(), x.end());
            size_t n = x.size();
            if (n%2 == 0) { // even elements, need intermediate value
                int n2f = std::floor(n/2.0);
                int n2c = n2f + 1;
                double med = x(n2f) + x(n2c);
                return med/2;
            } else {
                int n2 = n/2;
                return x(n2);
            }
        }

        template<class X>
        double range(X &x) {
            double max = *std::max_element(x.begin(), x.end());
            double min = *std::min_element(x.begin(), x.end());
            return max - min;
        }

        template<class X>
        stats data_to_stats(X &x) {
            double x_bar = mean<X>(x);
            double x_sd = std::sqrt(  var<X>(x)  );
            int n = x.size();

            stats x_stats(x_bar, x_sd, n);
            return x_stats;
        }

        template<class X>
        std::pair<stats,stats> data_to_stats(X &x1, X &x2) {
            double x1_bar = 0; double x2_bar = 0;
            unsigned int n1 = x1.size(); unsigned int n2 = x2.size();
            for (unsigned int i(0); i < std::max(n1,n2); ++i) {
                if (i < n1) {
                    x1_bar += x1.at(i);
                }
                if (i < n2) {
                    x2_bar += x2.at(i);
                }
            }
            x1_bar /= n1;
            x2_bar /= n2;

            double x1_sd = 0;
            double x2_sd = 0;
            for (unsigned int i(0); i < std::max(n1,n2); ++i) {
                if (i < n1) {
                    x1_sd += std::pow(x1.at(i) - x1_bar, 2);
                }
                if (i < n2) {
                    x2_sd += std::pow(x2.at(i) - x2_bar, 2);
                }
            }
            x1_sd /= (n1-1);
            x2_sd /= (n2-1);

            stats x1_stats(x1_bar, x1_sd, n1);
            stats x2_stats(x2_bar, x2_sd, n2);
            return {x1_stats, x2_stats};
        }

    //--- z-tests -------------//
        mean_test z_test(stats &x, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95); // one sample z-test using stats
        
        template<class X>
        mean_test z_test(X &x, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95) { // one sample using data
            stats x_stats = data_to_stats(x);
            return z_test(x_stats, mu, H1, confidence);
        }
        
        mean_test z_test(stats &x1, stats &x2, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95); // two samp using stats

        template<class X>
        mean_test z_test(X &x1, X &x2, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95, bool paired = false) { // two samp using data
            if (paired) { // paired z-test
                if (x1.size() == x2.size()) {
                    X paired_x(x1.size());
                    for (unsigned int i(0); i < x1.size(); ++i) {
                        paired_x.at(i) = x1.at(i) - x2.at(i);
                    }
                    stats x = data_to_stats(paired_x);
                    mean_test TEST = z_test(x,mu,H1,confidence);
                    TEST.T = z2_paired;
                    return TEST;
                } else { // error data not truly paired
                    std::cerr << "paired z_test() error: samples are not of the same size." << std::endl;
                    mean_test x;
                    return x;
                }
            } else { // unpaired
                std::pair<stats,stats> x_stat = data_to_stats(x1,x2);
                stats x1_stats = x_stat.first;
                stats x2_stats = x_stat.second;
                return z_test(x1_stats, x2_stats, mu, H1, confidence);
            }
        }
    //--- t-tests -------------//
        mean_test t_test(stats &x, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95); // one sample t-test with stats

        template<class X>
        mean_test t_test(X &x, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95) { //one samp with data
            stats x_stats = data_to_stats(x);
            return t_test(x_stats, mu, H1, confidence);
        }

        mean_test t_test(stats &x1, stats &x2, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95); // two sample t-test using stats
        
        template<class X>
        mean_test t_test(X &x1, X &x2, double mu = 0, enum hypothesis H1 = NEQ, double confidence = 0.95, bool paired = false) { // two sample t-test with data
            if (paired) {
                if (x1.size() == x2.size()) {
                    X paired_x(x1.size());
                    for (unsigned int i(0); i < x1.size(); ++i) {
                        paired_x.at(i) = x1.at(i) - x2.at(i);
                    }
                    stats x = data_to_stats(paired_x);
                    mean_test TEST = t_test(x, mu, H1, confidence);
                    TEST.T = t2_paired;
                    return TEST;
                } else {
                    std::cerr << "paired t_test() error: samples are not of the same size." << std::endl;
                    mean_test x;
                    return x;
                }
            } else {
                std::pair<stats,stats> x_stat = data_to_stats<X>(x1,x2);
                stats x1_stats = x_stat.first;
                stats x2_stats = x_stat.second;
                return t_test(x1_stats, x2_stats, mu, H1, confidence);
            }
        }
    //--- p-tests -------------//
        prop_test p_test(int sucesses, int n, double p0, enum hypothesis H1 = NEQ, double confidence = 0.95);
    //--- chi squared test ----//
        category_test chi_test(mat& observed, hypothesis H1);

        template<class X>
        category_test chi_test(X& observed, X& expected, hypothesis H1 = hypothesis::INDEPENDENCE) {
            if (observed.size() != expected.size()) { // error
                std::cerr << "chi_test() error : observed and expected samples are not the same size." << std::endl
                        << "observed.size() = " << observed.size() << " expected.size() = " << expected.size() << std::endl;
                return category_test();
            }
            X squares = pow(expected - observed, 2.0) / expected;
            double chi_sq = sum(squares);
            double df = observed.size() - 1;
            double p = chiCDF(chi_sq, df);
            category_test test;
            test.X2 = chi_sq;
            test.df = df;
            test.p  = p;
            test.H1 = H1;
            return test;
        }
    //--- permutation test ----//
        double perm_test(arma::vec& x1, arma::vec& x2, enum hypothesis H1 = hypothesis::NEQ, unsigned int num_trials = 1e4);
};