#include "statistics.hpp"

//--- one sample hypothesis z-test on mean ---//
//----- x  : sample statistics ---------------//
//----- mu : theoretical mean ----------------//
//----- H1 : alternative hypothesis ----------//
//----- confidence : confidence level --------//
statistics::mean_test statistics::z_test(stats &x, double mu, enum hypothesis H1, double confidence) {
    double s = x.x_sd/std::sqrt(x.n); // sigma/sqrt(n)
    double z = (x.x_bar - mu)/s; // z-score
    double alpha, z_a;
    std::pair<double,double> interval;
    double p = 0;
    if (H1 == NEQ) {
        p = 2*normalCDF(z); // 2*p(x < -|z|) area on tails
        alpha = (1 - confidence)/2.0;
        z_a = std::abs(normalQ(alpha));
        interval.first = x.x_bar - z_a*s;
        interval.second = x.x_bar + z_a*s;
    } else if (H1 == LESS) {
        p = 1 - normalCDF(z); // 1 - p(x < z) area on right
        alpha = (1 - confidence);
        z_a = std::abs(normalQ(alpha));
        interval.first = -INFINITY;
        interval.second = x.x_bar + z_a*s;        
    } else if (H1 == GREATER) {
        p = normalCDF(z); // p(x < z) area on left
        alpha = (1 - confidence);
        z_a = std::abs(normalQ(alpha));
        interval.first = x.x_bar - z_a*s;
        interval.second = INFINITY;
    }

    mean_test TEST;
    TEST.score = z;
    TEST.p = p;
    TEST.test_mu = mu;
    TEST.conf_interval = interval;
    TEST.conf_level = confidence;
    TEST.H1 = H1;
    TEST.T = z1;
    TEST.S = x;
    return TEST;
}

//--- unpaired two sample hypothesis z-test on difference of means ---//
//----- x1, x2  : sample statistics for each data set ----------------//
statistics::mean_test statistics::z_test(stats &x1, stats &x2, double mu, enum hypothesis H1, double confidence) {
    stats paired_x(
        x1.x_bar - x2.x_bar,
        std::sqrt( std::pow(x1.x_sd,2)/x1.n + std::pow(x1.x_sd,2)/x1.n ),
        1
    );
    mean_test TEST = z_test(paired_x, mu, H1, confidence);
    TEST.T = z2;
    return TEST;
}

//--- one sample hypothesis t-test on mean ---//
statistics::mean_test statistics::t_test(stats &x, double mu, enum hypothesis H1, double confidence) {
    double s = x.x_sd/std::sqrt(x.n); // sigma/sqrt(n)
    double t = (x.x_bar - mu)/s; // t-score
    double df = x.n - 1;
    double alpha, t_a;
    std::pair<double,double> interval;
    double p = 0;
    if (H1 == NEQ) {
        p = 2*tCDF(-std::abs(t), df); // 2*p(x < -|t|) area on tails
        alpha = (1 - confidence)/2.0;
        t_a = std::abs(tQ(alpha, df));
        interval.first = x.x_bar - t_a*s;
        interval.second = x.x_bar + t_a*s;
    } else if (H1 == LESS) {
        p = 1 - tCDF(t, df); // p(x > t) = 1 - p(x < t) area on right
        alpha = (1 - confidence);
        t_a = std::abs(tQ(alpha, df));
        interval.first = -INFINITY;
        interval.second = x.x_bar + t_a*s;        
    } else if (H1 == GREATER) {
        p = tCDF(t, df); // p(x < t) area on left
        alpha = (1 - confidence);
        t_a = std::abs(tQ(alpha, df));
        interval.first = x.x_bar - t_a*s;
        interval.second = INFINITY;
    }

    mean_test TEST;
    TEST.score = t;
    TEST.df = df;
    TEST.p = p;
    TEST.test_mu = mu;
    TEST.conf_interval = interval;
    TEST.conf_level = confidence;
    TEST.H1 = H1;
    TEST.T = t1;
    TEST.S = x;
    return TEST;
}

//--- unpaired two sample hypothesis t-test on difference of means ---//
statistics::mean_test statistics::t_test(stats &x1, stats &x2, double mu, enum hypothesis H1, double confidence) {
    double df_top = std::pow(x1.x_sd,2)/x1.n + std::pow(x2.x_sd,2)/x2.n;
    df_top = std::pow(df_top,2);
    double df_bot = std::pow(std::pow(x1.x_sd,2)/x1.n, 2)/(x1.n - 1) + std::pow(std::pow(x2.x_sd,2)/x2.n, 2)/(x2.n - 1);
    double df = df_top/df_bot;
    double var = std::pow(x1.x_sd,2)/x1.n + std::pow(x2.x_sd,2)/x2.n;
    stats paired_x(
        x1.x_bar - x2.x_bar,
        std::sqrt(var),
        df + 1.0 // we subtract the +1 in the t_test function
    );
    mean_test TEST = t_test(paired_x, mu, H1, confidence);
    TEST.T = t2;
    return TEST;
}

//--- one sample proportions test ---//
//----- S  : number of successes ----//
//----- n  : total trials -----------//
//----- p0 : expected proportion ----//
//----- H1 : alternative hypothesis -//
//----- confidence : conf level -----//
statistics::prop_test statistics::p_test(int S, int n, double p0, enum hypothesis H1, double confidence) {
    double p1 = (S + 2.0)/(n + 4.0);
    double s = std::sqrt(  (1-p1)*p1/n  ); // p1 is for confidence interval
    double pp = (double)S/n; // pp is for testing
    double sp = std::sqrt(  (1-pp)*pp/n  );
    double t = (pp - p0)/sp;
    double df = n - 1.0;
    double alpha, t_a;
    std::pair<double, double> interval;
    double p = 0;

    if (H1 == NEQ) {
        p = 2*tCDF(t, df); // 2*p(x < -|t|) area on tails
        alpha = (1 - confidence)/2.0;
        t_a = std::abs(tQ(alpha, df));
        interval.first = p1 - t_a*s;
        interval.second = p1 + t_a*s;
    } else if (H1 == LESS) {
        p = 1 - tCDF(t, df); // 1 - p(x < t) area on right
        alpha = (1 - confidence);
        t_a = std::abs(tQ(alpha, df));
        interval.first = 0;
        interval.second = p1 + t_a*s;        
    } else if (H1 == GREATER) {
        p = tCDF(t, df); // p(x < t) area on left
        alpha = (1 - confidence);
        t_a = std::abs(tQ(alpha, df));
        interval.first = p1 - t_a*s;
        interval.second = 1;
    }
    if (interval.first < 0) interval.first = 0;
    if (interval.second > 1) interval.second = 1;

    prop_test TEST;
    TEST.score = t;
    TEST.df = df;
    TEST.p = p;
    TEST.test_p0 = p0;
    TEST.conf_interval = interval;
    TEST.conf_level = confidence;
    TEST.H1 = H1;
    TEST.p1 = pp;
    TEST.p1_sd = sp*std::sqrt(n);
    return TEST;
}

//--- two sample permuations test for difference of means ---//
//----- x1 : first data set ---------------------------------//
//----- x2 : second data set --------------------------------//
//----- H1 : alternative hypothesis -------------------------//
//-----num_trials : number of possible permutations ---------//
double statistics::perm_test(arma::vec& x1, arma::vec& x2, enum hypothesis H1, unsigned int num_trials) {
    //--- (0.a) calculate important statistics
    arma::arma_rng::set_seed_random();
    size_t n1 = x1.size();
    size_t n2 = x2.size();
    double m = arma::mean(x1) - arma::mean(x2);
    double v1 = arma::var(x1)/n1;
    double v2 = arma::var(x2)/n2;
    double s = std::sqrt(v1 + v2);
    double t = m/s;

    //--- (0.b) group data
    arma::vec x = arma::join_cols(x1,x2);
    arma::vec perm_t_vals(num_trials, arma::fill::zeros);

    //--- (1) run permutations
    for (size_t i(0); i < num_trials; ++i) {
        arma::vec shuff_x = arma::shuffle(x);
        arma::vec samp1 = shuff_x(arma::span(0,n1-1));
        arma::vec samp2 = shuff_x(arma::span(n1,n1+n2-1));
        double m_new = arma::mean(samp1) - arma::mean(samp2);
        double v1_new = arma::var(samp1)/n1;
        double v2_new = arma::var(samp2)/n2;
        double s_new = std::sqrt(v1_new + v2_new);
        perm_t_vals(i) = m_new/s_new;
    }

    //--- (2) calculate p-value
    double p;
    if (H1 == hypothesis::NEQ) p = arma::sum(perm_t_vals < -std::abs(t)) + arma::sum(perm_t_vals > std::abs(t));
    else if (H1 == hypothesis::LESS) p = arma::sum(perm_t_vals > t);
    else if (H1 == hypothesis::GREATER) p = arma::sum(perm_t_vals < t);
    p /= num_trials;
    return p;
}