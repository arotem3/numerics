long double fact(int k) {
    if (k==1 || k==0) return 1.0;
    else return fact(k-1)*k;
}

arma::rowvec dq(double t, const arma::rowvec& q) {
    arma::rowvec DQ(2, arma::fill::zeros);
    DQ(0) = q(1);
    DQ(1) = q(0)*q(1)*q(1);
    return DQ;
}

double hyp_test::qnorm(double p) {
    if (p == 0.5) return 0;
    else if (p == 0) return -INFINITY;
    else if(p == 1) return INFINITY;
    double s;
    if (p < 0.5) s = -1;
    else s = 1;
    double k = s*1e-4;
    arma::vec t = {0.5, p};
    arma::mat U = {0, std::sqrt(2*M_PI)};
    numerics::rk5i(dq,t,U,k);
    U = U.tail_rows(1);
    return U(0);
}

double normalcdf_coef(int k) {
    int s = (k%2 == 0) ? (1) : (-1);
    long double kfinv = 1/fact(k);
    long double two2k = 1/std::pow<double,double>(2,k);
    long double odd2k = 1.0/(2*k+1);
    return s * kfinv * two2k * odd2k;
}

double hyp_test::pnorm(double x) { // area of normal dist on [-inf, a] = 0.5 + [0, a]
    if (x == 0) return 0.5;
    int k = 0;
    double next_term;
    double sum = 0;
    next_term = normalcdf_coef(k) * std::pow(x,2*k+1);
    while (std::abs(next_term) > 1e-10) {
        sum += next_term;
        k++;
        next_term = normalcdf_coef(k) * std::pow(x,2*k+1);
    }
    sum = 0.5 + sum*M_1_SQRT2PI;
    return sum;
}