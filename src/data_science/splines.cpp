#include "numerics.hpp"

/* splines(poly_degree=1) : initialize splines object.
 * --- poly_degree : degree polynomial used in fit, the default value results in a linear fit. This parameter should be taken so m is small, a good rule of thumb is: (2*poly_degree - x.n_cols) >= 1 */
numerics::splines::splines(uint poly_degree) : smoothing_param(_lambda), eff_df(_df), RMSE(_rmse), residuals(_res), poly_coef(_d), rbf_coef(_c), data_X(_X), data_Y(_Y), rbf_eigenvals(_eigvals), rbf_eigenvecs(_eigvecs) {
    _deg = poly_degree;
    _lambda = -1;
    _df = -1;
    _fitted = false;
    _use_df = false;
}

/* set_smoothing_param(lambda) : set the smoothing parameter for performing ridge regression on the kernel. If one is never set, it will be determined by smoothed cross-validation. If the fit() function has been called, this function will raise an error.
--- lambda : smoothing parameter >= 0 */
void numerics::splines::set_smoothing_param(double lambda) {
    if (_fitted) {
        std::cerr << "splines::set_smoothing_param() error: object has already been fitted, the smoothing parameter is " << _lambda << "\n";
        return;
    }
    if (lambda < 0) {
        std::cerr << "splines::set_smoothing_param() error: smoothing parameter must be non-negative, lambda provided = " << lambda << "\n";
        return;
    }
    _lambda = lambda;
    _use_df = false;
}

/* set_degrees_of_freedom(df) : set the effective degrees of freedom for the spline object, must be in bounds: 1 < x < num_obs
 * --- df : effective degrees of freedom. */
void numerics::splines::set_degrees_of_freedom(double df) {
    if (_fitted) {
        std::cerr << "splines::set_degrees_of_freedom() error: object has already been fitted, the effective degrees of freedom are " << _df << "\n";
        return;
    }
    if (df < 1) {
        std::cerr << "splines::set_degrees_of_freedom() error: effective degrees of freedom must be greater than 1, df provided = " << df << "\n";
        return;
    }
    _df = df;
    _use_df = true;
}

/* splines(in) : initialize spline object by loading object from a stream. */
numerics::splines::splines(const std::string& in) : smoothing_param(_lambda), eff_df(_df), RMSE(_rmse), residuals(_res), poly_coef(_d), rbf_coef(_c), data_X(_X), data_Y(_Y), rbf_eigenvals(_eigvals), rbf_eigenvecs(_eigvecs) {
    load(in);
}

/* fit(x,y) : fit splines object.
 * --- x : array of indpendent variable data, where each row is data point.
 * --- y : array of dependent variable data, where each row is a data point. */
void numerics::splines::fit(const arma::mat& x, const arma::mat& y) {
    if (_fitted) {
        std::cerr << "splines::fit() error: object has already been fitted.\n";
        return;
    }
    if (x.n_rows != y.n_rows) {
        std::cerr << "splines::fit() error: number of observations in x (" << x.n_rows << ") does not equal the number of observations in y (" << y.n_rows << ").\n";
        return;
    }
    _dim = x.n_cols;

    uint n_obs = x.n_rows;
    _X = x;
    _Y = y;

    // compute valuable polynomial terms with coefficients
    gen_monomials();
    arma::mat P = eval_poly(_X);
    numerics::lasso_cv Lasso;

    Lasso.fit(P, _Y, true);
    arma::mat d = Lasso.coef;
    
    // keep only nonzero coefficient terms
    std::vector<std::vector<uint>> mons;
    arma::uvec nnzi = arma::zeros<arma::uvec>(d.n_rows);
    nnzi(0) = 1; // always keep the intercept
    for (uint i=0; i < d.n_rows-1; ++i) {
        if (arma::any(arma::abs(d.row(i+1)) > 1e-5)) {
            mons.push_back(std::move(_monomials.at(i)));
            nnzi(i+1) = 1;
        }
    }
    _monomials = std::move(mons);
    nnzi = arma::find(nnzi==1);
    _d = d.rows(nnzi);
    P = P.cols(nnzi);
    uint nnz = nnzi.n_elem;

    if (2*nnz > n_obs) {
        std::cerr << "splines::fit() warning: too many significant polynomial terms, solution system may be ill-conditioned, consider reducing the degree.\n";
    }

    // produce RBF kernel
    arma::mat K = eval_rbf();
    arma::eig_sym(_eigvals, _eigvecs, K);
    arma::vec D2 = arma::pow(_eigvals, 2);
    
    if (_use_df) { // need to solve for lambda
        if (_df > n_obs) {
            std::cerr << "splines::fit() warning: requested degrees of freedom exceed the feasible range, setting _df ~ " << n_obs << "\n";
            _lambda = 0;
        } else if (_df == n_obs) {
            _lambda = 0;
        } else {
            auto g = [&](double L) -> double {
                return _df - arma::sum(D2 / (D2 + L));
            };
            double Dmin = D2.min();
            _lambda = numerics::bisect(g, Dmin, D2.max(), Dmin/2); // solution is guaranteed to be bounded in the range of eigenvalues
        }
    }

    if (_lambda >= 0) { // valid lambda provided
        if (_lambda == 0) { // interpolation
            _c = _eigvecs * arma::diagmat(1 / _eigvals) * _eigvecs.t() * _Y;
            if (!_use_df) _df = n_obs;
        } else {
            _c = _eigvecs * arma::diagmat(_eigvals / (D2 + _lambda)) * _eigvecs.t() * (_Y - P*_d);
            if (!_use_df) _df = arma::sum(D2 / (D2 + _lambda));
        }
    } else { // cross-validation
        arma::mat er = (_Y - P*_d);
        arma::mat Ver = _eigvecs.t() * er;
        auto GCV = [&](double lam) -> double {
            _c = _eigvecs * arma::diagmat(_eigvals / (D2 + lam)) * Ver;
            _df = arma::sum(D2 / (D2 + lam));
            double rmse = arma::norm(er - K*_c,"fro");
            return std::pow(rmse / (n_obs-_df), 2) * n_obs;
        };
        _lambda = numerics::fminbnd(GCV, 0, 1e4);
    }
    _res = _Y - predict(_X);
    _rmse = arma::norm(_res, "fro") / _res.n_elem;
}

/* rbf(xgrid) : build radial basis kernel matrix from fitted data evaluated at a new set of points.
 * --- xgrid : set of points to evaluate RBFs on. */
arma::mat numerics::splines::eval_rbf(const arma::mat& xgrid) {
    uint n = _X.n_rows;
    uint ngrid = xgrid.n_rows;
    uint k_order = 2*(_deg+1) - _Y.n_cols;
    arma::mat K = arma::zeros(ngrid, n);
    for (uint i=0; i < ngrid; ++i) {
        for (uint j=0; j < n; ++j) {
            double z = arma::norm(xgrid.row(i) - _X.row(j));
            if (_dim%2 == 0) {
                if (z < 1) K(i,j) = std::pow(z, k_order-1) * std::log(std::pow(z,z));
                else K(i,j) = std::pow(z, k_order) * std::log(z);
            } else K(i,j) = std::pow(z, k_order);
        }
    }
    return K;
}

/* eval_rbf() : build radial basis kernel matrix from fitted data evaluated on the original data, since the matrix is symmetric we only need to compute half as many values. */
arma::mat numerics::splines::eval_rbf() {
    uint n = _X.n_rows;
    uint k_order = 2*(_deg+1) - _Y.n_cols;
    arma::mat K = arma::zeros(n, n);
    for (uint i=0; i < n; ++i) {
        for (uint j=0; j < i; ++j) {
            double z = arma::norm(_X.row(i) - _X.row(j));
            if (_dim%2 == 0) {
                if (z < 1) K(i,j) = std::pow(z, k_order-1) * std::log(std::pow(z,z));
                else K(i,j) = std::pow(z, k_order) * std::log(z);
            } else K(i,j) = std::pow(z, k_order);
        }
    }
    return arma::symmatl(K);
}

/* eval_poly(xgrid) : build polynomial basis matrix evaluated at a set of points.
 * --- xgrid : set of points to evaluate polynomial basis on. */
arma::mat numerics::splines::eval_poly(const arma::mat& xgrid) {
    uint n = xgrid.n_rows;
    uint num_mons = _monomials.size() + 1;
    arma::mat P = arma::ones(n, num_mons); // first column is an intercept i.e. [1, ..., 1].t()
    for (uint i=1; i < num_mons; ++i) {
        for (uint r : _monomials.at(i-1) ) {
            P.col(i) %= xgrid.col(r);
        }
    }
    return P;
}

/* gen_monomials() : __private__ constructs list of all monomials of requested order and dimension. */
void numerics::splines::gen_monomials() {
    std::queue<std::vector<uint>> Q;
    std::set<std::vector<uint>> S;
    for (uint i=0; i < _dim; ++i) {
        std::vector<uint> str = {i};
        Q.push(str);
    }
    while ( !Q.empty() ) {
        std::vector<uint> str = Q.front();
        if (str.size() > _deg) { // invalid monomial
            Q.pop();
            continue;
        }
        if (S.count(str) > 0) { // discovered
            Q.pop();
            continue;
        } else { // new node
            S.insert(str);
            for (uint i=0; i < _dim; ++i) {
                std::vector<uint> str2 = str;
                str2.push_back(i);
                std::sort( str2.begin(), str2.end() );
                Q.push(str2);
            }
            Q.pop();
        }
    }
    _monomials.clear();
    for (const std::vector<uint>& str : S) _monomials.push_back(str);
}

/* predict(xgrid) : evaluate spline fit on a set of new points.
 * --- xgrid : set of points to evaluate spline fit on. */
arma::mat numerics::splines::predict(const arma::mat& xgrid) {
    if (xgrid.n_cols != _dim) {
        std::cerr << "splines::predict() error: dimension of new data do not match fitted data dimenstion." << std::endl
                  << "dim(fitted data) = " << _dim << " =/= " << xgrid.n_cols << " = dim(new data)" << std::endl;
        return arma::mat();
    }
    if (_lambda == 0) {
        return eval_rbf(xgrid)*_c;
    } else {
        return eval_rbf(xgrid)*_c + eval_poly(xgrid)*_d;
    }
}

/* splines::(xgrid) : same as predict(const arma::mat&). */
arma::mat numerics::splines::operator()(const arma::mat& xgrid) {
    return predict(xgrid);
}

/* load(in) : load in object from file. */
void numerics::splines::load(const std::string& fname) {
    std::ifstream in(fname);
    if (in.fail()) {
        std::cerr << "splines::load() error: failed to open file.\n";
        return; 
    }
    uint c_rows, d_rows, nx, ny, nm;
    in >> nx >> ny >> _dim >> _deg >> _lambda >> _df >> _rmse;

    in >> c_rows;
    _c = arma::zeros(ny, c_rows);
    for (uint i=0; i < ny; ++i) {
        for (uint j=0; j < c_rows; ++j) in >> _c(i,j);
    }
    _c = _c.t();

    in >> d_rows;
    _d = arma::zeros(ny, d_rows);
    for (uint i=0; i < ny; ++i) {
        for (uint j=0; j < d_rows; ++j) in >> _d(i,j);
    }
    _d = _d.t();

    _X = arma::zeros(_dim,nx);
    for (uint i=0; i < _dim; ++i) {
        for (uint j=0; j < nx; ++j) in >> _X(i,j);
    }
    _X = _X.t();

    _Y = arma::zeros(ny,nx);
    for (uint i=0; i < ny; ++i) {
        for (uint j=0; j < nx; ++j) in >> _Y(i,j);
    }
    _Y = _Y.t();

    _eigvecs = arma::zeros(nx,nx);
    _eigvals = arma::zeros(nx);
    for (uint i=0; i < nx; ++i) {
        for (uint j=0; j < nx; ++j) in >> _eigvecs(i,j);
    }
    _eigvecs = _eigvecs.t();

    for (uint i=0; i < nx; ++i) in >> _eigvals(i);

    _monomials.clear();
    while (true) {
        in >> nm;
        if (in.eof()) break;
        std::vector<uint> str(nm);
        for (uint i=0; i < nm; ++i) {
            in >> str.at(i);
        }
        _monomials.push_back(str);
    }
    in.close();
}

/* save(out) : save object to file. */
void numerics::splines::save(const std::string& fname) {
    std::ofstream out(fname);
    if (out.fail()) {
        std::cerr << "splines::save() error: failed to save instance to file.\n";
        return;
    }
    out << std::setprecision(10);
    out << _X.n_rows << " " << _Y.n_cols << " " << _dim << " "
        << _deg << " " << _lambda << " " << _df << " " << _rmse << "\n\n";

    out << _c.n_rows << "\n";
    _c.t().raw_print(out);

    out << _d.n_rows << "\n";
    _d.t().raw_print(out);

    _X.t().raw_print(out);
    _Y.t().raw_print(out);
    _eigvecs.raw_print(out);
    _eigvals.t().raw_print(out);

    for (std::vector<uint>& str : _monomials) {
        out << str.size() << " ";
        for (uint i : str) out << " " << i;
        out << "\n";
    }
    out.close();
}