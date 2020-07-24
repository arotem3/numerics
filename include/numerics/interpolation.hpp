#ifndef NUMERICS_INTERPOLATION_HPP
#define NUMERICS_INTERPOLATION_HPP

class Interpolator {
    protected:
    double _lb, _ub;
    void _check_xy(const arma::vec& x, const arma::mat& y) {
        if (x.n_rows != y.n_rows) {
            throw std::invalid_argument("dimension mismatch, x.n_rows (=" + std::to_string(x.n_rows) + ") != y.n_rows (=" + std::to_string(y.n_rows) + ")");
        }
    }
    void _check_x(const arma::vec& x) {
        for (u_long i=0; i < x.n_elem-1; ++i) {
            for (u_long j=i+1; j < x.n_elem; ++j) {
                if (x(i) == x(j)) {
                    throw std::runtime_error("one or more x values are repeting, therefore no cubic interpolation exists for this data");
                }
            }
        }
    }
    void _check_range(const arma::vec& t) const {
        if ( (t.min() < _lb) || (_ub < t.max()) ) {
            throw std::range_error("one or more input value is outside the domain of the interpolation. No possible evaluation exists.");
        }
    }

    public:
    virtual void fit(const arma::vec& x, const arma::mat& y) = 0;
    virtual arma::mat predict(const arma::vec& t) const = 0;
    virtual void load(std::istream&) = 0;
    virtual void save(std::ostream&) const = 0;
};

class CubicInterp : public Interpolator {
    protected:
    arma::mat _b;
    arma::mat _c;
    arma::mat _d;
    arma::vec _x;
    arma::mat _y;

    public:
    const arma::vec& x;
    const arma::mat& y;
    CubicInterp() : x(_x), y(_y) {}

    /* cubic interpolation with one independent variable
     * --- x : independent variable
     * --- y : dependent variable */
    void fit(const arma::vec& X, const arma::mat& Y) override;

    /* predict(t) : evaluate interpolator like a function at specific values.
     * --- t : points to evaluate interpolation on. */
    arma::mat predict(const arma::vec&) const override;
    
    /* save(out) : save data structure to file.
     * --- out : file/output stream pointing to write data to. */
    void save(std::ostream&) const override;
    
    /* load(in) : load data structure from file
     * --- in : file/input stream pointing to top of cubic interpolator object */
    void load(std::istream&) override;
};

class HSplineInterp : public Interpolator {
    protected:
    arma::mat _a,_b,_y,_dy;
    arma::vec _x;

    public:
    const arma::vec& x;
    const arma::mat& y;
    const arma::mat& dy;
    HSplineInterp() : x(_x), y(_y), dy(_dy) {}
    void fit(const arma::vec& x, const arma::mat& y, const arma::mat& dy);
    void fit(const arma::vec& x, const arma::mat& y) override;
    arma::mat predict(const arma::vec&) const override;
    arma::mat predict_derivative(const arma::vec&) const;
    void save(std::ostream&) const override;
    void load(std::istream&) override;
};

class PolyInterp : public Interpolator {
    private:
    arma::mat _p;

    public:
    const arma::mat& coefficients;
    PolyInterp() : coefficients(_p) {}
    void fit(const arma::vec&, const arma::mat&) override;
    void load(std::istream&) override;
    void save(std::ostream&) const override;
    arma::mat predict(const arma::vec&) const override;
};

arma::mat lagrange_interp(const arma::vec&, const arma::mat&, const arma::vec&, bool normalize = false);
arma::mat sinc_interp(const arma::vec&, const arma::mat&, const arma::vec&);

arma::vec polyder(const arma::vec& p, uint k = 1);
arma::vec polyint(const arma::vec& p, double c = 0);

#endif