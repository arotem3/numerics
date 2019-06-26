// --- interpolation ---------- //
class cubic_interp {
    private:
    int n;
    arma::mat b;
    arma::mat c;
    arma::mat d;
    arma::vec x;
    arma::mat y;

    public:
    cubic_interp();
    cubic_interp(std::istream&);
    cubic_interp(const arma::vec&, const arma::mat&);
    cubic_interp& fit(const arma::vec& X, const arma::mat& Y) {
        cubic_interp(X,Y);
        return *this;
    }
    arma::mat operator()(const arma::vec&);
    arma::mat predict(const arma::vec&);
    arma::vec data_X();
    arma::mat data_Y();
    void save(std::ostream&);
    void load(std::istream&);
};

class poly_interp {
    private:
    arma::vec x;
    arma::mat y;
    arma::mat p;

    public:
    poly_interp();
    poly_interp(const arma::vec&, const arma::mat&);
    poly_interp(std::istream&);
    poly_interp& fit(const arma::vec&, const arma::mat&);
    void load(std::istream&);
    void save(std::ostream&);
    arma::mat operator()(const arma::vec&);
    arma::mat predict(const arma::vec&);
    arma::mat coefficients() const;
    arma::vec data_X();
    arma::mat data_Y();
};

arma::mat lagrange_interp(const arma::vec&, const arma::mat&, const arma::vec&, bool normalize = false);
arma::mat sinc_interp(const arma::vec&, const arma::mat&, const arma::vec&);

arma::vec polyder(const arma::vec& p, uint k = 1);
arma::vec polyint(const arma::vec& p, double c = 0);