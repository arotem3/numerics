// --- interpolation ---------- //
    class CubicInterp {
        private:
        int n;
        arma::mat b;
        arma::mat c;
        arma::mat d;
        arma::vec x;
        arma::mat y;

        public:
        CubicInterp();
        CubicInterp(std::istream&);
        CubicInterp(const arma::vec&, const arma::mat&);
        arma::mat operator()(const arma::vec&);
        arma::mat operator()(uint);
        void save(std::ostream&);
        void load(std::istream&);
    };

    class polyInterp {
        private:
        arma::vec x;
        arma::mat p;

        public:
        polyInterp();
        polyInterp(const arma::vec&, const arma::mat&);
        polyInterp(std::istream&);
        void load(std::istream&);
        void save(std::ostream&);
        arma::mat operator()(const arma::vec&);
    };
    
    arma::mat nearestInterp(const arma::vec&, const arma::mat&, const arma::vec&);
    arma::mat linearInterp(const arma::vec&, const arma::mat&, const arma::vec&);
    arma::mat lagrangeInterp(const arma::vec&, const arma::mat&, const arma::vec&);
    arma::mat sincInterp(const arma::vec&, const arma::mat&, const arma::vec&);