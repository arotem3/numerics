// --- utitility -------------- //
namespace numerics_private_utility {
    // --- integral constants
    const long double X4[4] = {-1, -0.447213595499958, 0.447213595499958, 1};
    const long double W4[4] = {0.166666666666667, 0.833333333333333, 0.833333333333333, 0.166666666666667};

    const long double X7[7] = {-1, -0.468848793470714, -0.830223896278567, 0, 0.830223896278567, 0.468848793470714, 1};
    const long double W7[7] = {0.047619047619048, 0.431745381209863, 0.276826047361566, 0.487619047619048, 0.276826047361566, 0.431745381209863, 0.047619047619048};

    // --- cyclic queue
    class cyc_queue {
        private:
        uint max_elem;
        uint size;
        uint head;

        public:
        arma::mat A;
        cyc_queue(uint num_rows, uint max_size);
        void push(const arma::vec& x);
        arma::vec operator()(uint i);
        arma::vec end();
        int length();
        int col_size();
        void clear();
        arma::mat data();
    };
}

// --- misc
inline int mod(int a, int b) {
    return (a%b + b)%b;
}

void meshgrid(arma::mat&, arma::mat&, const arma::vec&, const arma::vec&);
void meshgrid(arma::mat&, const arma::vec&);

arma::vec sample_from(int, const arma::vec&, const arma::vec& labels = arma::vec());
double sample_from(const arma::vec&, const arma::vec& labels = arma::vec());