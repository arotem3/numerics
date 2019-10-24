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

    namespace kd_tree_util {
        struct node {
            node() {
                ind = 0;
                left_child = nullptr;
                right_child = nullptr;
                parent = nullptr;
            }
            ~node() {
                if (left_child != nullptr) {
                    left_child->~node();
                    delete left_child;
                }
                if (right_child != nullptr) {
                    right_child->~node();
                    delete right_child;
                }
            }
            uint ind;
            node* left_child;
            node* right_child;
            node* parent;
        };

        struct dist_ind {
            double dist;
            uint ind;
        };

        struct DI_less {
            bool operator() (const dist_ind& a, const dist_ind& b) {
                return a.dist < b.dist;
            }
        };

        typedef std::priority_queue<dist_ind,std::vector<dist_ind>,DI_less> pqueue;
        
        class kd_tree {
            private:
            uint first_split;
            node* head;
            arma::mat X;
            arma::mat bounding_box;
            node* build_tree(const arma::mat& data, const arma::uvec& inds, int d);
            void find_kNN(const arma::rowvec& pt, node* T, uint current_dim, const arma::mat& bounds, pqueue& kbest, const uint& k);
            double find_min(node* T, uint dim, uint current_dim);
            double find_max(node* T, uint dim, uint current_dim);

            public:
            kd_tree() {};
            kd_tree(const arma::mat& data);
            arma::mat data();
            arma::mat find_kNN(const arma::rowvec& pt, uint k=1);
            arma::uvec index_kNN(const arma::rowvec& pt, uint k=1);
            double min(uint dim);
            double max(uint dim);
            uint size();
            uint dim();
        };
    }
}

// --- misc
inline int mod(int a, int b) {
    return (a%b + b)%b;
}

void meshgrid(arma::mat&, arma::mat&, const arma::vec&, const arma::vec&);
void meshgrid(arma::mat&, const arma::vec&);

arma::uvec sample_from(int, const arma::vec&, const arma::uvec& labels = arma::uvec());
int sample_from(const arma::vec&, const arma::uvec& labels = arma::uvec());

uint index_median(const arma::vec& x);