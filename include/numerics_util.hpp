// --- utitility -------------- //
    // --- integral constants
        const double X4[4] = {-1, -0.447213595499958, 0.447213595499958, 1};
        const double W4[4] = {0.166666666666667, 0.833333333333333, 0.833333333333333, 0.166666666666667};

        const double X7[7] = {-1, -0.468848793470714, -0.830223896278567, 0, 0.830223896278567, 0.468848793470714, 1};
        const double W7[7] = {0.047619047619048, 0.431745381209863, 0.276826047361566, 0.487619047619048, 0.276826047361566, 0.431745381209863, 0.047619047619048};
    // --- optimization constants
        const int newton_max_iter   = 400;
        const int broyd_max_iter    = 400;
        const int bfgs_max_iter     = 400;
        const int gd_max_iter       = 2000;
        const int no_grad_max_iter  = 800;
        const int gen_pop           = 100;
        const int gen_div_lim       = 30;
        
        const double root_err       = 1e-5;
        const double gen_prob       = 0.5;
        const double gen_mut_rate   = 0.1;
    // --- enumerators
        typedef enum INTEGRATOR {
            SIMPSON,
            LOBATTO
        } integrator;

        typedef enum NONLIN_SOLVERS {
            NEWTON,
            BROYD,
            BFGS,
            LBFGS,
            LMLSQR,
            NLCGD,
            MGD,
            SGD,
            ADJGD
        } nonlin_solver;

        typedef enum KERNELS {
            RBF,
            square,
            triangle,
            parabolic
        } kernels;
    // --- input objects
        typedef std::function<arma::vec(const arma::vec&)> vector_func;
        typedef std::function<arma::mat(const arma::vec&)> vec_mat_func;
        typedef std::function<double(const arma::vec&,int)> sp_vector_func;
        typedef std::function<double(double)> dfunc;
        typedef std::function<double(const arma::vec&)> vec_dfunc;
    // --- output objects
        struct data_pair {arma::mat X, Y; arma::umat indices, exclude_indices;};
        typedef std::vector<data_pair> folds;
    // --- option structs
        typedef struct NONLIN_OPTS {
            // inputs
            double err;
            uint max_iter;
            bool use_FD_jacobian, use_cgd;
            double wolfe_c1;
            double wolfe_c2;
            double wolfe_scaling;
            arma::mat* init_jacobian;
            arma::mat* init_jacobian_inv;
            vec_mat_func* jacobian_func;

            // outputs
            uint num_iters_returned;
            uint num_FD_approx_needed;
            arma::mat final_jacobian;
            NONLIN_OPTS() {
                err = root_err;
                max_iter = 100;
                use_FD_jacobian = false;
                use_cgd = false;
                init_jacobian = nullptr;
                init_jacobian_inv = nullptr;
                jacobian_func = nullptr;
                wolfe_c1 = 1e-4;
                wolfe_c2 = 0.9;
                wolfe_scaling = 0.5;
                num_iters_returned = 0;
                num_FD_approx_needed = 0;
            }
        } nonlin_opts;

        typedef struct LBFGS_OPTS {
            // inputs
            double err;
            uint max_iter;
            uint num_iters_to_remember;
            double wolfe_c1;
            double wolfe_c2;
            double wolfe_scaling;
            arma::vec init_hess_diag_inv;

            // outputs
            uint num_iters_returned;
            LBFGS_OPTS() {
                err = root_err;
                max_iter = bfgs_max_iter;
                num_iters_to_remember = 5;
                num_iters_returned = 0;
                wolfe_c1 = 1e-4;
                wolfe_c2 = 0.9;
                wolfe_scaling = 0.5;
            }
        } lbfgs_opts;

        typedef struct LEAST_SQR_OPTS {
            // inputs
            double err;
            uint max_iter;
            double damping_param;
            double damping_scale;
            bool use_scale_invariance, use_cgd;
            vec_mat_func* jacobian_func;

            // outputs
            uint num_iters_returned;
            uint num_FD_approx_made;
            arma::mat final_jacobian;
            LEAST_SQR_OPTS() {
                err = 1e-6;
                max_iter = 100;
                damping_param = 1e-2;
                damping_scale = 2;
                use_scale_invariance = true;
                use_cgd = true;
                jacobian_func = nullptr;
                num_iters_returned = 0;
                num_FD_approx_made = 0;
            }
        } lsqr_opts;

        typedef struct CONJ_GRAD_OPTS {
            uint max_iter;
            arma::mat preconditioner;
            arma::sp_mat sp_precond;
            double err;
            bool is_symmetric;

            uint num_iters_returned;
            CONJ_GRAD_OPTS() {
                max_iter = 0;
                err = 1e-6;
                is_symmetric = true;
                num_iters_returned = 0;
            }
        } cg_opts;

        typedef struct GRADIENT_DESCENT_OPTS {
            // inputs
            double err;
            uint max_iter;
            uint grad_nelem;
            double damping_param;
            uint stochastic_batch_size;

            // outputs
            uint num_iters_returned;
            GRADIENT_DESCENT_OPTS() {
                err = 1e-4;
                max_iter = gd_max_iter;
                grad_nelem = 0;
                damping_param = 0.99;
                num_iters_returned = 0;
                stochastic_batch_size = 10;
            }
        } gd_opts;

        typedef struct FPI_OPTS {
            double err;
            unsigned int max_iter;
            unsigned int steps_to_remember;

            unsigned int num_iters_returned;

            FPI_OPTS() {
                err = root_err;
                max_iter = broyd_max_iter;
                steps_to_remember = 5;
            }
        } fpi_opts;

        typedef struct UNCONSTRAINED_OPTIM_OPTS {
            // inputs
            nonlin_solver solver; // -- general
            double tolerance; // -- general
            uint max_iter; // -- general
            bool use_FD_gradient; // -- general
            bool use_FD_hessian; // -- bfgs, lmlsqr
            bool use_scale_invariance; // -- lmlsqr
            double damping_param; // -- lmlsqr, mgd, sgd
            double damping_scale; // -- lmlsqr, mgd, sgd
            double wolfe_c1; // -- lbfgs, bfgs
            double wolfe_c2; // -- lbfgs, bfgs
            double wolfe_scaling; // -- lbfgs, bfgs
            uint stochastic_batch_size; // -- sgd
            uint num_iters_to_remember; // -- lbfgs
            arma::mat* init_hessian; // -- bfgs, lbfgs, lmlsqr
            arma::mat* init_hessian_inv; // -- bfgs, lbfgs, lmlsqr
            vec_mat_func* hessian_func; // -- newton, lmlsqr
            vector_func* gradient_func; // -- general (excluding sgd)
            sp_vector_func* indexed_gradient_func; // -- sgd

            //outputs
            uint num_iters_returned;

            UNCONSTRAINED_OPTIM_OPTS() {
                solver = LBFGS;
                tolerance = 1e-2;
                max_iter = 100;
                use_FD_hessian = true;
                use_FD_gradient = true;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 1e-4;
                wolfe_c2 = 0.9;
                wolfe_scaling = 0.5;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                hessian_func = nullptr;
                gradient_func = nullptr;
                num_iters_returned = 0;
                stochastic_batch_size = 0;
                num_iters_to_remember = 0;
            }
            
            void use_newton(vector_func* gradient,
            vec_mat_func* hessian) {
                if (hessian == nullptr || gradient == nullptr) { // error
                    std::cerr << "optim_opts::use_newton() error: invalid inputs. Newton requires both a gradient function and a hessian function." << std::endl;
                    return;
                }
                solver = NEWTON;
                tolerance = root_err;
                max_iter = newton_max_iter;
                hessian_func = hessian;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 1e-4;
                wolfe_c2 = 0.9;
                wolfe_scaling = 0.5;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
            }
            
            void use_bfgs(vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_bfgs() error: bfgs requires a gradient input." << std::endl;
                    return;
                }
                solver = BFGS;
                tolerance = root_err;
                max_iter = bfgs_max_iter;
                hessian_func = nullptr;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 1e-4;
                wolfe_c2 = 0.9;
                wolfe_scaling = 0.5;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
                num_iters_to_remember = 0;
            }

            void use_lbfgs(vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_lbfgs() error: bfgs requires a gradient input." << std::endl;
                    return;
                }
                solver = LBFGS;
                tolerance = root_err;
                max_iter = bfgs_max_iter;
                hessian_func = nullptr;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 1e-4;
                wolfe_c2 = 0.9;
                wolfe_scaling = 0.5;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
                num_iters_to_remember = 30;
            }

            void use_lmlsqr(vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_lmlsqr() error: lmlsqr requires a gradient input." << std::endl;
                    return;
                }
                solver = LMLSQR;
                tolerance = root_err;
                max_iter = newton_max_iter;
                hessian_func = nullptr;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = true;
                use_scale_invariance = true;
                damping_param = 1e-2;
                damping_scale = 2;
                wolfe_c1 = 0;
                wolfe_c2 = 0;
                wolfe_scaling = 0;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
                num_iters_to_remember = 0;
            }

            void use_momentum(vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_momentum() error: mgd requires a gradient input." << std::endl;
                    return;
                }
                solver = MGD;
                tolerance = 1e-2;
                max_iter = gd_max_iter;
                hessian_func = nullptr;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0.99;
                damping_scale = 0;
                wolfe_c1 = 0;
                wolfe_c2 = 0;
                wolfe_scaling = 0;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
                num_iters_to_remember = 0;
            }

            void use_sgd(sp_vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_sgd() error: mgd requires a gradient input." << std::endl;
                    return;
                }
                solver = SGD;
                tolerance = 1e-2;
                max_iter = gd_max_iter;
                hessian_func = nullptr;
                gradient_func = nullptr;
                indexed_gradient_func = gradient;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 0;
                wolfe_c2 = 0;
                wolfe_scaling = 0;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 10;
                num_iters_to_remember = 0;
            }

            void use_nlcgd(vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_nlcgd() error: nlcgd requires a gradient input." << std::endl;
                    return;
                }
                solver = NLCGD;
                tolerance = root_err;
                max_iter = bfgs_max_iter;
                hessian_func = nullptr;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 0;
                wolfe_c2 = 0;
                wolfe_scaling = 0;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
                num_iters_to_remember = 0;
            }

            void use_adj_gd(vector_func* gradient) {
                if (gradient == nullptr) {
                    std::cerr << "optim_opts::use_adj_gd() error: nlcgd requires a gradient input." << std::endl;
                    return;
                }
                solver = ADJGD;
                tolerance = root_err;
                max_iter = bfgs_max_iter;
                hessian_func = nullptr;
                gradient_func = gradient;
                indexed_gradient_func = nullptr;
                use_FD_gradient = false;
                use_FD_hessian = false;
                use_scale_invariance = false;
                damping_param = 0;
                damping_scale = 0;
                wolfe_c1 = 0;
                wolfe_c2 = 0;
                wolfe_scaling = 0;
                init_hessian = nullptr;
                init_hessian_inv = nullptr;
                stochastic_batch_size = 0;
                num_iters_to_remember = 0;
            }
        } optim_opts;

        typedef struct GENETIC_OPTS {
            double err;
            uint population_size;
            double reproduction_rate;
            uint diversity_limit;
            double mutation_rate;
            double search_radius;
            GENETIC_OPTS() {
                population_size = gen_pop;
                reproduction_rate = gen_prob;
                diversity_limit = gen_div_lim;
                mutation_rate = gen_mut_rate;
                err = 1e-2;
                search_radius = 1;
            }
        } gen_opts;
// --- misc ------------------- //
    inline double eps(double x = 1.0) {
        double e = x;
        while( x + e != x) {
            e /= 2;
        }
        return e;
    }
    
    inline int mod(int a, int b) {
        return (a%b + b)%b;
    }
    
    arma::vec eval(const vec_dfunc&, arma::mat&);

    arma::mat meshgrid(const arma::vec&);

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

    double wolfe_step(const vec_dfunc&, const vector_func&, const arma::vec&, const arma::vec&, double, double, double);
    double line_min(const dfunc&);

    arma::vec sample_from(int, const arma::vec&, const arma::vec& labels = arma::vec());
    double sample_from(const arma::vec&, const arma::vec& labels = arma::vec());

    void ichol(const arma::mat&, arma::mat&);
    void ichol(const arma::sp_mat&, arma::sp_mat&);