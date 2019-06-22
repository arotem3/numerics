// --- root finding ----------- //
//--- linear ---//
void cgd(arma::mat&, arma::mat&, arma::mat&, double tol = 1e-3, int max_iter = 0);
void cgd(const arma::sp_mat&, const arma::mat&, arma::mat&, double tol = 1e-3, int max_iter = 0);

void linear_adj_gd(arma::mat&, arma::mat&, arma::mat&, double tol = 1e-3, int max_iter = 0);
void linear_adj_gd(const arma::sp_mat&, const arma::mat&, arma::mat&, double tol = 1e-3, int max_iter = 0);

//--- nonlinear ---//
class nlsolver {
    protected:
    uint num_iter;
    int exit_flag;

    public:
    uint max_iterations;
    double tol;
    int num_iterations() {
        return num_iter;
    }
    int get_exit_flag() {
        return exit_flag;
    }
    int get_exit_flag(std::string& flag) {
        if (exit_flag == 0) {
            flag = "converged to root within specified tolerance in ";
            flag += std::to_string(num_iter) + " iterations.";
        } else if (exit_flag == 1) {
            flag = "could not converge to the specified tolerance within ";
            flag += std::to_string(max_iterations) + " iterations.";
        } else if (exit_flag == 2) {
            flag = "NaN or infinite value encountered after ";
            flag +=  std::to_string(num_iter) + " iterations.";
        } else {
            flag = "solver never called.";
        }
        return exit_flag;
    }
};

class newton : public nlsolver {
    public:
    bool use_cgd;

    newton(double tolerance = 1e-3) {
        tol = tolerance;
        use_cgd = true;
        exit_flag = -1;
        num_iter = 0;
        max_iterations = 100;
    }

    void fsolve(std::function<arma::vec(const arma::vec&)> f,
        std::function<arma::mat(const arma::vec&)> jacobian,
        arma::vec& x,
        int max_iter = -1);
};

class broyd : public nlsolver {
    public:
    broyd(double tolerance = 1e-3) {
        tol = tolerance;
        num_iter = 0;
        exit_flag = -1;
        max_iterations = 100;
    }

    void fsolve(std::function<arma::vec(const arma::vec&)> f,
                arma::vec& x,
                int max_iter = -1);
    void fsolve(std::function<arma::vec(const arma::vec&)> f,
                std::function<arma::mat(const arma::vec&)> jacobian,
                arma::vec& x,
                int max_iter = -1);
};

class lmlsqr : public nlsolver {
    public:
    double damping_param, damping_scale;
    bool use_cgd;
    int num_iterations() {
        return num_iter;
    }

    lmlsqr(double tolerance = 1e-3) {
        tol = tolerance;
        num_iter = 0;
        exit_flag = -1;
        max_iterations = 100;
        use_cgd = true;
        damping_param = 1e-2;
        damping_scale = 2;
    }

    void fsolve(std::function<arma::vec(const arma::vec& x)> f,
                arma::vec& x,
                int max_iter = -1);
    void fsolve(std::function<arma::vec(const arma::vec& x)> f,
                std::function<arma::mat(const arma::vec& x)> jacobian,
                arma::vec& x,
                int max_iter = -1);
};

class mix_fpi : public nlsolver {
    public:
    uint steps_to_remember;
    
    mix_fpi(double tolerance = 1e-3, uint num_steps = 5) {
        tol = tolerance;
        steps_to_remember = num_steps;
        max_iterations = 100;
        num_iter = 0;
        exit_flag = -1;
    }

    void find_fixed_point(std::function<arma::vec(const arma::vec&)> f,
                            arma::vec& x,
                            int max_iter = -1);
};

//--- univariate ---//
double fzero(const std::function<double(double)>& f, double a, double b, double tol = 1e-5);
double newton_1d(const std::function<double(double)>& f, const std::function<double(double)>& df, double x, double err = 1e-5);
double secant(const std::function<double(double)>& f, double a, double b, double tol = 1e-5);
double bisect(const std::function<double(double)>& f, double a, double b, double tol = 1e-2);
// --- optimization ----------- //
double wolfe_step(const std::function<double(const arma::vec&)>& f,
                    const std::function<arma::vec(const arma::vec&)>& grad_f,
                    const arma::vec& x,
                    const arma::vec& p,
                    double c1, double c2, double b);

double line_min(const std::function<double(double)>& line_f);

class optimizer {
    protected:
    uint num_iter;
    int exit_flag;

    public:
    uint max_iterations;
    double tol;
    int num_iterations() {
        return num_iter;
    }
    int get_exit_flag() {
        return exit_flag;
    }
    int get_exit_flag(std::string& flag) {
        if (exit_flag == 0) {
            flag = "converged to root within specified tolerance in ";
            flag += std::to_string(num_iter) + " iterations.";
        } else if (exit_flag == 1) {
            flag = "could not converge to the specified tolerance within ";
            flag += std::to_string(max_iterations) + " iterations.";
        } else if (exit_flag == 2) {
            flag = "NaN or infinite value encountered after ";
            flag +=  std::to_string(num_iter) + " iterations.";
        } else {
            flag = "solver never called.";
        }
        return exit_flag;
    }
};

class bfgs : public optimizer {
    public:
    double wolfe_c1, wolfe_c2, wolfe_scale;
    bool use_finite_difference_hessian;

    bfgs(double tolerance = 1e-3) {
        tol = tolerance;
        num_iter = 0;
        max_iterations = 100;
        wolfe_c1 = 1e-4;
        wolfe_c2 = 0.9;
        wolfe_scale = 0.5;
        use_finite_difference_hessian = false;
    }

    void minimize(std::function<double(const arma::vec&)> f,
                    std::function<arma::vec(const arma::vec&)> grad_f,
                    arma::vec& x, int max_iter = -1);
    void minimize(std::function<double(const arma::vec&)> f,
                    std::function<arma::vec(const arma::vec&)> grad_f,
                    std::function<arma::mat(const arma::vec&)> hessian,
                    arma::vec& x, int max_iter = -1);
};

class lbfgs : public optimizer {
    private:
    void lbfgs_update(arma::vec& p, numerics_private_utility::cyc_queue& S, numerics_private_utility::cyc_queue& Y);

    public:
    uint steps_to_remember;
    double wolfe_c1, wolfe_c2, wolfe_scale;

    lbfgs(double tolerance = 1e-3, uint num_steps = 5) {
        tol = tolerance;
        steps_to_remember = num_steps;
        num_iter = 0;
        exit_flag = -1;
        max_iterations = 100;
        wolfe_c1 = 1e-4;
        wolfe_c2 = 0.9;
        wolfe_scale = 0.5;
    }

    void minimize(std::function<double(const arma::vec&)> f,
                    std::function<arma::vec(const arma::vec&)> grad_f,
                    arma::vec& x, int max_iter);
};

class mgd :public optimizer {
    public:
    double damping_param, step_size;

    mgd(double tolerance = 1e-3) {
        tol = tolerance;
        num_iter = 0;
        exit_flag = -1;
        max_iterations = 100;
        damping_param = 0.99;
        step_size = 0;
    }

    void minimize(std::function<arma::vec(const arma::vec&)> grad_f, arma::vec& x, int max_iter = -1);
};

class nlcgd : public optimizer {
    public:
    double step_size;

    nlcgd(double tolerance = 1e-3) {
        tol = tolerance;
        num_iter = 0;
        exit_flag = -1;
        max_iterations = 100;
        step_size = 0;
    }

    void minimize(std::function<arma::vec(const arma::vec&)> grad_f, arma::vec& x, int max_iter = -1);
};

class adj_gd : public optimizer {
    public:
    double step_size;

    adj_gd(double tolerance = 1e-3) {
        tol = tolerance;
        num_iter = 0;
        exit_flag = -1;
        max_iterations = 100;
        step_size = 0;
    }

    void minimize(std::function<arma::vec(const arma::vec&)> grad_f, arma::vec& x, int max_iter = -1);
};
//--- linear contrained ---//
double simplex(arma::mat&, arma::vec&);
double simplex(const arma::rowvec&, const arma::mat&, const arma::vec&, arma::vec&);

//--- box contrained gradient free ---//
class gen_optim : public optimizer {
    private:
    arma::vec fitness(const std::function<double(const arma::vec&)>& f, const arma::mat& x, int n);
    arma::vec diversity(arma::mat x);
    arma::rowvec cross_over(const arma::rowvec& a, const arma::rowvec& b);


    public:
    double reproduction_rate, mutation_rate, search_radius, diversity_weight;
    uint population_size, diversity_cutoff, random_seed, max_iterations;

    gen_optim(double tolerance = 1e-1) {
        tol = tolerance;
        reproduction_rate = 0.5;
        mutation_rate = 0.5;
        diversity_weight = 0.2;
        search_radius = 1;
        population_size = 100;
        diversity_cutoff = 30;
        random_seed = 0;
        max_iterations = 100;
    }

    void maximize(const std::function<double(const arma::vec& x)>& f,
                    arma::vec& x,
                    const arma::vec& lower_bound,
                    const arma::vec& upper_bound);
    void maximize(const std::function<double(const arma::vec&)>& f, arma::vec& x);
};