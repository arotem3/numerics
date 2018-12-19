#include <functional>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

#define ARMA_USE_SUPERLU 1
#include <armadillo>

namespace numerics {
    // --- constants -------------- //
        // --- integral constants
        const double X4[4] = {-1, -0.447213595499958, 0.447213595499958, 1};
        const double W4[4] = {0.166666666666667, 0.833333333333333, 0.833333333333333, 0.166666666666667};

        const double X7[7] = {-1, -0.468848793470714, -0.830223896278567, 0, 0.830223896278567, 0.468848793470714, 1};
        const double W7[7] = {0.047619047619048, 0.431745381209863, 0.276826047361566, 0.487619047619048, 0.276826047361566, 0.431745381209863, 0.047619047619048};

        // optimization constants
        const int bfgs_max_iter     = 1000;
        const int sgd_max_iter      = 2000;
        const int no_grad_max_iter  = 400;
        const int newton_max_iter   = 400;
        const int broyd_max_iter    = 400;
        const int gen_pop           = 100;
        const int gen_div_lim       = 30;
        
        const double root_err       = 1e-5;
        const double gen_prob       = 0.5;
        const double gen_mut_rate   = 0.1;

        // enums
        typedef enum INTEGRATOR {
            SIMPSON,
            TRAPEZOID,
            LOBATTO
        } integrator;

        typedef enum NONLIN_SOLVERS {
            NEWTON,
            BROYD,
            BFGS,
            LMLSQR
        } nonlin_solver;

        // option structs
        typedef struct NONLIN_OPTS {
            // inputs
            double err;
            size_t max_iter;
            bool use_FD_jacobian;
            arma::mat* init_jacobian;
            arma::mat* init_jacobian_inv;

            // outputs
            size_t num_iters_returned;
            size_t num_FD_approx_needed;
            arma::mat final_jacobian;
            NONLIN_OPTS() {
                err = root_err;
                max_iter = 100;
                use_FD_jacobian = false;
                init_jacobian = nullptr;
                init_jacobian_inv = nullptr;
                num_iters_returned = 0;
                num_FD_approx_needed = 0;
            }
        } nonlin_opts;

        typedef struct LEAST_SQR_OPTS {
            // inputs
            double err;
            size_t max_iter;
            double damping_param;
            double damping_scale;
            bool use_scale_invariance;
            std::function<arma::mat(const arma::vec&)>* jacobian_func;

            // outputs
            size_t num_iters_returned;
            size_t num_FD_approx_made;
            arma::mat final_jacobian;
            LEAST_SQR_OPTS() {
                err = 1e-6;
                max_iter = 100;
                damping_param = 1e-2;
                damping_scale = 2;
                use_scale_invariance = true;
                jacobian_func = nullptr;
                num_iters_returned = 0;
                num_FD_approx_made = 0;
            }
        } lsqr_opts;

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
        
        arma::vec eval(std::function<double(const arma::vec&)>, arma::mat&);
    // --- integration ------------ //
        double integrate(std::function<double(double)>, double, double, integrator i = SIMPSON, double err = 1e-5);
        double Sintegrate(std::function<double(double)>, double, double, double err = 1e-5);
        double Tintegrate(std::function<double(double)>, double, double, double err = 1e-2);
        double Lintegrate(std::function<double(double)>, double, double, double err = 1e-5);
        
        double mcIntegrate(std::function<double(const arma::vec&)>, const arma::vec&, const arma::vec&, double err = 1e-2, int N = 1e3);
    // --- root finding ----------- //
        void newton(std::function<arma::vec(const arma::vec&)>, std::function<arma::mat(const arma::vec&)>, arma::vec&, nonlin_opts& opts);
        nonlin_opts newton(std::function<arma::vec(const arma::vec&)>, std::function<arma::mat(const arma::vec&)>, arma::vec&);
        
        void broyd(std::function<arma::vec(const arma::vec&)>, arma::vec&, nonlin_opts&);
        nonlin_opts broyd(std::function<arma::vec(const arma::vec&)>, arma::vec&);
        
        void bfgs(std::function<arma::vec(const arma::vec&)>, arma::vec&, nonlin_opts&);
        nonlin_opts bfgs(std::function<arma::vec(const arma::vec&)>, arma::vec&);

        void lmlsqr(std::function<arma::vec(const arma::vec&)>, arma::vec&, lsqr_opts&);
        lsqr_opts lmlsqr(std::function<arma::vec(const arma::vec&)>, arma::vec&);

        double newton(std::function<double(double)>, std::function<double(double)>, double, double err = 1e-10);
        double secant(std::function<double(double)>, double, double err = 1e-8);
        double bisect(std::function<double(double)>, double, double, double tol = 1e-8);
        double roots(std::function<double(double)>, double, double);
    // --- optimization ----------- //
        double sgd(std::function<double(const arma::vec&)>, std::function<arma::vec(const arma::vec&)>, arma::vec&);
        double sgd(std::function<double(const arma::vec&)>, arma::vec&);

        double simplex(arma::mat&, arma::vec&);
        double simplex(const arma::rowvec&, const arma::mat&, const arma::vec&, arma::vec&);

        double genOptim(std::function<double(const arma::vec&)>, arma::vec&, const arma::vec&, const arma::vec&);
        double genOptim(std::function<double(const arma::vec&)>, arma::vec&, double search_radius = 1);
        
        double boolOptim(std::function<double(const arma::uvec&)>, arma::uvec&, int);
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
            arma::mat operator()(size_t);
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
        
        arma::vec nearestInterp(const arma::vec&, const arma::vec&, const arma::vec&);
        arma::vec linearInterp(const arma::vec&, const arma::vec&, const arma::vec&);
        arma::vec lagrangeInterp(const arma::vec&, const arma::vec&, const arma::vec&);
        arma::vec sincInterp(const arma::vec&, const arma::vec&, const arma::vec&);

        arma::mat LPM(const arma::vec&, const arma::mat&, const arma::vec&);
        arma::mat LIM(const arma::vec&, const arma::mat&, const arma::vec&);
    // --- difference methods ----- //
        void approx_jacobian(std::function<arma::vec(const arma::vec&)>, arma::mat&, const arma::vec&, double err = 1e-2, bool catch_zero = true);

        arma::vec grad(std::function<double(const arma::vec&)>, const arma::vec&, double err = 1e-5, bool catch_zero = true);
        
        double deriv(std::function<double(double)>, double, double err = 1e-5, bool catch_zero = true);

        arma::vec specral_deriv(std::function<double(double)>, arma::vec&, int sample_points = 100);
    // --- data analysis ---------- //
        class kmeans {
            private:
            int k;              // number of clusters
            unsigned int dim;   // dimension of problem space
            arma::mat C;        // each column is a cluster mean
            arma::mat* data;    // the data set ~ this is a pointer to the data set, so it will not be deleted when kmeans is deleted!
            arma::rowvec dataCluster;       // the i^th elem of dataCluster is the cluster number for the i^th data input. 
            int closestC(const arma::vec&); // find closest cluster to a data pt

            public:
            kmeans(arma::mat&, int);            // constructor
            kmeans(kmeans& KM) {
                k = KM.k;
                dim = KM.dim;
                C = KM.C;
                arma::SizeMat sz = arma::size(*KM.data);
                data = new arma::mat(sz);
                *data = *KM.data;
                dataCluster = KM.dataCluster;
            }
            kmeans(std::istream&);
            void load(std::istream&);
            void save(std::ostream&);

            arma::rowvec getClusters() const;   // get rowvec dataCluster
            arma::mat getCentroids() const;     // get matrix C

            // all of these return the cluster number of a set of data points.
            arma::rowvec operator()(const arma::mat&);
            arma::rowvec place_in_cluster(const arma::mat&);
            int operator()(const arma::vec&);
            int place_in_cluster(const arma::vec&);
            int operator()(double);
            int place_in_cluster(double);

            // returns a matrix of the given cluster
            arma::mat operator[](int);
            arma::mat all_from_cluster(int);

            // prints an overview to output stream
            void print(std::ostream&);
        };
};