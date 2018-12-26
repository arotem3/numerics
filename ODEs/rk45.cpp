#include "ODE.hpp"

//--- multivariate RKF45 method for any explicit first order system of ODEs ---//
//--- our equations are of the form u' = f(t,u) -------------------------------//
//--- this implementation uses the Dormand-Prince O(k^4) method ---------------//
//----- f  : f(t,u) [t must be the first variable, u the second] --------------//
//----- t  : vector to store t-values initialized at {t0, tf} -----------------//
//----- U  : vector to store the solution first row is u(t0) ------------------//
//----- err: upper limit error bound, by default set to machine epsilon -------//
void ODE::rk45(const odefun& f, arma::vec& t, arma::mat& U, ivp_options& opts) {
    double err = std::abs(opts.adaptive_max_err);   // if err is set to zero (or lower) set it to eps(U0)
    double kmin = opts.adaptive_step_min;           // the minimum step-size allowed
    double kmax = opts.adaptive_step_max;           // the maximum step-size allowed
    double k = kmax;                                // initialize our step-size at maximum

    double t_temp = t(0);
    double tf = t(1);
    arma::rowvec U_temp = U.row(0);
    t = arma::zeros(20);                // initialize memory
    t(0) = t_temp;
    U = arma::zeros(20, U_temp.n_cols); // initialize memory
    U.row(0) = U_temp;
    
    bool notDone = true;    // check if the algorithm has reached the end
    unsigned short j = 0;   // iterator for indexing our vector during the while loop
    
    arma::rowvec V1;        // V1 - V6: RKF sub-equations used to construct our 4th order and 5th order solutions
    arma::rowvec V2;
    arma::rowvec V3;
    arma::rowvec V4;
    arma::rowvec V5;
    arma::rowvec V6;
    arma::rowvec rk4;       // rk4 - rk5: 4th/5th order solutions
    arma::rowvec rk5;
    double R;               // used for comparing the rk4/5 solutions
    double q;               // used to calculate our next step-size
    
    while(notDone) {
        // (1) --- calculate (t,U) at our next step
            V1 = k * f( t(j), U.row(j) );
            V2 = k * f( t(j) + 0.2*k, U.row(j) + 0.2*V1 );
            V3 = k * f( t(j) + 0.3*k, U.row(j) + (3.0/40)*V1 + (9.0/40)*V2 );
            V4 = k * f( t(j) + 0.8*k, U.row(j) + (44.0/45)*V1 - (56.0/15)*V2 + (32.0/9)*V3 );
            V5 = k * f( t(j) + (8.0/9)*k, U.row(j) + (19372.0/6561)*V1 - (25360.0/2187)*V2 + (64448.0/6561)*V3 - (212.0/729)*V4 );
            V6 = k * f( t(j) + k, U.row(j) + (9017.0/3168)*V1 - (355.0/33)*V2 + (46732.0/5247)*V3 + (49.0/176)*V4 - (5103.0/18656)*V5 );
            rk4 = U.row(j) + (35.0/384)*V1 + (500.0/1113)*V3 + (125.0/192)*V4 - (2187.0/6784)*V5 + (11.0/84)*V6;
            rk5 = U.row(j) + (5179.0/57600)*V1 + (7571.0/16695)*V3 + (393.0/640)*V4 - (92097.0/339200)*V5 + (187.0/2100)*V6 + (1.0/40)*k*f(t(j) + k, rk4);

        // (2) --- check if our solution is within the error bound
            double kk = 2*k; // meaningless initialization greater that k for event handling
            if (j > 0) kk = event_handle(opts,t(j), U.row(j), t(j) + k, rk4,k); // new k based on events
            
            R = arma::norm(rk4 - rk5)/k;
            if ( R < err ) {
                if (0 < kk && kk < k) {     // event require us to try again with a smaller step size;
                    k = kk;
                    continue;
                }

                t_temp = t(j) + k;
                U_temp = rk4;               // we keep rk4 solution because rk45 is proven to minimize error only for the 4th order solution, not the 5th
                t(j+1) = t_temp;            // add our current solution to the solution vector
                U.row(j+1) = U_temp;

                if (kk == 0) break;         // event requires us to stop
                j++;                        // move to the next step
                if (j+1 == t.n_rows) {
                    t = arma::join_cols(t, arma::zeros(arma::size(t)) ); // double storage
                    U = arma::join_cols(U, arma::zeros(arma::size(U)) );
                }
            }

        // (3) --- determine our next step-size q = (err/R)^(1/4)
            q = std::pow(err/R, 0.25);
            if (q < rk45_qmin) {                  // we want to control how quickly q changes so we limit q by the arbitrary values [0.1, 4]
                k *= rk45_qmin;
            }
            else if (q > rk45_qmax) {
                k *= rk45_qmax;
            }
            else {
                k *= q;
            }
            k = (k > kmax) ? (kmax) : (k);  // check if our step-size is too big and fix it
            if (t(j) >= tf) {               // check if we have reached tf
                notDone = false;
            }
            else if (t(j) + k > tf) {       // if we have reached tf, we change k
                k = tf - t(j);
            }
            else if (k < kmin) {            // k too small? our method does not converge to a solution
                notDone = false;
                std::cerr << "rk45() failed: method does not converge b/c minimum k exceeded." << std::endl;
                std::cerr << "\tfailed at t = " << t(j) << std::endl;
                U = U.row(0);               // reseting U and t as a strong reminder that method failed.
                double t0 = t(0);
                t = {t0,tf};
            }
        // (4) --- loop again
    }
    t = t( arma::span(0,j+1) );
    U = U.rows( arma::span(0,j+1) );
}

ODE::ivp_options ODE::rk45(const odefun& f, arma::vec& t, arma::mat& U) {
    ivp_options opts;
    opts.adaptive_max_err = 1e-6;
    opts.adaptive_step_min = rk45_kmin;
    opts.adaptive_step_max = rk45_kmax;
    rk45(f,t,U,opts);
    return opts;
}

arma::vec ODE::rk45(std::function<double(double,double)> f, arma::vec& t, double U0, ivp_options& opts) {
    double err = std::abs(opts.adaptive_max_err);        // if err set to 0 (or lower) set to eps(U0)
    double kmin = opts.adaptive_step_min;                // the minimum step-size allowed
    double kmax = opts.adaptive_step_max;                // the maximum step-size allowed
    double k = kmax;                                     // initialize our step-size at maximum

    double t_temp = t(0);
    double tf = t(1);
    double U_temp = U0;
    t = {t(0)};
    arma::vec U = {U(0)};

    bool notDone = true;    // check if the algorithm has reached the end
    short j = 0;            // iterator for indexing our vector during the while loop

    double V1, V2, V3, V4, V5, V6, rk4, rk5;
    double R;               // used for comparing the rk4/5 solutions
    double q;               // used to calculate our next step-size

    while (notDone) {
        // (1) --- calculate (t,U) at our next step
        V1 = k * f( t(j), U(j) );
        V2 = k * f( t(j) + 0.2*k, U(j) + 0.2*V1 );
        V3 = k * f( t(j) + 0.3*k, U(j) + (3.0/40)*V1 + (9.0/40)*V2 );
        V4 = k * f( t(j) + 0.8*k, U(j) + (44.0/45)*V1 - (56.0/15)*V2 + (32.0/9)*V3 );
        V5 = k * f( t(j) + (8.0/9)*k, U(j) + (19372.0/6561)*V1 - (25360.0/2187)*V2 + (64448.0/6561)*V3 - (212.0/729)*V4 );
        V6 = k * f( t(j) + k, U(j) + (9017.0/3168)*V1 - (355.0/33)*V2 + (46732.0/5247)*V3 + (49.0/176)*V4 - (5103.0/18656)*V5 );
        rk4 = U(j) + (35.0/384)*V1 + (500.0/1113)*V3 + (125.0/192)*V4 - (2187.0/6784)*V5 + (11.0/84)*V6;
        rk5 = U(j) + (5179.0/57600)*V1 + (7571.0/16695)*V3 + (393.0/640)*V4 - (92097.0/339200)*V5 + (187.0/2100)*V6 + (1.0/40)*k*f(t(j) + k, rk4);

        // (2) --- check if our solution is within the error bound
        R = std::abs(rk4 - rk5)/k;
        if ( R < err ) {
            t_temp = t(j) + k;
            U_temp = rk4;               // we keep rk4 solution because rkf45 is proven to minimize error only for the 4th order solution, not the 5th
            t.insert_rows(j+1,t_temp);  // add our current solution to the solution vector
            U.insert_rows(j+1,U_temp);
            
            j++;                        // move to the next step
        }

        // (3) --- determine our next step-size q = (err/R)^(1/4)
        q = std::pow(err/R, 0.25);
        if (q < rk45_qmin) {                  // we want to control how quickly q changes so we limit q by the arbitrary values [0.1, 4]
            k *= rk45_qmin;
        }
        else if (q > rk45_qmax) {
            k *= rk45_qmax;
        }
        else {
            k *= q;
        }
        k = (k > kmax) ? (kmax) : (k);  // check if our step-size is too big and fix it
        if (t(j) >= tf) {               // check if we have reached tf
            notDone = false;
        }
        else if (t(j) + k > tf) {       // if we have reached tf, we change k
            k = tf - t(j);
        }
        else if (k < kmin) {            // k too small? our method does not converge to a solution
            notDone = false;
            std::cerr << "rk45() failed: method does not converge b/c minimum k exceeded." << std::endl;
            std::cerr << "\tfailed at t = " << t(j) << std::endl;
            U = {U(0)};                 // reseting U and t as a strong reminder that method failed.
            double t0 = t(0);
            t = {t0,tf};
        }
        // (4) --- loop again
    }
    return U;
}

arma::vec ODE::rk45(std::function<double(double,double)> f, arma::vec& t, double U0) {
    ivp_options opts;
    opts.adaptive_max_err = 1e-6;
    opts.adaptive_step_min = rk45_kmin;
    opts.adaptive_step_max = rk45_kmax;
    return rk45(f,t,U0,opts);
}