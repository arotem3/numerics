#include <numerics.hpp>

/* ode_solve(f, t, U) : Dormand Prince adaptive runge kutta O(K^4) method for any explicit first order system of ODEs.
 * our equations are of the form u' = f(t,u) [u must be a row vector].
 * --- f  : f(t,u) [t must be the first variable, u the second].
 * --- t  : vector to store t-values initialized at {t_initial, t_final}.
 * --- U  : vector to store the solution first row must be u(t0). */
void numerics::ode::rk45::ode_solve(const std::function<arma::rowvec(double,const arma::rowvec&)>& f, arma::vec& t, arma::mat& U) {
    double err = std::abs(adaptive_max_err);   // if err is set to zero (or lower) set it to eps(U0)
    double kmin = adaptive_step_min;           // the minimum step-size allowed
    double kmax = adaptive_step_max;           // the maximum step-size allowed
    double k = kmax;                                // initialize our step-size at maximum

    double t_temp = t(0);
    double tf = t(1);
    arma::rowvec U_temp = arma::vectorise(U).t();
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
            if (j > 0) kk = event_handle(t(j), U.row(j), t(j) + k, rk4,k); // new k based on events
            
            R = arma::norm(rk4 - rk5)/k;
            if ( R < err ) {
                if (0 < kk && kk < k) {     // event require us to try again with a smaller step size;
                    k = kk;
                    continue;
                }

                t(j+1) = t(j) + k;            // add our current solution to the solution vector
                U.row(j+1) = rk5;

                if (kk == 0) break;         // event requires us to stop
                j++;                        // move to the next step
                if (j+1 == t.n_rows) {
                    t.resize(t.n_rows*2, 1);
                    U.resize(t.n_rows*2, U.n_cols);
                }
            }

        // (3) --- determine our next step-size q = (err/R)^(1/4)
            q = std::pow(err/R, 0.2);
            if (q < 0.1) k *= 0.1;
            else if (q > 10) k *= 10;
            else k *= q;
            
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
            }
        // (4) --- loop again
    }
    t = t( arma::span(0,j) );
    U = U.rows( arma::span(0,j) );
}