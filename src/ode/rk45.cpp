#include <numerics.hpp>

void numerics::ode::rk45::solve_ivp(const odefunc& f, double t0, double tf, const arma::vec& U0) {
    _check_range(t0, tf);
    double k = (tf - t0) / 100;

    _t.clear(); _U.clear();
    _t.push_back(t0);
    _U.push_back(U0);
    
    unsigned long long j = 0;   // iterator for indexing our vector during the while loop
    
    arma::vec V1, V2, V3, V4, V5, V6, rk4, rk5;
    double R;               // used for comparing the rk4/5 solutions
    double q;               // used to calculate our next step-size
    
    while(true) {
        // (1) --- calculate (_t,_U) at our next step
        V1 = k * f( _t.at(j), _U.at(j) );
        V2 = k * f( _t.at(j) + 0.2*k, _U.at(j) + 0.2*V1 );
        V3 = k * f( _t.at(j) + 0.3*k, _U.at(j) + (3.0/40)*V1 + (9.0/40)*V2 );
        V4 = k * f( _t.at(j) + 0.8*k, _U.at(j) + (44.0/45)*V1 - (56.0/15)*V2 + (32.0/9)*V3 );
        V5 = k * f( _t.at(j) + (8.0/9)*k, _U.at(j) + (19372.0/6561)*V1 - (25360.0/2187)*V2 + (64448.0/6561)*V3 - (212.0/729)*V4 );
        V6 = k * f( _t.at(j) + k, _U.at(j) + (9017.0/3168)*V1 - (355.0/33)*V2 + (46732.0/5247)*V3 + (49.0/176)*V4 - (5103.0/18656)*V5 );
        rk4 = _U.at(j) + (35.0/384)*V1 + (500.0/1113)*V3 + (125.0/192)*V4 - (2187.0/6784)*V5 + (11.0/84)*V6;
        rk5 = _U.at(j) + (5179.0/57600)*V1 + (7571.0/16695)*V3 + (393.0/640)*V4 - (92097.0/339200)*V5 + (187.0/2100)*V6 + (1.0/40)*k*f(_t.at(j) + k, rk4);

        // (2) --- check if our solution is within the error bound
        double kk;
        if (j > 0) kk = event_handle(_t.at(j), _U.at(j), _t.at(j) + k, rk4,k); // new k based on events
        else kk = 2*k; // ensure kk > k

        R = arma::norm(rk4 - rk5,"inf");
        if ( R < _max_err*arma::norm(_U.at(j),"inf") ) {
            if (0 < kk && kk < k) { // event require us to try again with a smaller step size;
                k = kk;
                continue;
            }
            
            _t.push_back(_t.at(j) + k); // add our current solution to the solution vector
            _U.push_back(std::move(rk5));
            j++;
        }
        if (kk == 0) break; // event requires us to stop

        // (3) --- determine our next step-size q = (err/R)^(1/4)
        k *= std::min(10.0, std::max(0.1, 0.9*std::pow(_max_err/R,0.2)));
        
        if (k < _step_min) { // k too small? our method does not converge to a solution
            std::cerr << "rk45 failed: method could not converge b/c current step-size (=" << k << ") < minimum step size (=" << _step_min << ")\n";
            std::cerr << "\tfailed at _t = " << _t.at(j) << "\n";
            break;
        }
        if (_t.at(j) >= tf) { // check if we have reached tf
            break;
        }
        if (_t.at(j) + k > tf) { // if we have reached tf, we change k
            k = tf - _t.at(j);
        }
    }
}