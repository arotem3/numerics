#include <numerics.hpp>

/* event_handle(prev_t, prev_U, t, V, k) : event handling for stopping an initial value problem when any event occurs.
* --- prev_t : previous t value.
* --- prev_U : previous solution value.
* --- t : current t value.
* --- V : current solution value.
* --- k : current step size. */
double numerics::ode::ivp::event_handle(double prev_t, const arma::rowvec& prev_U, double t, const arma::rowvec& V, double k) {
    if (  events.empty()  ) return k;
    int num_events = events.size();
    
    for (int i=0; i < num_events; ++i) {
        event_func event = events.at(i);
        event_out prev_result = event(prev_t, prev_U);
        event_out result = event(t,V);
        if (arma::sign(result.val) != arma::sign(prev_result.val)) { // event has occured
            if (result.val - prev_result.val < 0) {
                if (result.dir == NEGATIVE || result.dir == ALL) { // negative event
                    if (std::abs(result.val) < 1e-4) { // we stop!
                        stopping_event = i;
                        return 0;
                    } else { // update k
                        return k/10; // take a smaller step
                    }
                } else k = k; // false positive
            } else {
                if (result.dir == POSITIVE || result.dir == ALL) { // positive event
                    if (std::abs(result.val) < 1e-4) { // we stop!
                        stopping_event = i;
                        return 0;
                    } else { // update k
                        return k/10;
                    }
                } else k = k; // false positive
            }
        } else k = k;
    }
    return k;
}