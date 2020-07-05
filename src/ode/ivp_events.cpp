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
        auto event = events.at(i);
        double prev_result = event(prev_t, prev_U);
        double result = event(t,V);
        if (arma::sign(result) != arma::sign(prev_result)) { // event has occured
            if (result - prev_result < 0) {
                if (event_dirs.at(i) == event_direction::NEGATIVE || event_dirs.at(i) == event_direction::ALL) { // negative event
                    if (std::abs(result) < 1e-4) { // we stop!
                        stopping_event = i;
                        return 0;
                    } else { // update k
                        return k/10; // take a smaller step
                    }
                } else k = k; // false positive
            } else {
                if (event_dirs.at(i) == event_direction::POSITIVE || event_dirs.at(i) == event_direction::ALL) { // positive event
                    if (std::abs(result) < 1e-4) { // we stop!
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