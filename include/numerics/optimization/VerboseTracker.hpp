#ifndef NUMERICS_OPTIMIZATION_VERBOSETRACKER_HPP
#define NUMERICS_OPTIMIZATION_VERBOSETRACKER_HPP

#include <iostream>
#include <iomanip>

namespace numerics
{
namespace optimization
{

class VerboseTracker {
    protected:
    u_long max_iter;

    public:
    VerboseTracker(u_long m) {
        max_iter = m;
    }

    void header(const std::string& name) {
        std::cout << "|" << std::right << std::setw(6) << std::setfill(' ') << "iter"
                << "|" << std::right << std::setw(20) << std::setfill(' ') << "progress"
                << "|" << std::right << std::setw(12) << std::setfill(' ') << name
                << "|\n";
    }

    void iter(u_long iter, double fval) {
        std::string bar;
        float p = (float)iter/max_iter;
        for (int i=0; i < 20*p-1; ++i)
            bar += "=";
        bar += ">";
        std::cout << "|" << std::right << std::setw(6) << std::setfill(' ') << iter
                << "|" << std::left << std::setw(20) << std::setfill(' ') << bar
                << "|" << std::scientific << std::setprecision(4) << std::right << std::setw(12) << std::setfill(' ') << fval
                << "|\r" << std::flush;
    }

    void success_flag() {
        std::cout << std::endl << "---converged to solution within tolerance---\n";
    }

    void max_iter_flag() {
        std::cout << std::endl << "---maximum number of iterations reached---\n";
    }

    void failed_step_flag() {
        std::cout << std::endl << "---failed to compute step direction---\n";
    }

    void min_step_flag() {
        std::cout << std::endl << "---could not improve function in the current search direction---\n";
    }

    void empty_flag() {
        std::cout << std::endl;
    }
};

}
}

#endif
