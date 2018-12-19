#include "../numerics.hpp"

// g++ -Wall -g -o eps-ex examples/eps-ex.cpp eps.cpp

using namespace numerics;

int main() {
    std::cout << "Here we will show off the eps() function which calculates the distance between an input x and the next closest floating point value." << std::endl
              << "\tWhen we call eps() we get:                      " << eps() << std::endl
              << "\tWhen we call eps(1.0) we get the same thing:    " << eps(1.0) << std::endl
              << "\tWhen we call eps(1e-10) we get:                 " << eps(1e-10) << std::endl
              << "\tWhen we call eps(1e10) we get:                  " << eps(1e10) << std::endl;
    return 0;
}