#include "vector_operators.hpp"

// g++ -o vec_op_ex examples/vec_op_ex.cpp

using namespace std;
using namespace vector_operators;

int main() {
    vector<double> u  = {1,  2,  3,  4,  5};
    vector<int>    v  = {0, -1,  2,  4,  1};

    cout << "u = \t[ " << u << "]" << endl
         << "v = \t[ " << v << "]" << endl
         << "u + v = \t[ " << u + v << "]" << endl
         << "u * v = \t[ " << u * v << "]" << endl
         << "u > v = \t[ " << (u > v) << "]" << endl
         << "any(u == v) = " << any(u == v) << endl
         << "all(u <= v) = " << all(u <= v) << endl
         << "u++ = \t[ " << ++u << "]" << endl;

    return 0;
}