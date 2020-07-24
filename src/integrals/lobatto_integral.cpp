#include <numerics.hpp>

// double numerics::lobatto_integral(const std::function<double(double)>& f, double a, double b, double tol) {
//     tol = std::abs(tol);
//     double h = (b-a)/2;
//     double c = (b+a)/2;

//     double sum4(0), sum7(0);
//     for (int i(0); i < 7; i++) {
//         if (i < 4) {
//             sum4 += constants::lobatto_4pt_weights[i] * f(h * constants::lobatto_4pt_nodes[i] + c);
//         }
//         sum7 += constants::lobatto_7pt_weights[i] * f(h * constants::lobatto_7pt_nodes[i] + c);
//     }
//     sum4 *= h;
//     sum7 *= h;

//     if (std::abs(sum4 - sum7) < tol) return sum4;
//     else return lobatto_integral(f,a,c,tol/2) + lobatto_integral(f,c,b,tol/2);
// }

double lobatto4(const std::function<double(double)>& f, double a, double b) {
    double h = (b - a) / 2;
    double c = (b + a) / 2;
    double sum4 = 0;
    for (short i=0; i < 4; ++i) {
        sum4 += numerics::constants::lobatto_4pt_weights[i] * f(h * numerics::constants::lobatto_4pt_nodes[i] + c);
    }
    return sum4*h;       
}

double lobatto7(const std::function<double(double)>& f, double a, double b) {
    double h = (b - a) / 2;
    double c = (b + a) / 2;
    double sum7 = 0;
    for (short i=0; i < 7; ++i) {
        sum7 += numerics::constants::lobatto_7pt_weights[i] * f(h * numerics::constants::lobatto_7pt_nodes[i] + c);
    }
    return sum7*h;
}

double numerics::lobatto_integral(const std::function<double(double)>& f, double a, double b, double tol) {
    if (tol <= 0) throw std::invalid_argument("require tol (=" + std::to_string(tol) + ") > 0");
    if (b <= a) throw std::invalid_argument("(" + std::to_string(a) + ", " + std::to_string(b) + ") does not define an interval");

    double integral = 0;

    std::queue<double> aq; aq.push(a);
    std::queue<double> bq; bq.push(b);
    std::queue<double> tq; tq.push(tol);

    while (not aq.empty()) {
        a = aq.front(); aq.pop();
        b = bq.front(); bq.pop();
        tol = tq.front(); tq.pop();

        double sum4 = lobatto4(f, a, b);
        double sum7 = lobatto7(f, a, b);
        if (std::abs(sum4 - sum7) < tol) integral += sum7;
        else {
            double mid = (a+b) / 2;
            aq.push(a); bq.push(mid); tq.push(tol/2);
            aq.push(mid); bq.push(b); tq.push(tol/2);
        }
    }
    return integral;
}