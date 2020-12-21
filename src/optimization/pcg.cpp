#include "numerics.hpp"

void _not_sym_error() {
    throw std::invalid_argument("pcg() error: coefficient matrix A is not symmetric.");
}

void _not_square_error() {
    throw std::invalid_argument("pcg() error: coefficient matrix A is not square.");
}

void _check_A(const arma::mat& A) {
    if (not A.is_symmetric()) _not_sym_error();
    else if (not A.is_square()) _not_square_error();
}

void _check_A(const arma::sp_mat& A) {
    if (not A.is_symmetric()) _not_sym_error();
    else if (not A.is_square()) _not_square_error();
}

// check A and b is a valid system
void _check_A_b(uint An, uint bn) {
    if (An != bn) {
        throw std::runtime_error(
            "pcg() error: length of A*x (" + std::to_string(An)
            + ") does not match length of b (" + std::to_string(bn) + ")."
        );
    }
}

// set max_iter if neccessary
void _set_max_iter(int& max_iter, uint n) {
    if (max_iter <= 0) max_iter = n;
}

// require max_iter > 0
void _check_max_iter(int max_iter) {
    if (max_iter <= 0) throw std::invalid_argument("pcg() error: max_iter (" + std::to_string(max_iter) + ") must be > 0");
}

// check that M is the correct size
void _check_M(uint M, uint n) {
    if (M != n) {
        throw std::runtime_error(
            "gmres() error: length of M\\(b-Ax) (" + std::to_string(M)
            + ") does not match length of x = (" + std::to_string(n) + ")."
        );
    }
}

// require tol > 0
void _check_tol(double tol) {
    if (tol <= 0) throw std::invalid_argument("pcg() error: require tol (" + std::to_string(tol) + ") > 0");
}

bool _pcg(arma::vec& x, const numerics::optimization::VecFunc& A, const arma::vec& b, const numerics::optimization::VecFunc* M, double tol, int max_iter) {
    _check_tol(tol);
    _set_max_iter(max_iter, b.n_elem);
    if (x.n_elem != b.n_elem) x = arma::zeros(b.n_elem);
    
    arma::vec r = -A(x);
    _check_A_b(r.n_elem, b.n_elem);
    r += b;

    double bnorm = arma::norm(b);

    bool success = false;

    arma::vec p;
    arma::vec *z;
    double rho_prev, rho;
    for (uint i=0; i < max_iter; ++i) {
        if (M != nullptr) {
            if (z == nullptr) z = new arma::vec();
            *z = (*M)(r);
            _check_M(z->n_elem, r.n_elem);
        } else z = &r;
        
        rho = arma::dot(r,*z);
        if (i == 0) p = *z;
        else {
            double beta = rho / rho_prev;
            p = (*z) + p*beta;
        }
        arma::vec q = A(p);
        _check_A_b(q.n_elem, p.n_elem);
        double alpha = rho / arma::dot(p,q);
        x += alpha * p;
        r -= alpha * q;

        rho_prev = rho;

        if (arma::norm(r)/bnorm < tol) {
            success = true;
            break;
        }
    }
    if (M != nullptr) delete z;

    return success;
}

bool numerics::optimization::pcg(arma::vec& x, const arma::mat& A, const arma::vec& b, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    return _pcg(x, Aprod, b, nullptr, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const arma::mat& A, const arma::vec& b, const VecFunc& M, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    return _pcg(x, Aprod, b, &M, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const arma::mat& A, const arma::vec& b, const arma::sp_mat& M, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M, u);
    };

    return _pcg(x, Aprod, b, &Minv, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const arma::mat& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M2, arma::spsolve(M1, u));
    };

    return _pcg(x, Aprod, b, &Minv, tol, max_iter);
}

bool numerics::optimization::pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    return _pcg(x, Aprod, b, nullptr, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const VecFunc& M, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    return _pcg(x, Aprod, b, &M, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M, u);
    };

    return _pcg(x, Aprod, b, &Minv, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol, int max_iter) {
    _check_A(A);
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };

    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M2, arma::spsolve(M1, u));
    };

    return _pcg(x, Aprod, b, &Minv, tol, max_iter);
}

bool numerics::optimization::pcg(arma::vec& x, const VecFunc& A, const arma::vec& b, double tol, int max_iter) {
    return _pcg(x, A, b, nullptr, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const VecFunc& A, const arma::vec& b, const VecFunc& M, double tol, int max_iter) {
    return _pcg(x, A, b, &M, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const VecFunc& A, const arma::vec& b, const arma::sp_mat& M, double tol, int max_iter) {
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M, u);
    };

    return _pcg(x, A, b, &Minv, tol, max_iter);
}
bool numerics::optimization::pcg(arma::vec& x, const VecFunc& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol, int max_iter) {
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M2, arma::spsolve(M1, u));
    };

    return _pcg(x, A, b, &Minv, tol, max_iter);
}