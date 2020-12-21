#include <numerics.hpp>

void check_restart_maxit(int& r, int& m, int xn) {
    if (r > 0) {
        if (m <= 0) {
            m = std::min(10, xn);
        }
    } else {
        if (m > 0) {
            r = xn / m;
        } else {
            r = xn;
            m = 1;
        }
    }
}

void check_Ax_b(uint Ax, uint b) {
    if (Ax != b) {
        throw std::runtime_error(
            "gmres() error: length of A*x (" + std::to_string(Ax)
            + ") does not match length of b = (" + std::to_string(b) + ")."
        );
    }
}

void check_M(uint M, uint n) {
    if (M != n) {
        throw std::runtime_error(
            "gmres() error: length of M\\(b-Ax) (" + std::to_string(M)
            + ") does not match length of x = (" + std::to_string(n) + ")."
        );
    }
}

void arnoldi(arma::vec& h, arma::vec& q, const numerics::optimization::VecFunc& A, const std::vector<arma::vec>& Q, uint k) {
    q = A(Q.back());
    check_Ax_b(q.n_elem, Q.back().n_elem);
    h = arma::zeros(k+2);
    for (uint i=0; i <= k; ++i) {
        h(i) = arma::dot(q, Q.at(i));
        q -= h(i)*Q.at(i);
    }
    h(k+1) = arma::norm(q);
    q /= h(k+1);
}

void givens_rotations(arma::vec& h, std::vector<double>& cs, std::vector<double>& sn, uint k) {
    for (uint i=0; i < k; ++i) {
        double t = cs.at(i)*h(i) + sn.at(i)*h(i+1);
        h(i+1) = -sn.at(i)*h(i) + cs.at(i)*h(i+1);
        h(i) = t;
    }

    double t = std::sqrt(std::pow(h(k),2) + std::pow(h(k+1),2));
    cs.push_back(h(k) / t);
    sn.push_back(h(k+1) / t);

    h(k) = cs.at(k)*h(k) + sn.at(k)*h(k+1);
    h(k+1) = 0;
}

bool solve_update(arma::vec& x, const std::vector<arma::vec>& Q, const std::vector<arma::vec>& H, const std::vector<double>& beta) {
    uint k = H.size();
    arma::mat Hmat = arma::zeros(k,k);
    arma::vec rhs(k);
    for (uint j=0; j < k; ++j) {
        for (uint i=0; i <= j; ++i) {
            Hmat(i,j) = H.at(j)(i);
        }
        rhs(j) = beta.at(j);
    }
    arma::vec y;
    bool success = arma::solve(y, arma::trimatu(Hmat), rhs);
    if (not success) return false;
    for (uint j=0; j < k; ++j) {
        x += Q.at(j)*y(j);
    }
    return true;
}

bool _gmres(arma::vec& x, const numerics::optimization::VecFunc& A, const arma::vec& b, const numerics::optimization::VecFunc* M, double tol, int restart, int maxit) {
    check_restart_maxit(restart, maxit, b.n_elem);

    if (x.is_empty()) x = arma::zeros(b.n_elem);
    
    bool success = false;
    double bnorm = arma::norm(b);

    for (uint i=0; i < maxit; ++i) {
        arma::vec r = -A(x);
        check_Ax_b(r.n_elem, b.n_elem);
        r += b;

        if (M != nullptr) {
            r = (*M)(r);
            check_M(r.n_elem, x.n_elem);
        }

        double rnorm = arma::norm(r);

        double err = rnorm / bnorm;

        std::vector<double> sn;
        std::vector<double> cs;
        std::vector<double> beta = {rnorm};

        std::vector<arma::vec> Q = {r/rnorm};
        std::vector<arma::vec> H;

        uint k;
        for (k=0; k < restart; ++k) {
            arma::vec h,q;
            arnoldi(h, q, A, Q, k);
            Q.push_back(std::move(q));

            givens_rotations(h, cs, sn, k);
            H.push_back(std::move(h));

            beta.push_back(-sn.at(k)*beta.at(k));
            beta.at(k) = cs.at(k) * beta.at(k);

            err = std::abs(beta.at(k+1)) / bnorm;

            if (err < tol) {
                success = true;
                break;
            }
        }
        success = solve_update(x, Q, H, beta);
        if (not success) break;
    }
    return success;
}

bool numerics::optimization::gmres(arma::vec& x, const arma::mat& A, const arma::vec& b, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    return _gmres(x, Aprod, b, nullptr, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const arma::mat& A, const arma::vec& b, const VecFunc& M, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    return _gmres(x, Aprod, b, &M, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const arma::mat& A, const arma::vec& b, const arma::sp_mat& M, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M, u);
    };
    return _gmres(x, Aprod, b, &Minv, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const arma::mat& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M2, arma::spsolve(M1, u));
    };
    return _gmres(x, Aprod, b, &Minv, tol, restart, maxit);
}

bool numerics::optimization::gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    return _gmres(x, Aprod, b, nullptr, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const VecFunc& M, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    return _gmres(x, Aprod, b, &M, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M, u);
    };
    return _gmres(x, Aprod, b, &Minv, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const arma::sp_mat& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol, int restart, int maxit) {
    if (not A.is_square()) throw std::invalid_argument("gmres() error: coefficient matrix A must be square.");
    VecFunc Aprod = [&](const arma::vec& u) -> arma::vec {
        return A*u;
    };
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M2, arma::spsolve(M1, u));
    };
    return _gmres(x, Aprod, b, &Minv, tol, restart, maxit);
}

bool numerics::optimization::gmres(arma::vec& x, const VecFunc& A, const arma::vec& b, double tol, int restart, int maxit) {
    return _gmres(x, A, b, nullptr, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const VecFunc& A, const arma::vec& b, const VecFunc& M, double tol, int restart, int maxit) {
    return _gmres(x, A, b, &M, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const VecFunc& A, const arma::vec& b, const arma::sp_mat& M, double tol, int restart, int maxit) {
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M, u);
    };
    return _gmres(x, A, b, &Minv, tol, restart, maxit);
}
bool numerics::optimization::gmres(arma::vec& x, const VecFunc& A, const arma::vec& b, const arma::sp_mat& M1, const arma::sp_mat& M2, double tol, int restart, int maxit) {
    VecFunc Minv = [&](const arma::vec& u) -> arma::vec {
        return arma::spsolve(M2, arma::spsolve(M1, u));
    };
    return _gmres(x, A, b, &Minv, tol, restart, maxit);
}