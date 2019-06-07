#include "numerics.hpp"

/* GENOPTIM : box constrained maximization using a genetic algorithm.
 * --- f  : double = f(x) function to maximize.
 * --- x  : vec where the maximum is.
 * --- xMin,xMax : bounds for box constraints.
 * --- opts : options struct, pass solver parameters. */
double numerics::genOptim(const vec_dfunc& f, arma::vec& x, const arma::vec& xMin, const arma::vec& xMax, gen_opts& opts) {
    arma::arma_rng::set_seed_random();
    int dim = xMin.n_elem; // dimenstion of domain space
    int n = opts.population_size; // number of members in the population

    auto T = [&](arma::vec& x){x = (xMax-xMin)%x + xMin;}; // [0,1] -> [xMin, xMax]
    auto fitness = [&f,n](arma::mat& x){ 
        arma::rowvec F(n,arma::fill::zeros);
        for (int q(0); q < n; ++q) {
            arma::vec v = x.col(q);
            F(q) = f(v);
        }
        F = arma::normalise(F);
        return F;
    }; // fitness of population
    auto divers = [&](arma::mat& x){
        arma::mat D = x;
        arma::vec d = arma::mean(x,1);
        D.each_col() -= d;
        D.each_col([](arma::vec& v){v(0) = arma::norm(v);});
        arma::rowvec D1 = D.row(0);
        D1 = arma::normalise(D1);
        return D1;
    }; // diversity of population
    auto crossOvr = [dim](arma::Col<double>& A1, arma::Col<double>& A2){
        arma::vec nA(dim, arma::fill::zeros);
        for (int i(0); i < dim; ++i) {
            if (std::abs(A1(i) - A2(i)) < 0.1) {
                nA(i) = A1(i) + A2(i);
                nA(i) /= 2;
            } else if (arma::randu() > 0.5) {
                nA(i) = A1(i);
            } else {
                nA(i) = A2(i);
            }
        }
        return nA;
    };

    arma::mat A(dim, n, arma::fill::randu); // initialized population
    A.each_col(T);

    double pc = opts.reproduction_rate; // probability of reproduction for best member of population (value is arbitrary)
    arma::rowvec P(n, arma::fill::zeros); // probability of each member mating
    arma::mat nextA(dim,n,arma::fill::zeros); // next generation
    int numGens = 1;

    auto nextGen = [&]() {
        arma::rowvec B = 2*fitness(A);
        if (numGens < opts.diversity_limit) {
            B += divers(A); // fitness + diversity = measure of good a member of a population is
        }
        arma::umat ind = arma::sort_index(-B);
        for (int i(0); i < n; ++i) {
            if (i == n-1) {
                P(i) = std::pow(1-pc, n-1);
            } else {
                P(i) = std::pow(1-pc, i) * pc;
            }
        }
        
        arma::mat mates(2, n, arma::fill::randu); // pairs of members chosen for reproduction
        for (int i(0); i < 2; ++i) {
            for (int j(0); j < n; ++j) {
                for (int k(0); k < n-1; ++k) {
                    if (mates(i,j) < P(0)) {
                        mates(i,j) = ind(0);
                        break;
                    } else if ( arma::accu(P(arma::span(0,k))) < mates(i,j) && mates(i,j) <= arma::accu(P(arma::span(0,k+1))) ) {
                        mates(i,j) = ind(k);
                        break;
                    }
                }
            }
        }

        arma::vec stnd = 2*arma::stddev(nextA,0,1); // standard deviation of each row of A
        for (int i(0); i < n; ++i) {
            arma::vec A1 = A.col(mates(0,i));
            arma::vec A2 = A.col(mates(1,i));
            nextA.col(i) = crossOvr(A1,A2);
            nextA.col(i) += stnd%arma::randu(dim) - stnd/2;
        }
        for (int i(0); i < dim; ++i) { // box constraints
            for (int j(0); j < n; ++j) {
                if (nextA(i,j) < xMin(i)) {
                    nextA(i,j) = xMin(i);
                } else if (nextA(i,j) > xMax(i)) {
                    nextA(i,j) = xMax(i);
                }
            }
        }
        numGens++;
    }; // produce next generation

    nextGen(); // initialized nextA
    while (arma::norm(A - nextA) > opts.err) {
        A = nextA;
        nextGen();
    }
    x = nextA.col( eval(f,nextA).index_max() );
    return f(x);
}

/* GENOPTIM : box constrained maximization using a genetic algorithm.
 * --- f  : double = f(x) function to maximize.
 * --- x  : vec where the maximum is.
 * --- xMin,xMax : bounds for box constraints. */
double numerics::genOptim(const vec_dfunc& f, arma::vec& x, const arma::vec& xMin, const arma::vec& xMax) {
    gen_opts opts;
    return genOptim(f,x,xMin,xMax,opts);
}

/* GENOPTIM : unconstrained maximization using a genetic algorithm.
 * --- f  : double = f(x) function to maximize.
 * --- x0 : initial guess for local minimum.
 * --- opts : pass solver parameters. */
double numerics::genOptim(const vec_dfunc& f, arma::vec& x0, gen_opts& opts) {
    arma::arma_rng::set_seed_random();
    int dim = x0.n_elem; // dimenstion of domain space
    int n = opts.population_size; // number of members in the population

    auto fitness = [&f,n](arma::mat& x){ 
        arma::rowvec F(n,arma::fill::zeros);
        for (int q(0); q < n; ++q) {
            arma::vec v = x.col(q);
            F(q) = f(v);
        }
        F = arma::normalise(F);
        return F;
    }; // fitness of population
    auto divers = [&](arma::mat& x){
        arma::mat D = x;
        arma::vec d = arma::mean(x,1);
        D.each_col() -= d;
        D.each_col([](arma::vec& v){v(0) = arma::norm(v);});
        arma::rowvec D1 = D.row(0);
        D1 = arma::normalise(D1);
        return D1;
    }; // diversity of population
    auto crossOvr = [dim](arma::Col<double>& A1, arma::Col<double>& A2){
        arma::vec nA(dim, arma::fill::zeros);
        for (int i(0); i < dim; ++i) {
            if (std::abs(A1(i) - A2(i)) < 0.1) {
                nA(i) = A1(i) + A2(i);
                nA(i) /= 2;
            } else if (arma::randu() > 0.5) {
                nA(i) = A1(i);
            } else {
                nA(i) = A2(i);
            }
        }
        return nA;
    };

    arma::mat A(dim, n, arma::fill::randn); // initialized population
    arma::mat tempA = arma::randn(dim,n);
    A += opts.search_radius * tempA;

    double pc = opts.reproduction_rate; // probability of reproduction for best member of population (value is arbitrary)
    arma::rowvec P(n, arma::fill::zeros); // probability of each member mating
    arma::mat nextA(dim,n,arma::fill::zeros); // next generation
    int numGens = 1;

    auto nextGen = [&]() {
        arma::rowvec B = fitness(A);
        if (numGens < gen_div_lim) {
            B += divers(A); // fitness + diversity = measure of good a member of a population is
        }
        arma::umat ind = arma::sort_index(-B);
        for (int i(0); i < n; ++i) {
            if (i == n-1) {
                P(i) = std::pow(1-pc, n-1);
            } else {
                P(i) = std::pow(1-pc, i) * pc;
            }
        }
        
        arma::mat mates(2, n, arma::fill::randu); // pairs of members chosen for reproduction
        for (int i(0); i < 2; ++i) {
            for (int j(0); j < n; ++j) {
                for (int k(0); k < n-1; ++k) {
                    if (mates(i,j) < P(0)) {
                        mates(i,j) = ind(0);
                        break;
                    } else if ( arma::accu(P(arma::span(0,k))) < mates(i,j) && mates(i,j) <= arma::accu(P(arma::span(0,k+1))) ) {
                        mates(i,j) = ind(k);
                        break;
                    }
                }
            }
        }

        arma::vec stnd = 2*arma::stddev(nextA,0,1); // standard deviation of each row of A
        for (int i(0); i < n; ++i) {
            arma::vec A1 = A.col(mates(0,i));
            arma::vec A2 = A.col(mates(1,i));
            nextA.col(i) = crossOvr(A1,A2);
            nextA.col(i) += stnd%arma::randu(dim) - stnd/2;
        }
        numGens++;
    }; // produce next generation

    nextGen(); // initialized nextA
    while (arma::norm(A - nextA) > opts.err) {
        A = nextA;
        nextGen();
    }
    x0 = nextA.col( eval(f,nextA).index_max() );
    return f(x0);
}

/* GENOPTIM : unconstrained maximization using a genetic algorithm.
 * --- f  : double = f(x) function to maximize.
 * --- x0 : initial guess for local minimum. */
double numerics::genOptim(const vec_dfunc& f, arma::vec& x0) {
    gen_opts opts;
    return genOptim(f,x0,opts);
}

/* BOOLOPTIM : boolean maximization using genetic algorithm.
 * --- f  : double = f(x) function to maximize.
 * --- x  : initial guess and solution.
 * --- solution_dimension : dimension of solution. */
double numerics::boolOptim(std::function<double(const arma::uvec&)> f, arma::uvec& x, uint solution_dimension) {
    arma::arma_rng::set_seed_random();
    int n = gen_pop;

    auto fitness = [&f,n](arma::umat& x){ 
        arma::rowvec F(n,arma::fill::zeros);
        for (int q(0); q < n; ++q) {
            arma::uvec v = x.col(q);
            F(q) = f(v);
        }
        F = arma::normalise(F);
        return F;
    }; // fitness of population
    auto crossOvr = [solution_dimension](arma::uvec& A1, arma::uvec& A2){
        arma::uvec nA(solution_dimension, arma::fill::zeros);
        for (int i(0); i < solution_dimension; ++i) {
            if ( A1(i) == A2(i) ) {
                nA(i) = A1(i);
            } else if (arma::randu() > 0.5) {
                nA(i) = A1(i);
            } else {
                nA(i) = A2(i);
            }
        }
        return nA;
    };

    arma::mat tempA(solution_dimension, n, arma::fill::randn); // initialized population
    arma::umat A = tempA > 0.5; // boolean population

    double pc = gen_prob;
    arma::rowvec P(n, arma::fill::zeros); // probability of each member mating
    arma::umat nextA(solution_dimension,n,arma::fill::zeros); // next generation
    int numGens = 1;

    auto nextGen = [&]() {
        arma::rowvec B = fitness(A);
        arma::umat ind = arma::sort_index(-B);
        for (int i(0); i < n; ++i) {
            if (i == n-1) {
                P(i) = std::pow(1-pc, n-1);
            } else {
                P(i) = std::pow(1-pc, i) * pc;
            }
        }
        
        arma::mat mates(2, n, arma::fill::randu); // pairs of members chosen for reproduction
        for (int i(0); i < 2; ++i) {
            for (int j(0); j < n; ++j) {
                for (int k(0); k < n-1; ++k) {
                    if (mates(i,j) < P(0)) {
                        mates(i,j) = ind(0);
                        break;
                    } else if ( arma::accu(P(arma::span(0,k))) < mates(i,j) && mates(i,j) <= arma::accu(P(arma::span(0,k+1))) ) {
                        mates(i,j) = ind(k);
                        break;
                    }
                }
            }
        }
        for (int i(0); i < n; ++i) {
            arma::uvec A1 = A.col(mates(0,i));
            arma::uvec A2 = A.col(mates(1,i));
            nextA.col(i) = crossOvr(A1,A2);
            for (int j(0); j < solution_dimension; ++j) {
                if (arma::randu() < gen_mut_rate) { // mutation rate
                    bool tempij = nextA(j,i);
                    nextA(j,i) = !tempij;
                }
            }
        }
        numGens++;
    }; // produce next generation

    nextGen();
    while (arma::accu(arma::abs(A - nextA)) != 0 ) {
        A = nextA;
        nextGen();
    }
    x = nextA.col(0);
    return f(x);
}