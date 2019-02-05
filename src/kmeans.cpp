#include "numerics.hpp"

/* KMEANS : use k-means algorithm to cluster data.
 * --- A : data matrix, each col is a data input.
 * --- k : number of clusters. */
numerics::kmeans::kmeans(arma::mat& A, int k) {
    if (k <= 1) {
        std::cerr << "kmeans() error: invalid input for k (valid k > 1)." << std::endl;
        std::cerr << "\terror thrown during call initialization of a kmeans object." << std::endl;
        C = {NAN};
        k = -1;
        dim = -1;
        dataCluster = {NAN};
        return;
    }

    dim = A.n_rows;
    int numPts = A.n_cols;
    this->k = k;
    this->data = &A;

    // initialized cluster means C by setting them equal to k unique pts from the data
    arma::arma_rng::set_seed_random();
    arma::uvec temp = arma::regspace<arma::uvec>(0, numPts-1); // 0:(numPts-1)
    temp = arma::shuffle(temp); // randomize the order
    temp = temp(arma::span(0,k-1)); // collect the first k elems from the list
    C = A.cols(temp); // all that so they aren't repeating
    
    // begin looping until convergence
    bool notDone = true;
    int n = 0;
    arma::mat C1, sums;
    arma::rowvec counts;
    while (notDone) {
        C1 = C;

        sums = arma::zeros(dim,k);
        counts = arma::zeros<arma::rowvec>(k);
        
        for (int i(0); i < numPts; ++i) {
            int a = closestC(A.col(i));
            sums.col(a) += A.col(i); // if A.col(i) is in the cluster, then add it to the cluster sum
            counts(a)++; // if A.col(i) is in the cluster, then increase the count of vectors in the cluster
        }
        sums.each_row() /= counts; // compute the means of the clusters
        C = sums;

        if (arma::norm(C1 - C,"fro") < 0.01) notDone = false; // convergence criteria
        n++;
        if (n > 100) { // failed to converge in time
            std::cerr << "kmeans() error: failed to converge within the maximum number of iterations allowed." << std::endl
                      << "\treturning best clusters found so far." << std::endl;
            notDone = false;
        }
    }

    // asign a cluster to the original data
    dataCluster = place_in_cluster(A);
}

/* KMEANS : load kmeans object from file on construction.
 * --- in : file/input stream to load object from. */
numerics::kmeans::kmeans(std::istream& in) {
    load(in);
}

/* LOAD : load kmeans object from file on construction.
 * --- in : file/input stream to load object from. */
void numerics::kmeans::load(std::istream& in) {
    int numPts;
    in >> k >> dim >> numPts;
    C = arma::zeros(dim,k);
    data = new arma::mat(dim, numPts);
    dataCluster = arma::zeros<arma::rowvec>(numPts);

    for (size_t i(0); i < dim; ++i) {
        for (int j(0); j < k; ++j) {
            in >> C(i,j);
        }
    }
    for (size_t i(0); i < dim; ++i) {
        for (int j(0); j < numPts; ++j) {
            in >> data->at(i,j);
        }
    }
    for (int i(0); i < numPts; ++i) {
        in >> dataCluster(i);
    }
}

/* LOAD : save kmeans object to file or stream.
 * --- out : file/output stream to save object to. */
void numerics::kmeans::save(std::ostream& out) {
    out << k << " " << dim << " " << data->n_cols << std::endl;
    C.raw_print(out);
    data->raw_print(out);
    dataCluster.raw_print(out);
}

/* CLOSESTC : finding closest cluster to data point (private)
 * x : place x in cluster. */
int numerics::kmeans::closestC(const arma::vec& x) {
    double min = arma::norm(C.col(0) - x);
    int c = 0;
    
    for (int i(1); i < k; ++i) {
        double temp = arma::norm(C.col(i) - x);
        if (temp < min) {
            min = temp;
            c = i;
        }
    }

    return c;
}

/* GETCLUSTERS : return the cluster number of each col of the original data input  */
arma::rowvec numerics::kmeans::getClusters() const {
    return dataCluster;
}

/* GETCENTROIDS : returns a matrix where each col is a cluster mean. */
arma::mat numerics::kmeans::getCentroids() const {
    return C;
}

/* KMEANS(B) : assign cluster numbers to new data.
 * --- B : data matrix each col is a data point. */
arma::rowvec numerics::kmeans::operator()(const arma::mat& B) {
    return place_in_cluster(B);
}

/* PLACE_IN_CLUSTER : assign cluster numbers to new data. same as operator().
 * --- B : data matrix each col is a data point. */
arma::rowvec numerics::kmeans::place_in_cluster(const arma::mat& B) {
    if ( B.n_rows != (unsigned)dim ) {
        std::cerr << "kmeans() error: cannot cluster " << B.n_rows << "dimensional data in a " << dim << "dimensional space." << std::endl
                  << "\terror thrown during evaluation of kmeans::place_in_cluster() with an input of type arma::mat" << std::endl;
        return {-1};
    }
    
    arma::rowvec clustering = arma::zeros<arma::rowvec>(B.n_cols);
    for (unsigned int i(0); i < B.n_cols; ++i) {
        clustering(i) = closestC(B.col(i));
    }
    return clustering;
}

/* KMEANS(X) : assign cluster numbers to new data point.
 * --- x : single data point. */
int numerics::kmeans::operator()(const arma::vec& x) {
    return place_in_cluster(x);
}

/* PLACE_IN_CLUSTER : assign cluster number to data.
 * --- x : single data point. */
int numerics::kmeans::place_in_cluster(const arma::vec& x) {
    if (x.n_elem != dim) {
        std::cerr << "kmeans() error: cannot cluster " << x.n_elem << "dimensional data in a " << dim << "dimensional space." << std::endl
                  << "\terror thrown during evaluation of kmeans::place_in_cluster() with an input of type arma::vec" << std::endl;
        return -1;
    }
    return closestC(x);
}

/* ALL_FROM_CLUSTER : return all data points in a cluster.
 * --- i : cluster number. */
arma::mat numerics::kmeans::all_from_cluster(int i) {
    if (i < 0 || i >= k) {
        std::cerr << "kmeans() error: invalid choice of cluster, clusters are ordered 0 to " << k-1 << std::endl
                  << "\terror thrown during call to kmeans::all_from_cluster()" << std::endl;
        return {NAN};
    }
    
    arma::mat cluster = data->cols( arma::find(dataCluster == i) );
    return cluster;
}

/* KMEANS[i] : return all data points in the i^th cluster.
 * --- i : cluster number. */
arma::mat numerics::kmeans::operator[](int i) {
    return all_from_cluster(i);
}

/* SUMMARY : print overview of kmeans analysis.
 * out : file/output stream to print analysis to. */
std::ostream& numerics::kmeans::summary(std::ostream& out) {
    out << "----------------------------------------------------" << std::endl;
    out << "\t\tk-means clustering" << std::endl
        << "number of clusters: " << k << "\t\tnumber of data entries: " << data->n_cols << std::endl
        << "\t\t\tcentroids:" << std::endl;
    
    out << "\t";
    for (int i(0); i < k; ++i) out << "|\t" << i << "\t|";
    out << std::endl << std::setprecision(3) << std::fixed;
    
    for (int i(0); i < k; ++i) {
        out << "\t";
        for (size_t j(0); j < dim; ++j) {
            out << "|\t" << C(i,j) << "\t|";
        }
        out << std::endl;
    }
    out << "----------------------------------------------------" << std::endl;
    return out;
}