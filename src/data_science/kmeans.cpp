#include <numerics.hpp>

/* kmeans(A, k) : use k-means algorithm to cluster data.
 * --- A : data matrix, each col is a data input.
 * --- k : number of clusters. */
numerics::kmeans::kmeans(arma::mat& A, int k) {
    if (k <= 1) {
        std::cerr << "kmeans() error: invalid input for k (valid 1 < k < #data)." << std::endl;
        std::cerr << "\terror thrown during call initialization of a kmeans object." << std::endl;
        C = {NAN};
        k = -1;
        dim = -1;
        dataCluster = {NAN};
        return;
    }

    dim = A.n_cols;
    int numPts = A.n_rows;
    this->k = k;
    data = A;

    // initialized cluster means C by setting them equal to k unique pts from the data
    C = init_clusters();
    
    // begin looping until convergence
    bool notDone = true;
    num_iters = 0;
    arma::mat means;
    arma::vec counts;
    while (notDone) {
        means = arma::zeros(k,dim);
        counts = arma::zeros(k);
        
        for (int i=0; i < numPts; ++i) {
            int a = closest_cluster(A.row(i));
            means.row(a) += A.row(i); // if A.row(i) is in the cluster, then add it to the cluster sum
            counts(a)++; // if A.row(i) is in the cluster, then increase the count of vectors in the cluster
        }
        means.each_col() /= counts; // compute the means of the clusters

        if (arma::norm(means - C,"fro") < 0.01) notDone = false; // convergence criteria
        C = means;
        num_iters++;
        if (num_iters >= 100) { // failed to converge in time
            std::cerr << "kmeans() error: failed to converge within the maximum number of iterations allowed." << std::endl
                      << "\treturning best clusters found so far." << std::endl;
            notDone = false;
        }
    }

    // asign a cluster to the original data
    dataCluster = predict(A);
}

/* init_clusters() : kmeans++ algorithm for initialising
 * clusters. Returns the initial cluster centroids. */
arma::mat numerics::kmeans::init_clusters() {
    int n = data.n_rows;
    int i = arma::randi( arma::distr_param(0, n-1) );
    arma::mat C = arma::zeros(k,dim);
    C.row(0) = data.row(i);
    for (int m = 1; m < k; ++m) {
        arma::vec D = arma::zeros(n);
        for (i=0; i < n; ++i) {
            double d = arma::norm( data.row(i) - C.row(0) );
            for (int j=1; j<m; ++j) {
                double d1 = arma::norm( data.row(i) - C.row(j) );
                if (d1 < d) d = d1;
            }
            D(i) = d*d;
        }
        D /= arma::sum(D);
        i = sample_from(D);
        C.row(m) = data.row(i);
    }
    return C;
}

/* kmeans(in) : load kmeans object from file on construction.
 * --- in : file/input stream to load object from. */
numerics::kmeans::kmeans(std::istream& in) {
    load(in);
}

/* load(in) : load kmeans object from file on construction.
 * --- in : file/input stream to load object from. */
void numerics::kmeans::load(std::istream& in) {
    int numPts;
    in >> k >> dim >> numPts;
    C = arma::zeros(dim,k);
    data = arma::mat(dim, numPts);
    dataCluster = arma::zeros<arma::rowvec>(numPts);

    for (uint i(0); i < dim; ++i) {
        for (int j(0); j < k; ++j) {
            in >> C(i,j);
        }
    }
    for (uint i(0); i < dim; ++i) {
        for (int j(0); j < numPts; ++j) {
            in >> data(i,j);
        }
    }
    for (int i(0); i < numPts; ++i) {
        in >> dataCluster(i);
    }
}

/* save(out) : save kmeans object to file or stream.
 * --- out : file/output stream to save object to. */
void numerics::kmeans::save(std::ostream& out) {
    out << k << " " << dim << " " << data.n_cols << std::endl;
    C.raw_print(out);
    data.raw_print(out);
    dataCluster.raw_print(out);
}

/* closest_cluster(x) : finding closest cluster to data point (private)
 * x : place x in cluster. */
int numerics::kmeans::closest_cluster(const arma::rowvec& x) {
    double min = arma::norm(C.row(0) - x);
    int c = 0;
    
    for (int i(1); i < k; ++i) {
        double temp = arma::norm(C.row(i) - x);
        if (temp < min) {
            min = temp;
            c = i;
        }
    }

    return c;
}

/* get_clusters() : return the cluster number of each col of the original data input  */
arma::vec numerics::kmeans::get_clusters() const {
    return dataCluster;
}

/* get_centroids() : returns a matrix where each col is a cluster mean. */
arma::mat numerics::kmeans::get_centroids() const {
    return C;
}

/* kmeans::(B) : assign cluster numbers to new data.
 * --- B : data matrix each col is a data point. */
arma::vec numerics::kmeans::operator()(const arma::mat& B) {
    return predict(B);
}

/* predict(B) : assign cluster numbers to new data. same as operator().
 * --- B : data matrix each col is a data point. */
arma::vec numerics::kmeans::predict(const arma::mat& B) {
    if ( B.n_cols != (unsigned)dim ) {
        std::cerr << "kmeans() error: cannot cluster " << B.n_cols << "dimensional data in a " << dim << "dimensional space." << std::endl
                  << "\terror thrown during evaluation of kmeans::predict() with an input of type arma::mat" << std::endl;
        return {-1};
    }
    
    arma::vec clustering = arma::zeros(B.n_rows);
    for (unsigned int i(0); i < B.n_rows; ++i) {
        clustering(i) = closest_cluster(B.row(i));
    }
    return clustering;
}

/* kmeans::(x) : assign cluster numbers to new data point.
 * --- x : single data point. */
int numerics::kmeans::operator()(const arma::rowvec& x) {
    return predict(x);
}

/* predict(x) : assign cluster number to data.
 * --- x : single data point. */
int numerics::kmeans::predict(const arma::rowvec& x) {
    if (x.n_elem != dim) {
        std::cerr << "kmeans() error: cannot cluster " << x.n_elem << "dimensional data in a " << dim << "dimensional space." << std::endl
                  << "\terror thrown during evaluation of kmeans::predict() with an input of type arma::rowvec" << std::endl;
        return -1;
    }
    return closest_cluster(x);
}

/* all_from_cluster(i) : return all data points in a cluster.
 * --- i : cluster number. */
arma::mat numerics::kmeans::all_from_cluster(uint i) {
    if (i < 0 || k <= i) {
        std::cerr << "kmeans() error: invalid choice of cluster, clusters are ordered 0 to " << k-1 << std::endl
                  << "\terror thrown during call to kmeans::all_from_cluster()" << std::endl;
        return {NAN};
    }
    
    arma::mat cluster = data.rows( arma::find(dataCluster == i) );
    return cluster;
}

/* kmeans::[i] : return all data points in the i^th cluster.
 * --- i : cluster number. */
arma::mat numerics::kmeans::operator[](uint i) {
    return all_from_cluster(i);
}

/* summary(out) : print overview of kmeans analysis.
 * out : file/output stream to print analysis to. */
std::ostream& numerics::kmeans::summary(std::ostream& out) {
    out << "----------------------------------------------------" << std::endl;
    out << "\t\tk-means clustering" << std::endl
        << "number of clusters: " << k << "\t\tnumber of data entries: " << data.n_cols << std::endl
        << "\t\t\tcentroids:" << std::endl;
    
    out << "\t";
    for (int i(0); i < k; ++i) out << "|\t" << i << "\t|";
    out << std::endl << std::setprecision(3) << std::fixed;
    
    for (int i(0); i < k; ++i) {
        out << "\t";
        for (uint j(0); j < dim; ++j) {
            out << "|\t" << C(j,i) << "\t|";
        }
        out << std::endl;
    }
    out << "----------------------------------------------------" << std::endl;
    return out;
}

/* help(out) : prints out documentation for the kmeans object.
 * out : file/output stream to print docstring to. */
 std::ostream& numerics::kmeans::help(std::ostream& out) {
     out << "----------------------------------------------------" << std::endl
         << "out = kmeans(arma::mat data, int k) :" << std::endl
         << "\tdata : data to cluster, each row is a datum instance." << std::endl
         << "\tk : number of clusters." << std::endl << std::endl
         << "Clusters data according to the K-Means heuristic" << std::endl
         << "using the K-Means++ procedure for cluster initialization" << std::endl
         << "returns an object that can be used as a clasifier." << std::endl << std::endl
         << "obj : " << std::endl
         << "\tload/save(ostream) : functions to save and load function to stream/file" << std::endl
         << "\tgetClusters : returns cluster labels of original data." << std::endl
         << "\tgetCentroids : returns cluster centers in order, i.e. listed 0,...,k-1" << std::endl
         << "\tpredict(matrix) : given new data matrix, predict the likely associated cluster" << std::endl
         << "\toperator()(matrix) : same as predict" << std::endl
         << "\tall_from_cluster(int) : returns submatrix of the data containing the data from clusters" << std::endl
         << "\toperator[](int) : same as all_from_cluster." << std::endl
         << "\tsummary(ostream) : print summary results about the clustering." << std::endl
         << "----------------------------------------------------" << std::endl;
     return out;
 }