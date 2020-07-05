#include "numerics.hpp"

/* cyc_queue(num_rows, max_size) : build cyclic queue with random access, utility class.
 * --- num_rows: number of rows per element (each element is a vector).
 * --- max_size: total number of elements in cyclic queue. */
numerics::numerics_private_utility::cyc_queue::cyc_queue(uint num_rows, uint max_size) {
    max_elem = max_size;
    size = 0;
    head = 0;
    A = arma::zeros(num_rows, max_size);
}

/* push(x) : add an element to the queue, removing first in element if the queue is full */
void numerics::numerics_private_utility::cyc_queue::push(const arma::vec& x) {
    if (size < max_elem) { // queue not full
        A.col(size) = x;
        size++;
    } else { // queue is full
        A.col(head) = x;
        head = mod(head + 1, size);
    }
}

/* cyc_queue::(i) : element access. */
arma::vec numerics::numerics_private_utility::cyc_queue::operator()(uint i) {
    if (i >= size) {
        std::cerr << "cyc_queue::element access out of bounds." << std::endl;
        return {0};
    } else {
        int ind = mod(i + head, size);
        return A.col(ind);
    }
}

/* end() : last element. */
arma::vec numerics::numerics_private_utility::cyc_queue::end() {
    return A.col(size-1);
}

/* length() : return current length of cyclic queue (relevent especially when not full) */
int numerics::numerics_private_utility::cyc_queue::length() {
    return size;
}

/* col_size() : return num_rows of the elements (each a vector) of the queue */
int numerics::numerics_private_utility::cyc_queue::col_size() {
    return A.n_rows;
}

/* clear() : empty out queue. */
void numerics::numerics_private_utility::cyc_queue::clear() {
    A.fill(0);
}

/* data() : return matrix storing all elements in queue. */
arma::mat numerics::numerics_private_utility::cyc_queue::data() {
    arma::mat D  = arma::zeros(A.n_rows, size);
    for (unsigned int i(0); i < size; ++i) {
        D.col(i) = operator()(i);
    }
    return D;
}