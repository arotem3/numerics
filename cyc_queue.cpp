#include "numerics.hpp"

//--- build cyclic queue with random access, utility class for LBFGS-------//
//----- num_rows: number of rows per element (each element is a vector) ---//
//----- max_size: total number of elements in cyclic queue ----------------//
numerics::cyc_queue::cyc_queue(size_t num_rows, size_t max_size) {
    max_elem = max_size;
    size = 0;
    head = 0;
    A = arma::zeros(num_rows, max_size);
}

//--- add an element to the queue, removing first in element if the queue is full ---//
void numerics::cyc_queue::push(arma::vec& x) {
    if (size < max_elem) { // queue not full
        A.col(size) = x;
        size++;
    } else { // queue is full
        A.col(head) = x;
        head = mod(head + 1, size);
    }
}

//--- element access for queue ---//
arma::vec numerics::cyc_queue::operator()(size_t i) {
    if (i >= size) {
        std::cerr << "cyc_queue::element access out of bounds." << std::endl;
        return {0};
    }
    else {
        int ind = mod(i + head, size);
        return A.col(ind);
    }
}

//--- return current length of cyclic queue (relevent especially when not full) ---//
int numerics::cyc_queue::length() {
    return size;
}

//--- return num_rows of the elements (each a vector) of the queue ---//
int numerics::cyc_queue::col_size() {
    return A.n_rows;
}