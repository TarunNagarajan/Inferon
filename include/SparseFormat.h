#ifndef INFERON_SPARSE_FORMAT_H
#define INFERON_SPARSE_FORMAT_H

class SparseFormat {
public:
    virtual ~SparseFormat() = default;
    // Pure virtual method to define the interface for sparse format handling
    // Specific sparse formats (e.g., CSR, CSC) will inherit from this.
};

#endif // INFERON_SPARSE_FORMAT_H
