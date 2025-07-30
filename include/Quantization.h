#ifndef INFERON_QUANTIZATION_H
#define INFERON_QUANTIZATION_H

class QuantizationScheme {
public:
    virtual ~QuantizationScheme() = default;
    // Pure virtual method to define the interface for quantization schemes
    // Specific schemes (e.g., per-tensor, per-channel) will inherit from this.
    // For now, just a placeholder.
};

#endif // INFERON_QUANTIZATION_H
