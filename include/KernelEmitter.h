#ifndef INFERON_KERNEL_EMITTER_H
#define INFERON_KERNEL_EMITTER_H

// feat: forward decl.
class Operator;

class KernelEmitter {
public:
    virtual ~KernelEmitter() = default;
    virtual void emit(const Operator& op) = 0;
};

#endif // INFERON_KERNEL_EMITTER_H
