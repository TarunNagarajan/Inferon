#include "InferenceEngine.h"
#include <iostream>

namespace inferon {
    
    InferenceEngine::InferenceEngine(const std::string& model_path) {
        if (!_parser.parse(model_path)) {
            throw std::runtime_error("Failed to parse ONNX model: " + model_path);
        }
        _graph = _parser.get_graph(); 
        _initial_tensors = _parser.get_tensors(); 
        
        std::cout << "ONNX model parsed successfully. Graph contains " << _graph.size() << " nodes." << std::endl;
    }

    std::vector<float> InferenceEngine::get_tensor_data(const std::string& name) {
        auto it_intermediate = 
        
    }
}