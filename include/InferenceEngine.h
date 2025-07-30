// Inferon/include/InferenceEngine.h

#ifndef INFERON_INFERENCE_ENGINE_H
#define INFERON_INFERENCE_ENGINE_H

#include "onnx_parser/include/ONNXParser.h"
#include "DenseGEMM.h"
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

namespace inferon {

class InferenceEngine {
public:
    /**
    * @brief construct an ineference engine and parse the ONNX model
    * @param model_path the path to the model's ONNX file 
    **/

    InferenceEngine(const std::string& model_path);

    /**
    * @brief executes the looded ONNX model
    * @param input_name name of the input tensor
    * @param input_data data for the input tensor
    * @param input_dims dims for the input tensor, 
    * @return map of output tensor names -> data
    - from the ONNX graph
     */

    std::map<std::string, std::vector<float>> run(
        const std::string& input_name, 
        const std::vector<float>& input_data, 
        const std::vector<int64_t>& input_dims
    ); 

private:
    ONNXParser _parser; 
    std::vector<OperationNode> _graph;

    std::map<std::string, CSRMatrix> _initial_tensors; // weights and biases from the initializer
    std::map<std::string, std::vector<float>> _intermediate_tensors; // activations and intermediate results

    // now, for the helper functions

    // to get tensor data
    std::vector<float> get_tensor_data(const std::string& name);
    std::vector<int64_t> get_tensor_dims(const std::string& name);

    // to store tensor data 
    void set_tensor_data(const std::string& name, std::vector<float>& data);
    void set_tensor_dims(const std::string& name, std::vector<int64_t>& dims);

}; // end of class def
} // namespace inferon