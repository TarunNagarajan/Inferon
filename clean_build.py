import shutil
import os

onnx_src_dir = os.path.join('C:', 'Users', 'ultim', 'Downloads', 'Inferon', 'build', 'onnx_cpp-src')

if os.path.exists(onnx_src_dir):
    print(f"Attempting to remove: {onnx_src_dir}")
    shutil.rmtree(onnx_src_dir, ignore_errors=True)
    if not os.path.exists(onnx_src_dir):
        print(f"Successfully removed: {onnx_src_dir}")
    else:
        print(f"Failed to remove: {onnx_src_dir}. It might be in use. Please close any programs that might be accessing this directory.")
else:
    print(f"ONNX source directory not found: {onnx_src_dir}")