# 🦙 LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions

It forked form [advimman/lama](https://github.com/advimman/lama)

The primary goal is to provide tools for converting the Lama model into TensorFlow, TensorFlow Lite, and ONNX formats.

## **Features**

- Conversion of Lama models to TensorFlow.
- Support for exporting models to TensorFlow Lite for mobile and embedded devices.
- ONNX format compatibility for cross-framework integration.

## **Getting Started**

### **Prerequisites**

```commandline
docker build -t lama-convert . 
```

### **Usage**

1. **Save lama model to `ONNX` and `torch_jit`**:

   using [save_model_to_tflite.py](./save_model_to_jit.py)
   ```bash
   python save_model_to_tflite.py
   ```  

2. **Convert Lama to `TensorFlow` and `TensorflowLite`**:

   using [save_model_to_tflite.py](./save_model_to_tflite.py)
   ```bash
   python save_model_to_tflite.py
   ```

## **Contributing**

Feel free to submit pull requests or report issues. Contributions are welcome!

## **License**

This project follows the same license as the original [Lama repository](https://github.com/advimman/lama).

## **Acknowledgments**

- Special thanks to the creators of [advimman/lama](https://github.com/advimman/lama) for their foundational work.
- Gratitude
  to [Carve-Photos/lama](https://github.com/Carve-Photos/lama)([huggingface](https://huggingface.co/Carve/LaMa-ONNX))
  for resolving the FFT model conversion issue, which significantly improved the conversion process.