import sys
import torch
try:
    print("PyTorch imported successfully.")
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("PyTorch is not installed.")

def main():
    print("Hello from jigsaws-pytorch-dataset!")

    print("Python interpreter path:", sys.executable)


if __name__ == "__main__":
    main()
