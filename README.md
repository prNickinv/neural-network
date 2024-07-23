# Neural Network

This is a neural network implementation in C++.

## Getting started

Clone the repository and run the following commands to enable the submodules:

```bash
git submodule update --init --recursive
```
Then run the following commands to build the project:

```bash
mkdir build && cd build
cmake ..
make
```

If you want to run the unit-tests, you can run the following command:

```bash
cd tests && ctest
```

If you want to check, what our library is capable of, you can run the example program, 
which trains a neural network to recognize handwritten digits from the MNIST dataset and 
fashion products from the Fashion MNIST dataset:

```bash
./neural-network
```

The example of building, training and testing a neural network can be found in
`MnistTest.h` and `MnistTest.cpp` files.
