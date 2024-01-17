

Building:

To get LTO for sleef to work, I've configured with the following
```
CC=clang CXX=clang++ cmake -B build -S . -G Ninja -DLLVM_AR_COMMAND=llvm-ar-14
```
Replace `llvm-ar-14` with the appropriate version number present on your machine