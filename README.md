# simple FFT

This is a collection of simple Gauss-style FFT implementations in recursive and iterative styles. It also includes a brute force FT, IFT and performance comparisons. 

# Building 

## With plotting

Python and Matplotlib are required for plotting the output. 

If on linux remove the include of numpy in the CMakeLists.txt as this is automatic.

If on Windows and matplotlib is installed in the global site-packages, update the include path accordingly.

Then just configure and build the project. 

## Without plotting

If you don't want plotting, set the PLOT_OUTPUT define to 0 in the FFT.cpp file. Then you can build the project without Python and Matplotlib simply by building the single file.
