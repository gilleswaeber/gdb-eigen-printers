Improved GDB Eigen Printers
===========================
Allows browsing Eigen objects in the GDB debugger in a more convenient way, especially in IDEs like CLion.

## Features
- Matrices in row-first or column-first order
- Support for sparse matrices
- per-row and per-column formatted display

## Setup
1. Clone this repository
2. Create a `.gdbinit` file in your home directory with the following contents:
```.gdbinit
python
import sys
sys.path.insert(0, '/path/to/the/clone/of/this/repo')
from gdb_eigen_printers import register_eigen_printers
register_eigen_printers(None)
end
```
3. Add the following declarations to a `.cpp` file in the project you want to debug:
```cpp
namespace EigenPrinters {
    struct Info { char _; } instInfo;
    struct RowFirst { char _; } instRowFirst;
    struct ColFirst { char _; } instColFirst;
    struct ByRow { char _; } instByRow;
    struct ByCol { char _; } instByCol;
}
```
4. Build you project and run with the gdb debugger

## Installation script
```sh
INSTALL_DIR=~/.gdb_eigen_printers
git clone https://github.com/gilleswaeber/gdb-eigen-printers.git "$INSTALL_DIR"
printf >> ~/.gdbinit "\npython\nimport sys\nsys.path.insert(0, r'''%s''')\nfrom gdb_eigen_printers import register_eigen_printers\nregister_eigen_printers(None)\nend\n" "$INSTALL_DIR"
cp --interactive "$INSTALL_DIR/eigenprinters.cpp" ./
```