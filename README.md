Improved GDB Eigen Printers
===========================
Allows browsing Eigen objects in the GDB debugger in a more convenient way, especially in IDEs like CLion.

Based on the built-in Eigen printer script https://gitlab.com/libeigen/eigen/-/blob/master/debug/gdb/printers.py

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
3. Compile the `eigenprinters.cpp` file as a library with debug symbols
4. Build you project and run with the gdb debugger

## Installation script
```sh
INSTALL_DIR=~/.gdb_eigen_printers
git clone https://github.com/gilleswaeber/gdb-eigen-printers.git "$INSTALL_DIR"
printf >> ~/.gdbinit "\npython\nimport sys\nsys.path.insert(0, r'''%s''')\nfrom gdb_eigen_printers import register_eigen_printers\nregister_eigen_printers(None)\nend\n" "$INSTALL_DIR"
g++ -g -c "$INSTALL_DIR/eigenprinters.cpp" -o "$INSTALL_DIR/eigenprinters.o"  # or python3 "$INSTALL_DIR/compile.py"
```

## License
As for the original, the code is available under the [Mozilla Public License, v. 2.0](http://mozilla.org/MPL/2.0/)

## Known issues
- May have issues with non-initialized memory
