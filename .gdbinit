python
import sys
sys.path.insert(0, '/install/dir')
from gdb_eigen_printers import register_eigen_printers
register_eigen_printers(None)
end