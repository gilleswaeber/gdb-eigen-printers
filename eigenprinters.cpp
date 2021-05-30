// This file declares and instantiates some types for use with the improved Eigen printer
// Compile as library and it will be automatically loaded when needed
// https://github.com/gilleswaeber/gdb-eigen-printers.git

namespace EigenPrinters {
    struct Info { char _; } instInfo;
    struct RowFirst { char _; } instRowFirst;
    struct ColFirst { char _; } instColFirst;
    struct ByRow { char _; } instByRow;
    struct ByCol { char _; } instByCol;
}