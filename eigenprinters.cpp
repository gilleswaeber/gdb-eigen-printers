// This file declares and instantiates some types for use with the improved Eigen printer
// It must be linked to the project to allow the printer to be used fully
// There is no need to include this file with a #include directive
// Also, do not refer to those types in the code
// https://github.com/gilleswaeber/gdb-eigen-printers.git

namespace EigenPrinters {
    struct Info { char _; } instInfo;
    struct RowFirst { char _; } instRowFirst;
    struct ColFirst { char _; } instColFirst;
    struct ByRow { char _; } instByRow;
    struct ByCol { char _; } instByCol;
}