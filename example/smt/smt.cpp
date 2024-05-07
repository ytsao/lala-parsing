#include "lala/SMT_parser.hpp"
#include "battery/allocator.hpp"

#include <iostream>



int main(){
    std::cout << "-------------------- Start SMT file parsing test --------------------" << std::endl;
    
    // auto f = lala::parse_smt<battery::standard_allocator>("test.smt");
    // -> test.smt does not exist:

    // test for onnx file parsing
    // test();

    // test for vnnlib file parsing
    // auto f = lala::parse_smt<battery::standard_allocator>("test_prop.vnnlib");
    // if(!f) {
    //     std::cerr << "Could not parse the SMT file test_prop.vnnlib" << std::endl;
    //     return 1;
    // }
    // else {
    //     std::cout << "Successful parsing of the file test_prop.vnnlib" << std::endl;
    //     std::cout << "Show parsed data:" << std::endl;
        
    // }
    
    std::cout << "\n-------------------- End SMT file parsing test --------------------" << std::endl;


    return 0;
}