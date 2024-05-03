#include "lala/SMT_parser.hpp"
#include "battery/allocator.hpp"

#include <iostream>


void testing(lala::SMTOutput<battery::standard_allocator>& smt_output){
    lala::impl::SMTParser<battery::standard_allocator> smt_parser(smt_output);
    smt_parser.test();
}


int main(){
    std::cout << "-------------------- Start SMT file parsing test --------------------" << std::endl;
    
    // auto f = lala::parse_smt<battery::standard_allocator>("test.smt");
    // -> test.smt does not exist:

    // testing section
    const battery::standard_allocator allocator = battery::standard_allocator();
    lala::SMTOutput<battery::standard_allocator> smt_output(allocator);
    testing(smt_output);    
    


    
    std::cout << "-------------------- End SMT file parsing test --------------------" << std::endl;


    return 0;
}