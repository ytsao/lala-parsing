// Copyright 2024 Yi-Nung Tsao

#ifndef LALA_SMT_PARSER_HPP
#define LALA_SMT_PARSER_HPP

#include <iostream> 
#include <optional>
#include <string> 
#include <fstream>
#include <streambuf>
#include "peglib.h"
#include "onnx/onnx.proto3.pb.h"

#include "battery/shared_ptr.hpp"
#include "lala/logic/ast.hpp"


typedef std::map<std::string, std::pair<float, float>> box;
typedef std::vector<std::vector<float*>>matrix;
typedef std::vector<float*> vector;
typedef std::vector<std::pair<size_t, size_t>> layers;


namespace lala{

template<class Allocator>
class SMTOutput{
    using bstring = battery::string<Allocator>;
    template<class T> using bvector = battery::vector<T, Allocator>;
    using array_dim_t = bvector<battery::tuple<size_t,size_t>>;
    using F = TFormula<Allocator>;

    bvector<bstring> output_vars;
    // For each array, we store the name of the array, the dimension of the array, and the type of the array.
    bvector<battery::tuple<bstring, array_dim_t, bvector<bstring>>> output_arrays;

public:
    template<class Alloc2>
    friend class SMTOutput;

    CUDA SMTOutput(const Allocator& alloc)
        : output_vars(alloc)
        , output_arrays(alloc) 
    {}

    SMTOutput(SMTOutput&&) = default;
    SMTOutput<Allocator>& operator=(const SMTOutput<Allocator>&) = default;

    template<class Alloc>
    CUDA SMTOutput<Allocator>& operator=(const SMTOutput<Alloc>& other){
        output_vars = other.output_vars;
        output_arrays = other.output_arrays;
        return *this;
    }

    template<class Alloc2>
    CUDA SMTOutput(const SMTOutput<Alloc2>& other, const Allocator& allocator = Allocator{})
        : output_vars(other.output_vars, allocator)
        , output_arrays(other.output_arrays, allocator)
    {}
};


namespace impl{
/**/
inline logic_real string_to_real(const std::string& s){
    #if !defined(__GNUC__) && !defined(_MSC_VER)
        #pragma STDC FENV_ACCESS ON
    #endif 

    int old = std::fegetround();
    int r = std::fesetround(FE_DOWNWARD);
    assert(r == 0);
    double lb = std::strtod(s.c_str(), nullptr);
    r = std::fesetround(FE_UPWARD);
    assert(r == 0);
    double ub = std::strtod(s.c_str(), nullptr);
    std::fesetround(old);
    return battery::make_tuple(lb, ub);
}

template<class Allocator> 
class SMTParser{
    using allocator_type = Allocator;
    using F = TFormula<allocator_type>;
    using SV = peg::SemanticValues;
    using Set = logic_set<F>;
    using So = Sort<allocator_type>;
    using bstring = battery::string<Allocator>;
    using FSeq = typename F::Sequence;

    std::map<std::string, F> params; // Name and value of the parameters occuring in the model.
    std::map<std::string, int> arrays; // Size of all named arrays (parameters and variables).
    bool error; // If an error was found during parsing.
    bool silent; // If we do not want to output error messages.
    SMTOutput<Allocator>& output;

    // Contains all the annotations ignored.
    // It is used to avoid printing an error message more than once per annotation.
    std::set<std::string> ignored_annotations;

    enum class TableKind{
        PLAIN, 
        SHORT,
        COMPRESSED,
        BASIC
    };

public: 
    SMTParser(SMTOutput<Allocator>& output): error(false), silent(false), output(output){}

    battery::shared_ptr<F, allocator_type> parse(const std::string& input){
        peg::parser parser(R"(
            # Grammar for SMT format files
            Statement <- (Identifier)

            Identifier <- 'declare-const' / 'assert'

            VariableName <- [X/Y]_[Integer]
            ValueType <- Real
            Real <- < (
                    'inf'
                /   '-inf'
                /   [+-]? [0-9]+ (('.' (&'..' / !'.') [0-9]*) / ([Ee][+-]?[0-9]+)) ) >
            Integer <- < [+-]?[0-9]+ >


            ConstraintType <- 'or' / 'and'
            Signs <- '<=' / '>='

            %whitespace <- [ \n\r\t]*
        )");

        assert(static_cast<bool>(parser) = true);

        parser["Integer"] = [](const SV& sv) {return F::make_z(sv.token_to_number<logic_int>());};
        parser["Real"] = [](const SV& sv) {return F::make_real(string_to_real(sv.token_to_string())); };
        parser["Identifier"] = [](const SV& sv) {return sv.token_to_string();};
        parser["VariableName"] = [](const SV& sv){};
        parser["ValueType"] = [](const SV& sv){};        
        parser["ConstraintType"] = [](const SV& sv){};
        parser["Signs"] = [](const SV& sv){};
        parser.set_logger([](size_t line, size_t col, const std::string& msg, const std::string& rule){
            std::cerr << line << ":" << col << ": " << msg << "\n";
        });

        F f;
        if (parser.parse(input.c_str(), f) && !error){
            // return battery::
            std::cout << "Parsing successful" << std::endl;
        }
        else{
            return nullptr;
        }
    }

    void test(){
        std::cout << "Test" << std::endl;   
    }
private:

    size_t get_num_inputs(const std::string onnx_filename){
        size_t result = 0;

        return result;
    }

    size_t get_num_outputs(const std::string onnx_filename){
        size_t result = 0;

        return result;
    }

    matrix get_weights_matrix(const std::string onnx_filename){
        matrix result;

        std::ifstream input(onnx_filename, std::ios::ate | std::ios::binary); // open file and move current position in file to the end
        std::streamsize size = input.tellg(); // get current position in file 
        input.seekg(0, std::ios::beg); // move to start of file 
        std::vector<char> buffer(size);
        input.read(buffer.data(), size); // read raw data

        onnx::ModelProto model;
        model.ParseFromArray(buffer.data(), size);
        
        for (const onnx::TensorProto& tensor : model.graph().initializer()){
            if (tensor.name().find("_W") != std::string::npos || tensor.name().find("MatMul") != std::string::npos){
                float *data = (float *)tensor.raw_data().c_str();

                assert(tensor.dims_size() == 2);
                int input_dim = tensor.dims(0);
                int output_dim = tensor.dims(1);
                assert(input_dim * output_dim == tensor.raw_data().size() / sizeof(float));
                int count = 0;
                for (int i = 0; i < input_dim; i++){
                    vector v;
                    for (int j = 0; j < output_dim; j++){
                        v.push_back(&data[count]);
                        count++;
                    }
                    result.push_back(v);
                }
            }
        }

        return result;
    }

    vector get_biases_vector(const std::string onnx_filename){
        vector result;

        std::ifstream input(onnx_filename, std::ios::ate | std::ios::binary); // open file and move current position in file to the end 
        std::streamsize size = input.tellg(); // get current position in file
        input.seekg(0, std::ios::beg); // move to start of file
        std::vector<char> buffer(size);
        input.read(buffer.data(), size); // read raw data

        onnx::ModelProto model;
        model.ParseFromArray(buffer.data(), size); // parse protobuf

        for (const onnx::TensorProto& tensor : model.graph().initializer()){
            if (tensor.name().find("B") != std::string::npos || tensor.name().find("Add") != std::string::npos){
                float *data = (float *)tensor.raw_data().c_str();
                int output_dim = tensor.dims(0);
                assert(output_dim == tensor.raw_data().size() / sizeof(float));
                for (int i = 0; i < output_dim; i++){
                    result.push_back(&data[i]);
                }
            }
        }

        return result;
    }

    layers get_layer_dim(const std::string onnx_filename){
        layers result;

        return result;
    }
};
}

    /*
    Parse an SMT-LIB file and return the corresponding formula.
    Ref:
    */
    template<class Allocator>
    battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt_str(const std::string& input, SMTOutput<Allocator>& output){
        impl::SMTParser<Allocator> parser(output);
        return parser.parse(input);
    }

    template<class Allocator>
    battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt(const std::string& filename, SMTOutput<Allocator>& output){
        std::ifstream t(filename);
        if(t.is_open()){
            std::string input((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
            return parse_smt_str(input, output);
        }
        else{
            std::cerr << "FIle `" << filename << "` does not exist:." << std::endl;
        }
        return nullptr;
    }

    template<class Allocator>
    battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt_str(const std::string& input, const Allocator& allocator = Allocator()){
        SMTOutput<Allocator> output(allocator);
        return parse_smt_str(input, output);
    }

    template<class Allocator>
    battery::shared_ptr<TFormula<Allocator>, Allocator> parse_smt(const std::string& filename, const Allocator& allocator = Allocator()){
        SMTOutput<Allocator> output(allocator);
        return parse_smt(filename, output);
    }
}

#endif // LALA_SMT_PARSER_HPP