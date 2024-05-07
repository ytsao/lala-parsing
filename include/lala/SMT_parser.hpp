// Copyright 2024 Yi-Nung Tsao

#ifndef LALA_SMT_PARSER_HPP
#define LALA_SMT_PARSER_HPP

#include <iostream> 
#include <optional>
#include <string> 
#include <fstream>
#include <streambuf>
#include "peglib.h"
#include "onnx.proto3.pb.h"

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
        //TODO: Missing implementation about post-condition contains 'and' & 'or' operators
        peg::parser parser(R"(
            Statements <- ( '(' VariableDeclaration ')' / '(' InputRegion ')' / '(' Property ')' / Comment)+

            VariableDeclaration <- 'declare-const' VariableName ValueType
            VariableName <- < [a-zA-Z][a-zA-Z0-9_]* >
            ValueType <- 'Real'
            Value <- < (
                'inf'
                / '-inf'
                / [+-]? [0-9]+ (('.' (&'..' / !'.') [0-9]*) / ([Ee][+-]?[0-9]+)) ) >

            InputRegion <- 'assert' '(' Signs VariableName Value ')' 
            Property <- 'assert' '(' Signs VariableName VariableName ')'

            Signs <- '<=' / '>='

            ~Comment    <- ';' [^\n\r]* [ \n\r\t]*
            %whitespace <- [ \n\r\t]*
        )");

        assert(static_cast<bool>(parser) == true);

        parser["Value"] =[](const SV& sv){return F::make_real(string_to_real(sv.token_to_string()));};
        parser["InputRegion"] = [](const SV& sv) {return sv.token_to_string();};
        parser["VariableDeclaration"] = [](const SV& sv){F();}; // work, but, why F()?
        parser["Property"] = [](const SV& sv){return sv.token_to_string();};
        parser["VariableName"] = [](const SV& sv){return sv.token_to_string();}; 
        parser["ValueType"] = [](const SV& sv){return sv.token_to_string();};
        parser["Signs"] = [](const SV& sv){return sv.token_to_string();};
        parser["Statements"] = [this](const SV& sv){return make_statements(sv);}; // error, bad_any_cast()
        
        parser.set_logger([](size_t line, size_t col, const std::string& msg, const std::string& rule){
            std::cerr << line << ":" << col << ": " << msg << "\n";
        });

        F f;
        if (parser.parse(input.c_str(), f) && !error){
            return battery::make_shared<TFormula<Allocator>, Allocator>(std::move(f));
        }
        else{
            return nullptr;
        }
    }

private:

    /*
    from vnnlib file information
    */
    static F f(const std::any& any){
        return std::any_cast<F>(any);
    }

    static battery::tuple<F, F> itv(const std::any& any){
        return std::any_cast<battery::tuple<F, F>>(any);
    }

    F make_error(const SV& sv, const std::string& msg){
        if (!silent){
            std::cerr << sv.line_info().first << ":" << sv.line_info().second << ": " << msg << std::endl;
        }
        error = true;
        return F::make_false();
    }

    F make_statements(const SV& sv){
        std::cout << "test make_statements function" << std::endl;
        if (sv.size() == 1){
            return f(sv[0]);
        }
        else{
            FSeq children;
            for (int i = 0; i < sv.size(); ++i){
                F formula = f(sv[i]);
                if (!formula.is_true()){
                    children.push_back(formula);
                }
            }
            return F::make_nary(AND, std::move(children));
        }
    }
    
    F make_variable_init_declaration(const SV& sv){
        auto name = std::any_cast<std::string>(sv[1]);
        auto var_decl = make_variable_declaration(sv, name, sv[0], sv[2]);
        
        if (sv.size() == 4){
            return F::make_binary(std::move(var_decl), AND, 
                F::make_binary(
                    F::make_lvar(UNTYPED, LVar<allocator_type>(name.data))),
                    EQ, 
                    f(sv[3])
                );
        }
        else{
            return std::move(var_decl);
        }
    }

    F make_variable_declaration(const SV& sv, const std::string& name, const std::any& typeVar, const std::any& annots){
        try{
            auto ty = std::any_cast<So>(typeVar);
            return make_existential(sv, ty, name, annots);
        }
        catch(std::bad_any_cast){
            auto typeValue = f(typeVar);
            auto inConstraint = F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(name.data())), IN, typeValue);
            auto sort = typeValue.sort();
            if (!sort.has_value() || !sort->is_set()){
                return make_error(sv, "We only type-value of variables to be of type Set.");
            }
            auto exists = make_existential(sv, *(sort->sub), name, annots);
            return F::make_binary(std::move(exists), AND, std::move(inConstraint));
        }
    }

    F make_existential(const SV& sv, const So& ty, const std::string& name, const std::any& sv_annots){
        auto f = F::make_exists(UNTYPED, LVar<allocator_type>(name.data()), ty);
        auto annots = std::any_cast<SV>(sv_annots);
        return update_with_annotations(sv, f, annots);
    }

    F update_with_annotations(const SV& sv, F formula, const SV& annots){
        for(int i = 0; i < annots.size(); ++i){
            auto annot = std::any_cast<SV>(annots[i]);
            auto name = std::any_cast<std::string>(annot[0]);
            if (name == "abstract"){
                AType ty = f(annot[1]).z(); // assignment of logic_int (int64_t) to ty (int) truncates ty (bug?)
                formula.type_as(ty);
            }
            else if (name == "is_defined_var"){}
            else if (name == "defines_var") {}
            else if (name == "var_is_introduced"){}
            else if (name == "output_var" && formula.is(F::E)){
                output.add_var(battery::get<0>(formula.exists()));
            }
            else if (name == "output_array" && formula.is(F::E)){
                auto array_name = std::any_cast<std::string>(sv[2]);
                auto dims = std::any_cast<SV>(annot[1]);
                output.add_array_var(array_name, battery::get<0>(formula.exists()), dims);
            }
            else{
                if (!ignored_annotations.contains(name)){
                    ignored_annotations.insert(name);
                    std::cerr << "% WARNING: ANnotation " + name + " is unknown and was ignored." << std::endl;
                }
            }
        }
        return std::move(formula);
    }



    /*
    from onnx file information
    */   
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

        std::cout << "ok" << std::endl;


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
                    std::cout << data[i] << std::endl;
                }
            }
        }

        return result;
    }

    layers get_layer_dim(const std::string onnx_filename){
        layers result;

        return result;
    }

    void test(){
        std::ifstream input("test_tiny.onnx", std::ios::ate | std::ios::binary); // open file and move current position in file to the end

        std::streamsize size = input.tellg(); // get current position in file
        input.seekg(0, std::ios::beg); // move to start of file

        std::vector<char> buffer(size);
        input.read(buffer.data(), size); // read raw data

        onnx::ModelProto model;
        model.ParseFromArray(buffer.data(), size); // parse protobuf

        auto graph = model.graph();

        // std::cout << "graph inputs:\n";
        // print_io_info(graph.input());

        // std::cout << "graph outputs:\n";
        // print_io_info(graph.output());
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