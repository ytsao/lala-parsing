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

    void add_array_var(const std::string& name, const bstring& var_name, const peg::SemanticValues& sv){
        int idx = -1;
        auto array_name = bstring(name.data());
        for (int i = 0; i < output_arrays.size(); ++i){
            if (battery::get<0>(output_arrays[i]) == array_name){
                idx = i;
                break;
            }
        }
        if (idx == -1){
            output_arrays.push_back(battery::make_tuple<bstring, array_dim_t, bvector<bstring>>(bstring(array_name), {}, {}));
            idx = static_cast<int>(output_arrays.size()) - 1;
            // Add the dimension of the array.
            for (int i = 0; i < sv.size(); ++i){
                auto range = std::any_cast<F>(sv[i]);
                for (int j = 0; j < range.s().size(); ++j){
                    const auto& itv = range.s()[j];
                    battery::get<1>(output_arrays[idx]).push_back(battery::make_tuple(battery::get<0>(itv).z(), battery::get<1>(itv).z()));
                }
            }
        }
        battery::get<2>(output_arrays[idx]).push_back(var_name);
    }

    void add_var(const bstring& var_name){
        output_vars.push_back(var_name);
    }
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

public: 
    SMTParser(SMTOutput<Allocator>& output): error(false), silent(false), output(output){}

    battery::shared_ptr<F, allocator_type> parse(const std::string& input){
        //TODO: Missing implementation about post-condition contains 'and' & 'or' operators
        peg::parser parser(R"(
            Statements <- ( '(' VariableDeclaration ')' / '(' BoundConstr ')' / '(' PropertyConstr ')' / Comment)+

            VariableDeclaration <- 'declare-const' Identifier Type
            VariableName <- Identifier
            Identifier <- < [a-zA-Z][a-zA-Z0-9_]* >
            Type <- RealType
            RealType <- 'Real'
            Constant <- < (
                'inf'
                / '-inf'
                / [+-]? [0-9]+ (('.' (&'..' / !'.') [0-9]*) / ([Ee][+-]?[0-9]+)) ) >

            BoundConstr <- 'assert' '(' Operators VariableName Constant ')' 
            PropertyConstr <- 'assert' '(' Operators VariableName VariableName ')'
            ConstraintDeclaration <- 'assert' '(' Constraint ')'
            Constraint <- (Operators VariableName (Constant / VariableName))+

            Operators <- ('<=' / '>=' / '>' / '<' / 'and' / 'or' / '(' / ')' )

            ~Comment    <- ';' [^\n\r]* [ \n\r\t]*
            %whitespace <- [ \n\r\t]*
        )");

        assert(static_cast<bool>(parser) == true);

        parser["Constant"] =[](const SV& sv){return F::make_real(string_to_real(sv.token_to_string()));};
        parser["BoundConstr"] = [this](const SV& sv) {return make_bound_constraint_declaration(sv);};
        parser["VariableDeclaration"] = [this](const SV& sv){return make_variable_init_declaration(sv);}; 
        parser["PropertyConstr"] = [this](const SV& sv){return make_property_constraint_declaration(sv);};
        parser["Constraint"] = [](const SV& sv){return F();};
        parser["VariableName"] = [](const SV& sv){ return F::make_lvar(UNTYPED, LVar<Allocator>(std::any_cast<std::string>(sv[0])));};
        parser["Identifier"] = [](const SV& sv){return sv.token_to_string();};
        parser["RealType"] = [](const SV& sv){return So(So::Real);};
        parser["Operators"] = [](const SV& sv){return sv.token_to_string();};
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

    F make_arity_error(const SV& sv, Sig sig, int expected, int obtained){
        return make_error(sv, "Thy symbol `" + std::string(string_of_sig(sig)) +
            "` expects `" + std::to_string(expected) + "` parameters" +
            ", but we got `" + std::to_string(obtained) + "` parameters.");
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
        // sv.size () == 2
        // sv[0] = variable name
        // sv[1] = type
        auto name = std::any_cast<std::string>(sv[0]);
        auto var_decl = make_variable_declaration(sv, name, sv[1]);
        return std::move(var_decl);
    }

    F make_variable_declaration(const SV& sv, const std::string& name, const std::any& typeVar){
        try{
            auto ty = std::any_cast<So>(typeVar);
            return make_existential(sv, ty, name);
        }
        catch(std::bad_any_cast){
            auto typeValue = f(typeVar);
            auto inConstraint = F::make_binary(F::make_lvar(UNTYPED, LVar<allocator_type>(name.data())), IN, typeValue);
            auto sort = typeValue.sort();
            if (!sort.has_value() || !sort->is_set()){
                return make_error(sv, "We only type-value of variables to be of type Set.");
            }
            auto exists = make_existential(sv, *(sort->sub), name);
            return F::make_binary(std::move(exists), AND, std::move(inConstraint));
        }
    }

    F make_existential(const SV& sv, const So& ty, const std::string& name){
        auto f = F::make_exists(UNTYPED, LVar<allocator_type>(name.data()), ty);
        if (f.is(F::E)){
            output.add_var(battery::get<0>(f.exists()));
        }
        return std::move(f);
    }

    F make_bound_constraint_declaration(const SV& sv){
        std::string sign = std::any_cast<std::string>(sv[0]);
        std::string variable_name = f(sv[1]).lv().data();
        if (sign == "<="){
            return make_linear_constraint(variable_name, LEQ, sv);
        }
        else if (sign == ">="){
            return make_linear_constraint(variable_name, GEQ, sv);
        }
        else{
            return make_error(sv, "Unknown sign `" + sign + "`.");
        }
    }

    F make_property_constraint_declaration(const SV& sv){
        std::string sign = std::any_cast<std::string>(sv[0]);
        std::string variable_name = f(sv[1]).lv().data();
        if (sign == "<="){
            return make_linear_constraint(variable_name, LEQ, sv);
        }
        else if (sign == ">="){
            return make_linear_constraint(variable_name, GEQ, sv);
        }
        else{
            return make_error(sv, "Unknown sign `" + sign + "`.");
        }
        //TODO: Add disjunctive property which is 'or' & 'and' operator in the formula
    }

    F make_linear_constraint(const std::string& name, Sig sig, const SV& sv){
        if (sv.size() != 3){
            return make_error(sv, "`" + name + "` expects 2 parameters, but we got `" + std::to_string(sv.size() - 1) + "` parameters.");
        }

        FSeq lhs;
        auto variable = f(sv[1]);
        if (variable.is(F::LV)){
            lhs.push_back(f(sv[1])); 
        }
        
        std::cout << sv.sv() << std::endl;
        auto rhs = f(sv[2]);
        F constraint = 
            lhs.size() == 1 
            ? F::make_binary(std::move(lhs[0]), sig, rhs)
            : F::make_binary(F::make_nary(ADD, std::move(lhs)), sig, rhs);
        
        return std::move(constraint);
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