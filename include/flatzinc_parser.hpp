// Copyright 2022 Pierre Talbot

#ifndef FLATZINC_PARSER_HPP
#define FLATZINC_PARSER_HPP

#include "peglib.h"
#include <cassert>
#include <cstdlib>
#include <string>
#include <istream>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <cfenv>

#include "logic/ast.hpp"
#include "shared_ptr.hpp"

namespace lala {
  namespace impl {
    /** Unfortunately, I'm really not sure this function works in all cases due to compiler bugs with rounding modes... */
    inline logic_real string_to_real(const std::string& s) {
      #ifndef __GNUC__
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

    template<class F>
    void update_with_annotation(F& f, const peg::SemanticValues& annots) {
      using Allocator = typename F::allocator_type;
      for(int i = 0; i < annots.size(); ++i) {
        auto name = std::any_cast<LVar<Allocator>>(annots[i]);
        if(name == "abstract") {
          ++i;
          AType ty = std::any_cast<F>(annots[i]).z();
          f.type_as(ty);
        }
        else if(name == "under") { f.approx_as(UNDER); }
        else if(name == "exact") { f.approx_as(EXACT); }
        else if(name == "over") { f.approx_as(OVER); }
        else if(name == "is_defined_var") {}
        else if(name == "var_is_introduced") {}
        else if(name == "output_var") {}
        else {
          std::cerr << "Annotation " << name.data() << " is unknown." << std::endl;
        }
      }
    }

    template<class F>
    F make_binary(Sig sig, const peg::SemanticValues &vs) {
      if(vs.size() != 3) {
        std::cerr << "The symbol `" << string_of_sig(sig) << "` must be of binary arity, but "
          << (vs.size()-1) << " parameters were passed." << std::endl;
        return F();
      }
      return F::make_binary(std::any_cast<F>(vs[1]), sig, std::any_cast<F>(vs[2]));
    }

    template<class F>
    F make_unary_fun_eq(Sig sig, const peg::SemanticValues &vs, Sig eq_kind = EQ) {
      if(vs.size() != 3) {
        std::cerr << "The symbol `" << string_of_sig(sig) << "` must be of unary arity, but "
          << (vs.size()-2) << " parameters were passed." << std::endl;
        return F();
      }
      auto fun = F::make_unary(sig, std::any_cast<F>(vs[1]));
      return F::make_binary(fun, eq_kind, std::any_cast<F>(vs[2]));
    }

    template<class F>
    F make_binary_fun_eq(Sig sig, const peg::SemanticValues &vs, Sig eq_kind = EQ) {
      if(vs.size() != 4) {
        std::cerr << "The symbol `" << string_of_sig(sig) << "` must be of binary arity, but "
          << (vs.size()-2) << " parameters were passed." << std::endl;
        return F();
      }
      auto fun = F::make_binary(std::any_cast<F>(vs[1]), sig, std::any_cast<F>(vs[2]));
      return F::make_binary(fun, eq_kind, std::any_cast<F>(vs[3]));
    }

    template <class F>
    F make_float_in(const peg::SemanticValues &vs) {
      return F::make_binary(
          F::make_binary(std::any_cast<F>(vs[1]), GEQ, std::any_cast<F>(vs[2])),
          AND,
          F::make_binary(std::any_cast<F>(vs[1]), LEQ, std::any_cast<F>(vs[3])));
    }

    template <class F>
    F make_log(int base, const peg::SemanticValues &vs) {
      return F::make_binary(std::any_cast<F>(vs[1]), LOG, F::make_z(base));
    }
  }

  /** We parse the constraint language FlatZinc as described in the documentation: https://www.minizinc.org/doc-2.4.1/en/fzn-spec.html#specification-of-flatzinc.
   * We also extend FlatZinc conservatively for the purposes of our framework:

      - Add the type alias `real` (same as `float`).
      - Add the predicates `int_ge`, `int_gt` mainly to simplify testing in lala_core.
      - Add the functions `int_minus`, `float_minus`, `int_neg`, `float_neg`.
  */
  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc_str(const std::string& input) {

    using F = TFormula<Allocator>;

    peg::parser parser(R"(
        Statements  <- (VariableDecl / ConstraintDecl)+

        Literal     <- Boolean / Real / Integer / VariableLit

        VariableLit <- Identifier
        Identifier  <- < [a-zA-Z_][a-zA-Z0-9_]* >
        Boolean     <- < 'true' / 'false' >
        Real        <- < (
             'inf'
           / '-inf'
           / [+-]? [0-9]+ (('.' [0-9]*) / ([Ee][+-]?[0-9]+)) ) >
        Integer     <- < [+-]? [0-9]+ >

        VariableDecl <- 'var' Type ':' Identifier Annotations ';'

        IntType <- 'int'
        RealType <- 'float' / 'real'
        BoolType <- 'bool'
        SetType <- 'set' 'of' Type
        Type <- IntType / RealType / BoolType / SetType

        Annotations <- ('::' Identifier ('(' Literal ')')?)*

        ConstraintDecl <- 'constraint' PredicateCall Annotations ';'

        PredicateCall <- Identifier '(' Literal (',' Literal)* ')'

        %whitespace <- [ \n\r\t]*
    )");

    assert(static_cast<bool>(parser) == true);

    parser["Integer"] = [](const peg::SemanticValues &vs) {
      return F::make_z(vs.token_to_number<logic_int>());
    };

    parser["Real"] = [](const peg::SemanticValues &vs) {
      return F::make_real(impl::string_to_real(vs.token_to_string()));
    };

    parser["Boolean"] = [](const peg::SemanticValues &vs) {
      return F::make_z(vs.token_to_string() == "true" ? 1 : 0);
    };

    parser["Identifier"] = [](const peg::SemanticValues &vs) {
      return LVar<Allocator>(vs.token_to_string().c_str());
    };

    parser["VariableLit"] = [](const peg::SemanticValues &vs) {
      return F::make_lvar(UNTYPED, std::any_cast<LVar<Allocator>>(vs[0]));
    };

    parser["IntType"] = [](const peg::SemanticValues &vs) {
      return CType<Allocator>(CType<Allocator>::Int);
    };

    parser["RealType"] = [](const peg::SemanticValues &vs) {
      return CType<Allocator>(CType<Allocator>::Real);
    };

    parser["BoolType"] = [](const peg::SemanticValues &vs) {
      return CType<Allocator>(CType<Allocator>::Bool);
    };

    parser["SetType"] = [](const peg::SemanticValues &vs) {
      CType<Allocator> sub_ty = std::any_cast<CType<Allocator>>(vs[0]);
      return CType<Allocator>(CType<Allocator>::Set, std::move(sub_ty));
    };

    parser["Annotations"] = [](const peg::SemanticValues &vs) {
      return vs;
    };

    parser["VariableDecl"] = [](const peg::SemanticValues &vs) {
      auto ty = std::any_cast<CType<Allocator>>(vs[0]);
      auto f = F::make_exists(UNTYPED,
        std::any_cast<LVar<Allocator>>(vs[1]),
        ty,
        ty.default_approx());
      auto annots = std::any_cast<peg::SemanticValues>(vs[2]);
      impl::update_with_annotation(f, annots);
      return f;
    };

    parser["ConstraintDecl"] = [](const peg::SemanticValues &vs) {
      auto f = std::any_cast<F>(vs[0]);
      auto annots = std::any_cast<peg::SemanticValues>(vs[1]);
      impl::update_with_annotation(f, annots);
      return f;
    };

    parser["PredicateCall"] = [](const peg::SemanticValues &vs) {
      auto name = std::any_cast<LVar<Allocator>>(vs[0]);
      using namespace impl;
      if(name == "int_le") { return make_binary<F>(LEQ, vs); }
      else if(name == "int_lt") { return make_binary<F>(LT, vs); }
      else if(name == "int_ge") { return make_binary<F>(GEQ, vs); }
      else if(name == "int_gt") { return make_binary<F>(GT, vs); }
      else if(name == "int_eq") { return make_binary<F>(EQ, vs); }
      else if(name == "int_ne") { return make_binary<F>(NEQ, vs); }
      else if(name == "int_abs") { return make_unary_fun_eq<F>(ABS, vs); }
      else if(name == "int_neg") { return make_unary_fun_eq<F>(NEG, vs); }
      else if(name == "int_div") { return make_binary_fun_eq<F>(EDIV, vs); }
      else if(name == "int_mod") { return make_binary_fun_eq<F>(EMOD, vs); }
      else if(name == "int_plus") { return make_binary_fun_eq<F>(ADD, vs); }
      else if(name == "int_minus") { return make_binary_fun_eq<F>(SUB, vs); }
      else if(name == "int_pow") { return make_binary_fun_eq<F>(POW, vs); }
      else if(name == "int_times") { return make_binary_fun_eq<F>(MUL, vs); }
      else if(name == "int_max") { return make_binary_fun_eq<F>(MAX, vs); }
      else if(name == "int_min") { return make_binary_fun_eq<F>(MIN, vs); }
      else if(name == "int_eq_reif") { return make_binary_fun_eq<F>(EQ, vs, EQUIV); }
      else if(name == "int_le_reif") { return make_binary_fun_eq<F>(LEQ, vs, EQUIV); }
      else if(name == "int_lt_reif") { return make_binary_fun_eq<F>(LT, vs, EQUIV); }
      else if(name == "int_ne_reif") { return make_binary_fun_eq<F>(NEQ, vs, EQUIV); }
      else if(name == "bool2int") { return make_binary<F>(EQ, vs); }
      else if(name == "bool_and") { return make_binary_fun_eq<F>(AND, vs, EQUIV); }
      else if(name == "bool_eq") { return make_binary<F>(EQ, vs); }
      else if(name == "bool_le") { return make_binary<F>(LEQ, vs); }
      else if(name == "bool_lt") { return make_binary<F>(LT, vs); }
      else if(name == "bool_eq_reif") { return make_binary_fun_eq<F>(EQ, vs, EQUIV); }
      else if(name == "bool_le_reif") { return make_binary_fun_eq<F>(LEQ, vs, EQUIV); }
      else if(name == "bool_lt_reif") { return make_binary_fun_eq<F>(LT, vs, EQUIV); }
      else if(name == "bool_not") { return make_binary<F>(NEQ, vs); }
      else if(name == "bool_or") { return make_binary_fun_eq<F>(OR, vs, EQUIV); }
      else if(name == "bool_xor") {
        if(vs.size() == 3) { return make_binary<F>(XOR, vs); }
        else { return make_binary_fun_eq<F>(XOR, vs, EQUIV); }
      }
      else if(name == "set_card") { return make_unary_fun_eq<F>(CARD, vs); }
      else if(name == "set_diff") { return make_binary_fun_eq<F>(DIFFERENCE, vs); }
      else if(name == "set_eq") { return make_binary<F>(EQ, vs); }
      else if(name == "set_eq_reif") { return make_binary_fun_eq<F>(EQ, vs, EQUIV); }
      else if(name == "set_in") { return make_binary<F>(IN, vs); }
      else if(name == "set_in_reif") { return make_binary_fun_eq<F>(IN, vs, EQUIV); }
      else if(name == "set_intersect") { return make_binary_fun_eq<F>(INTERSECTION, vs, EQUIV); }
      else if(name == "set_union") { return make_binary_fun_eq<F>(UNION, vs, EQUIV); }
      else if(name == "set_ne") { return make_binary<F>(NEQ, vs); }
      else if(name == "set_ne_reif") { return make_binary_fun_eq<F>(NEQ, vs, EQUIV); }
      else if(name == "set_subset") { return make_binary<F>(SUBSETEQ, vs); }
      else if(name == "set_subset_reif") { return make_binary_fun_eq<F>(SUBSETEQ, vs, EQUIV); }
      else if(name == "set_superset") { return make_binary<F>(SUPSETEQ, vs); }
      else if(name == "set_symdiff") { return make_binary_fun_eq<F>(SYMMETRIC_DIFFERENCE, vs, EQUIV); }
      else if(name == "set_le") { return make_binary<F>(LEQ, vs); }
      else if(name == "set_le_reif") { return make_binary_fun_eq<F>(LEQ, vs, EQUIV); }
      else if(name == "set_lt") { return make_binary<F>(LT, vs); }
      else if(name == "set_lt_reif") { return make_binary_fun_eq<F>(LT, vs, EQUIV); }
      else if(name == "float_abs") { return make_binary_fun_eq<F>(ABS, vs); }
      else if(name == "float_neg") { return make_binary_fun_eq<F>(NEG, vs); }
      else if(name == "float_plus") { return make_binary_fun_eq<F>(ADD, vs); }
      else if(name == "float_minus") { return make_binary_fun_eq<F>(SUB, vs); }
      else if(name == "float_times") { return make_binary_fun_eq<F>(MUL, vs); }
      else if(name == "float_acos") { return make_unary_fun_eq<F>(ACOS, vs); }
      else if(name == "float_acosh") { return make_unary_fun_eq<F>(ACOSH, vs); }
      else if(name == "float_asin") { return make_unary_fun_eq<F>(ASIN, vs); }
      else if(name == "float_asinh") { return make_unary_fun_eq<F>(ASINH, vs); }
      else if(name == "float_atan") { return make_unary_fun_eq<F>(ATAN, vs); }
      else if(name == "float_atanh") { return make_unary_fun_eq<F>(ATANH, vs); }
      else if(name == "float_cos") { return make_unary_fun_eq<F>(COS, vs); }
      else if(name == "float_cosh") { return make_unary_fun_eq<F>(COSH, vs); }
      else if(name == "float_sin") { return make_unary_fun_eq<F>(SIN, vs); }
      else if(name == "float_sinh") { return make_unary_fun_eq<F>(SINH, vs); }
      else if(name == "float_tan") { return make_unary_fun_eq<F>(TAN, vs); }
      else if(name == "float_tanh") { return make_unary_fun_eq<F>(TANH, vs); }
      else if(name == "float_div") { return make_binary<F>(DIV, vs); }
      else if(name == "float_eq") { return make_binary<F>(EQ, vs); }
      else if(name == "float_eq_reif") { return make_binary_fun_eq<F>(EQ, vs, EQUIV); }
      else if(name == "float_le") { return make_binary<F>(LEQ, vs); }
      else if(name == "float_le_reif") { return make_binary_fun_eq<F>(LEQ, vs, EQUIV); }
      else if(name == "float_ne") { return make_binary<F>(NEQ, vs); }
      else if(name == "float_ne_reif") { return make_binary_fun_eq<F>(NEQ, vs, EQUIV); }
      else if(name == "float_lt") { return make_binary<F>(LT, vs); }
      else if(name == "float_lt_reif") { return make_binary_fun_eq<F>(LT, vs, EQUIV); }
      else if(name == "float_in") { return make_float_in<F>(vs); }
      else if(name == "float_in_reif") {
        return F::make_binary(make_float_in<F>(vs), EQUIV, std::any_cast<F>(vs[4]));
      }
      else if(name == "float_log10") { return make_log<F>(10, vs); }
      else if(name == "float_log2") { return make_log<F>(2, vs); }
      else if(name == "float_min") { return make_binary_fun_eq<F>(MIN, vs); }
      else if(name == "float_max") { return make_binary_fun_eq<F>(MAX, vs); }
      else if(name == "float_exp") { return make_unary_fun_eq<F>(EXP, vs); }
      else if(name == "float_ln") { return make_unary_fun_eq<F>(LN, vs); }
      else if(name == "float_pow") { return make_binary_fun_eq<F>(POW, vs); }
      else if(name == "float_sqrt") { return make_unary_fun_eq<F>(SQRT, vs); }
      else if(name == "int2float") { return make_binary<F>(EQ, vs); }
      else {
        std::cerr << "Predicate " << name.data() << " unsupported." << std::endl;
      }
      return F();
    };

    parser["Statements"] = [](const peg::SemanticValues &vs) {
      if(vs.size() == 1) return std::any_cast<F>(vs[0]);
      else {
        typename F::Sequence children;
        children.reserve(vs.size());
        for(int i = 0; i < vs.size(); ++i) {
          children.push_back(std::any_cast<F>(vs[i]));
        }
        return F::make_nary(AND, std::move(children));
      }
    };

    F f;
    parser.parse(input.c_str(), f);
    return battery::make_shared<TFormula<Allocator>, Allocator>(std::move(f));
  }

  template<class Allocator>
  battery::shared_ptr<TFormula<Allocator>, Allocator> parse_flatzinc(const std::string& filename) {
    std::ifstream t(filename);
    std::string input((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    return parse_flatzinc_str<Allocator>(input);
  }
}

#endif
