#ifndef SYMBOLIC_PLANNER_PARSER_HPP
#define SYMBOLIC_PLANNER_PARSER_HPP

#include "symbolic_planner/types.hpp"

#include <string>
#include <vector>

std::vector<std::string> parse_symbols(const std::string &symbols_str);
Env *create_env(const char *filename);
Env *create_env(char *filename);

#endif
