#ifndef SYMBOLIC_PLANNER_TRANSITION_HPP
#define SYMBOLIC_PLANNER_TRANSITION_HPP

#include "symbolic_planner/grounding.hpp"
#include "symbolic_planner/state.hpp"

bool operator_applicable(const State &state, const GroundedOperator &op);
State apply_operator(const State &state, const GroundedOperator &op);

#endif
