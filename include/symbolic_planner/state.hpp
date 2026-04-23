#ifndef SYMBOLIC_PLANNER_STATE_HPP
#define SYMBOLIC_PLANNER_STATE_HPP

#include "symbolic_planner/types.hpp"

#include <string>
#include <vector>

using State = GroundedConditionSet;

bool condition_holds(const State &state, const GroundedCondition &condition);
bool goals_satisfied(const State &state, const GroundedConditionSet &goals);
int count_unsatisfied_goals(const State &state, const GroundedConditionSet &goals);
State apply_effects(const State &state, const std::vector<GroundedCondition> &effects);
std::string state_key(const State &state);

#endif
