#ifndef SYMBOLIC_PLANNER_GROUNDING_HPP
#define SYMBOLIC_PLANNER_GROUNDING_HPP

#include "symbolic_planner/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

struct GroundedOperator
{
    GroundedAction action;
    std::vector<GroundedCondition> preconditions;
    std::vector<GroundedCondition> effects;

    std::string toString() const
    {
        return action.toString();
    }
};

std::vector<GroundedOperator> generate_grounded_operators(const Env &env);

#endif
