#include "symbolic_planner/transition.hpp"

bool operator_applicable(const State &state, const GroundedOperator &op)
{
    for (const GroundedCondition &precondition : op.preconditions)
    {
        if (!condition_holds(state, precondition))
            return false;
    }
    return true;
}

State apply_operator(const State &state, const GroundedOperator &op)
{
    return apply_effects(state, op.effects);
}
