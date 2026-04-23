#include "symbolic_planner/state.hpp"

#include <algorithm>

bool condition_holds(const State &state, const GroundedCondition &condition)
{
    GroundedCondition positive = condition.positive();
    bool present = state.find(positive) != state.end();
    return condition.get_truth() ? present : !present;
}

bool goals_satisfied(const State &state, const GroundedConditionSet &goals)
{
    for (const GroundedCondition &goal : goals)
    {
        if (!condition_holds(state, goal))
            return false;
    }
    return true;
}

int count_unsatisfied_goals(const State &state, const GroundedConditionSet &goals)
{
    int missing = 0;
    for (const GroundedCondition &goal : goals)
    {
        if (!condition_holds(state, goal))
            ++missing;
    }
    return missing;
}

State apply_effects(const State &state, const std::vector<GroundedCondition> &effects)
{
    State next = state;

    for (const GroundedCondition &effect : effects)
    {
        if (!effect.get_truth())
            next.erase(effect.positive());
    }

    for (const GroundedCondition &effect : effects)
    {
        if (effect.get_truth())
            next.insert(effect);
    }

    return next;
}

std::string state_key(const State &state)
{
    std::vector<std::string> facts;
    facts.reserve(state.size());
    for (const GroundedCondition &condition : state)
        facts.push_back(condition.toString());

    std::sort(facts.begin(), facts.end());

    std::string key;
    for (const std::string &fact : facts)
    {
        key += fact;
        key += ';';
    }
    return key;
}
