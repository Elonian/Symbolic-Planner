#include "symbolic_planner/heuristics.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace
{
const int INF_HEURISTIC = std::numeric_limits<int>::max() / 4;

int saturated_add(int lhs, int rhs)
{
    if (lhs >= INF_HEURISTIC || rhs >= INF_HEURISTIC)
        return INF_HEURISTIC;
    if (lhs > INF_HEURISTIC - rhs)
        return INF_HEURISTIC;
    return lhs + rhs;
}

bool relaxed_condition_holds(const GroundedConditionSet &facts, const GroundedCondition &condition)
{
    if (!condition.get_truth())
        return true;
    return facts.find(condition.positive()) != facts.end();
}

bool relaxed_preconditions_hold(const GroundedConditionSet &facts, const GroundedOperator &op)
{
    for (const GroundedCondition &precondition : op.preconditions)
    {
        if (!relaxed_condition_holds(facts, precondition))
            return false;
    }
    return true;
}

bool relaxed_goals_hold(const GroundedConditionSet &facts, const GroundedConditionSet &goals)
{
    for (const GroundedCondition &goal : goals)
    {
        if (!relaxed_condition_holds(facts, goal))
            return false;
    }
    return true;
}

std::string fact_key(const GroundedCondition &condition)
{
    return condition.positive().toString();
}

bool action_adds_fact(const GroundedOperator &op, const GroundedCondition &fact)
{
    for (const GroundedCondition &effect : op.effects)
    {
        if (effect.get_truth() && effect.positive() == fact.positive())
            return true;
    }
    return false;
}

int relaxed_fact_cost(const GroundedCondition &condition,
                      const std::unordered_map<std::string, int> &costs)
{
    if (!condition.get_truth())
        return 0;

    auto it = costs.find(fact_key(condition));
    if (it == costs.end())
        return INF_HEURISTIC;
    return it->second;
}

int precondition_cost(const GroundedOperator &op,
                      const std::unordered_map<std::string, int> &costs,
                      DeleteRelaxationHeuristic::Variant variant)
{
    int result = 0;
    for (const GroundedCondition &precondition : op.preconditions)
    {
        int cost = relaxed_fact_cost(precondition, costs);
        if (cost >= INF_HEURISTIC)
            return INF_HEURISTIC;

        if (variant == DeleteRelaxationHeuristic::Variant::HMax)
            result = std::max(result, cost);
        else
            result = saturated_add(result, cost);
    }
    return result;
}

int delete_relaxed_cost(const State &state,
                        const Env &env,
                        const std::vector<GroundedOperator> &operators,
                        DeleteRelaxationHeuristic::Variant variant)
{
    std::unordered_map<std::string, int> costs;
    for (const GroundedCondition &fact : state)
        costs[fact_key(fact)] = 0;

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const GroundedOperator &op : operators)
        {
            int op_cost = precondition_cost(op, costs, variant);
            if (op_cost >= INF_HEURISTIC)
                continue;
            op_cost = saturated_add(op_cost, 1);

            for (const GroundedCondition &effect : op.effects)
            {
                if (!effect.get_truth())
                    continue;

                std::string key = fact_key(effect);
                auto previous = costs.find(key);
                if (previous == costs.end() || op_cost < previous->second)
                {
                    costs[key] = op_cost;
                    changed = true;
                }
            }
        }
    }

    int result = 0;
    for (const GroundedCondition &goal : env.get_goal_conditions())
    {
        int cost = relaxed_fact_cost(goal, costs);
        if (cost >= INF_HEURISTIC)
            return INF_HEURISTIC;

        if (variant == DeleteRelaxationHeuristic::Variant::HMax)
            result = std::max(result, cost);
        else
            result = saturated_add(result, cost);
    }
    return result;
}

int missing_precondition_count(const GroundedOperator &op, const GroundedConditionSet &facts)
{
    int missing = 0;
    for (const GroundedCondition &precondition : op.preconditions)
    {
        if (precondition.get_truth() && facts.find(precondition.positive()) == facts.end())
            ++missing;
    }
    return missing;
}

int relaxed_plan_cost(const State &state,
                      const Env &env,
                      const std::vector<GroundedOperator> &operators)
{
    if (goals_satisfied(state, env.get_goal_conditions()))
        return 0;

    std::vector<GroundedConditionSet> fact_layers;
    std::vector<std::vector<size_t>> action_layers;
    fact_layers.push_back(state);

    while (!relaxed_goals_hold(fact_layers.back(), env.get_goal_conditions()))
    {
        const GroundedConditionSet &current = fact_layers.back();
        std::vector<size_t> applicable_actions;
        GroundedConditionSet next = current;

        for (size_t i = 0; i < operators.size(); ++i)
        {
            const GroundedOperator &op = operators[i];
            if (!relaxed_preconditions_hold(current, op))
                continue;

            applicable_actions.push_back(i);
            for (const GroundedCondition &effect : op.effects)
            {
                if (effect.get_truth())
                    next.insert(effect.positive());
            }
        }

        if (next == current)
            return INF_HEURISTIC;

        action_layers.push_back(std::move(applicable_actions));
        fact_layers.push_back(std::move(next));
    }

    std::vector<GroundedConditionSet> needed(fact_layers.size());
    needed.back() = env.get_goal_conditions();
    std::unordered_set<std::string> selected_actions;

    for (int layer = static_cast<int>(fact_layers.size()) - 1; layer > 0; --layer)
    {
        std::vector<GroundedCondition> goals(needed[layer].begin(), needed[layer].end());
        std::sort(goals.begin(), goals.end(),
                  [](const GroundedCondition &lhs, const GroundedCondition &rhs) {
                      return lhs.toString() < rhs.toString();
                  });

        for (const GroundedCondition &goal : goals)
        {
            if (relaxed_condition_holds(fact_layers[layer - 1], goal))
                continue;

            int best_score = INF_HEURISTIC;
            size_t best_action = operators.size();

            for (size_t action_index : action_layers[layer - 1])
            {
                const GroundedOperator &op = operators[action_index];
                if (!action_adds_fact(op, goal))
                    continue;

                int score = missing_precondition_count(op, fact_layers[layer - 1]);
                if (best_action == operators.size() ||
                    score < best_score ||
                    (score == best_score && op.preconditions.size() < operators[best_action].preconditions.size()))
                {
                    best_score = score;
                    best_action = action_index;
                }
            }

            if (best_action == operators.size())
                return INF_HEURISTIC;

            const GroundedOperator &chosen = operators[best_action];
            selected_actions.insert(chosen.action.toString());
            for (const GroundedCondition &precondition : chosen.preconditions)
            {
                if (precondition.get_truth())
                    needed[layer - 1].insert(precondition.positive());
            }
        }
    }

    return static_cast<int>(selected_actions.size());
}
} // namespace

std::string ZeroHeuristic::name() const
{
    return "zero";
}

int ZeroHeuristic::estimate(const State &, const Env &) const
{
    return 0;
}

std::string UnsatisfiedGoalHeuristic::name() const
{
    return "unsatisfied-goals";
}

int UnsatisfiedGoalHeuristic::estimate(const State &state, const Env &env) const
{
    return count_unsatisfied_goals(state, env.get_goal_conditions());
}

DeleteRelaxationHeuristic::DeleteRelaxationHeuristic(Variant variant)
    : variant(variant)
{
}

std::string DeleteRelaxationHeuristic::name() const
{
    switch (variant)
    {
    case Variant::HMax:
        return "delete-relaxation-hmax";
    case Variant::HAdd:
        return "delete-relaxation-hadd";
    case Variant::RelaxedPlan:
        return "delete-relaxation-relaxed-plan";
    }
    return "delete-relaxation";
}

const std::vector<GroundedOperator> &DeleteRelaxationHeuristic::operators_for(const Env &env) const
{
    if (cached_env != &env)
    {
        cached_env = &env;
        cached_operators = generate_grounded_operators(env);
    }
    return cached_operators;
}

int DeleteRelaxationHeuristic::estimate(const State &state, const Env &env) const
{
    const std::vector<GroundedOperator> &operators = operators_for(env);
    if (variant == Variant::RelaxedPlan)
        return relaxed_plan_cost(state, env, operators);
    return delete_relaxed_cost(state, env, operators, variant);
}
