#include "symbolic_planner/grounding.hpp"

#include <algorithm>

namespace
{
std::vector<std::string> unique_variables(const Action &action)
{
    std::vector<std::string> variables;
    for (const std::string &arg : action.get_arg_vector())
    {
        if (std::find(variables.begin(), variables.end(), arg) == variables.end())
            variables.push_back(arg);
    }
    return variables;
}

std::string substitute_token(const std::string &token,
                             const std::unordered_map<std::string, std::string> &binding)
{
    auto it = binding.find(token);
    if (it == binding.end())
        return token;
    return it->second;
}

GroundedCondition ground_condition(const Condition &condition,
                                   const std::unordered_map<std::string, std::string> &binding)
{
    std::vector<std::string> values;
    for (const std::string &arg : condition.get_arg_vector())
        values.push_back(substitute_token(arg, binding));

    return GroundedCondition(condition.get_predicate(), values, condition.get_truth());
}

GroundedOperator ground_action(const Action &action,
                               const std::unordered_map<std::string, std::string> &binding)
{
    std::vector<std::string> action_values;
    for (const std::string &arg : action.get_arg_vector())
        action_values.push_back(substitute_token(arg, binding));

    GroundedOperator op;
    op.action = GroundedAction(action.get_name(), action_values);

    for (const Condition &condition : action.get_preconditions())
        op.preconditions.push_back(ground_condition(condition, binding));
    for (const Condition &effect : action.get_effects())
        op.effects.push_back(ground_condition(effect, binding));

    return op;
}

void enumerate_bindings(const Action &action,
                        const std::vector<std::string> &symbols,
                        const std::vector<std::string> &variables,
                        size_t depth,
                        std::unordered_map<std::string, std::string> &binding,
                        std::vector<GroundedOperator> &operators)
{
    if (depth == variables.size())
    {
        operators.push_back(ground_action(action, binding));
        return;
    }

    const std::string &variable = variables[depth];
    for (const std::string &symbol : symbols)
    {
        binding[variable] = symbol;
        enumerate_bindings(action, symbols, variables, depth + 1, binding, operators);
    }
    binding.erase(variable);
}
} // namespace

std::vector<GroundedOperator> generate_grounded_operators(const Env &env)
{
    std::vector<GroundedOperator> operators;
    std::vector<std::string> symbols = env.get_sorted_symbols();
    std::vector<Action> actions = env.get_actions();

    std::sort(actions.begin(), actions.end(),
              [](const Action &lhs, const Action &rhs) {
                  return lhs.toString() < rhs.toString();
              });

    for (const Action &action : actions)
    {
        std::vector<std::string> variables = unique_variables(action);
        std::unordered_map<std::string, std::string> binding;
        enumerate_bindings(action, symbols, variables, 0, binding, operators);
    }

    std::sort(operators.begin(), operators.end(),
              [](const GroundedOperator &lhs, const GroundedOperator &rhs) {
                  return lhs.toString() < rhs.toString();
              });

    return operators;
}
