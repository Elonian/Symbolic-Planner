#ifndef SYMBOLIC_PLANNER_TYPES_HPP
#define SYMBOLIC_PLANNER_TYPES_HPP

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class GroundedCondition
{
    std::string predicate;
    std::vector<std::string> arg_values;
    bool truth = true;

public:
    GroundedCondition() = default;

    GroundedCondition(const std::string &predicate,
                      const std::vector<std::string> &arg_values,
                      bool truth = true)
        : predicate(predicate), arg_values(arg_values), truth(truth)
    {
    }

    GroundedCondition(const std::string &predicate,
                      const std::list<std::string> &arg_values,
                      bool truth = true)
        : predicate(predicate), arg_values(arg_values.begin(), arg_values.end()), truth(truth)
    {
    }

    const std::string &get_predicate() const
    {
        return predicate;
    }

    std::list<std::string> get_arg_values() const
    {
        return std::list<std::string>(arg_values.begin(), arg_values.end());
    }

    const std::vector<std::string> &get_arg_vector() const
    {
        return arg_values;
    }

    bool get_truth() const
    {
        return truth;
    }

    GroundedCondition positive() const
    {
        return GroundedCondition(predicate, arg_values, true);
    }

    bool same_atom(const GroundedCondition &rhs) const
    {
        return predicate == rhs.predicate && arg_values == rhs.arg_values;
    }

    bool operator==(const GroundedCondition &rhs) const
    {
        return truth == rhs.truth && same_atom(rhs);
    }

    std::string toString() const
    {
        std::string temp;
        if (!truth)
            temp += "!";
        temp += predicate;
        temp += "(";
        for (size_t i = 0; i < arg_values.size(); ++i)
        {
            if (i > 0)
                temp += ",";
            temp += arg_values[i];
        }
        temp += ")";
        return temp;
    }
};

inline std::ostream &operator<<(std::ostream &os, const GroundedCondition &pred)
{
    os << pred.toString() << " ";
    return os;
}

struct GroundedConditionComparator
{
    bool operator()(const GroundedCondition &lhs, const GroundedCondition &rhs) const
    {
        return lhs == rhs;
    }
};

struct GroundedConditionHasher
{
    size_t operator()(const GroundedCondition &gcond) const
    {
        size_t seed = std::hash<std::string>{}(gcond.get_predicate());
        seed ^= std::hash<bool>{}(gcond.get_truth()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for (const std::string &arg : gcond.get_arg_vector())
        {
            seed ^= std::hash<std::string>{}(arg) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using GroundedConditionSet =
    std::unordered_set<GroundedCondition, GroundedConditionHasher, GroundedConditionComparator>;

class Condition
{
    std::string predicate;
    std::vector<std::string> args;
    bool truth = true;

public:
    Condition() = default;

    Condition(const std::string &pred, const std::vector<std::string> &args, bool truth)
        : predicate(pred), args(args), truth(truth)
    {
    }

    Condition(const std::string &pred, const std::list<std::string> &args, bool truth)
        : predicate(pred), args(args.begin(), args.end()), truth(truth)
    {
    }

    const std::string &get_predicate() const
    {
        return predicate;
    }

    std::list<std::string> get_args() const
    {
        return std::list<std::string>(args.begin(), args.end());
    }

    const std::vector<std::string> &get_arg_vector() const
    {
        return args;
    }

    bool get_truth() const
    {
        return truth;
    }

    bool operator==(const Condition &rhs) const
    {
        return truth == rhs.truth && predicate == rhs.predicate && args == rhs.args;
    }

    std::string toString() const
    {
        std::string temp;
        if (!truth)
            temp += "!";
        temp += predicate;
        temp += "(";
        for (size_t i = 0; i < args.size(); ++i)
        {
            if (i > 0)
                temp += ",";
            temp += args[i];
        }
        temp += ")";
        return temp;
    }
};

inline std::ostream &operator<<(std::ostream &os, const Condition &cond)
{
    os << cond.toString() << " ";
    return os;
}

class Action
{
    std::string name;
    std::vector<std::string> args;
    std::vector<Condition> preconditions;
    std::vector<Condition> effects;

public:
    Action() = default;

    Action(const std::string &name,
           const std::vector<std::string> &args,
           const std::vector<Condition> &preconditions,
           const std::vector<Condition> &effects)
        : name(name), args(args), preconditions(preconditions), effects(effects)
    {
    }

    const std::string &get_name() const
    {
        return name;
    }

    std::list<std::string> get_args() const
    {
        return std::list<std::string>(args.begin(), args.end());
    }

    const std::vector<std::string> &get_arg_vector() const
    {
        return args;
    }

    const std::vector<Condition> &get_preconditions() const
    {
        return preconditions;
    }

    const std::vector<Condition> &get_effects() const
    {
        return effects;
    }

    std::string toString() const
    {
        std::string temp;
        temp += name;
        temp += "(";
        for (size_t i = 0; i < args.size(); ++i)
        {
            if (i > 0)
                temp += ",";
            temp += args[i];
        }
        temp += ")";
        return temp;
    }
};

inline std::ostream &operator<<(std::ostream &os, const Action &ac)
{
    os << ac.toString() << std::endl;
    os << "Precondition: ";
    for (const Condition &precond : ac.get_preconditions())
        os << precond;
    os << std::endl;
    os << "Effect: ";
    for (const Condition &effect : ac.get_effects())
        os << effect;
    os << std::endl;
    return os;
}

class Env
{
    GroundedConditionSet initial_conditions;
    GroundedConditionSet goal_conditions;
    std::vector<Action> actions;
    std::unordered_set<std::string> symbols;

public:
    void remove_initial_condition(const GroundedCondition &gc)
    {
        initial_conditions.erase(gc.positive());
    }

    void add_initial_condition(const GroundedCondition &gc)
    {
        if (gc.get_truth())
            initial_conditions.insert(gc);
        else
            remove_initial_condition(gc);
    }

    void add_goal_condition(const GroundedCondition &gc)
    {
        if (gc.get_truth())
            goal_conditions.insert(gc);
        else
            remove_goal_condition(gc);
    }

    void remove_goal_condition(const GroundedCondition &gc)
    {
        goal_conditions.erase(gc.positive());
    }

    void add_symbol(const std::string &symbol)
    {
        if (!symbol.empty())
            symbols.insert(symbol);
    }

    void add_symbols(const std::vector<std::string> &new_symbols)
    {
        for (const std::string &symbol : new_symbols)
            add_symbol(symbol);
    }

    void add_action(const Action &action)
    {
        actions.push_back(action);
    }

    Action get_action(const std::string &name) const
    {
        for (const Action &action : actions)
        {
            if (action.get_name() == name)
                return action;
        }
        throw std::runtime_error("Action " + name + " not found!");
    }

    std::unordered_set<std::string> get_symbols() const
    {
        return symbols;
    }

    std::vector<std::string> get_sorted_symbols() const
    {
        std::vector<std::string> sorted(symbols.begin(), symbols.end());
        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }

    const GroundedConditionSet &get_initial_conditions() const
    {
        return initial_conditions;
    }

    const GroundedConditionSet &get_goal_conditions() const
    {
        return goal_conditions;
    }

    const std::vector<Action> &get_actions() const
    {
        return actions;
    }
};

inline std::ostream &operator<<(std::ostream &os, const Env &w)
{
    os << "***** Environment *****" << std::endl
       << std::endl;
    os << "Symbols: ";
    for (const std::string &s : w.get_sorted_symbols())
        os << s + ",";
    os << std::endl;
    os << "Initial conditions: ";
    std::vector<std::string> initial;
    for (const GroundedCondition &s : w.get_initial_conditions())
        initial.push_back(s.toString());
    std::sort(initial.begin(), initial.end());
    for (const std::string &s : initial)
        os << s << " ";
    os << std::endl;
    os << "Goal conditions: ";
    std::vector<std::string> goals;
    for (const GroundedCondition &g : w.get_goal_conditions())
        goals.push_back(g.toString());
    std::sort(goals.begin(), goals.end());
    for (const std::string &g : goals)
        os << g << " ";
    os << std::endl;
    os << "Actions:" << std::endl;
    for (const Action &g : w.get_actions())
        os << g << std::endl;
    os << "***** Environment Created! *****" << std::endl;
    return os;
}

class GroundedAction
{
    std::string name;
    std::vector<std::string> arg_values;

public:
    GroundedAction() = default;

    GroundedAction(const std::string &name, const std::vector<std::string> &arg_values)
        : name(name), arg_values(arg_values)
    {
    }

    GroundedAction(const std::string &name, std::initializer_list<std::string> arg_values)
        : name(name), arg_values(arg_values)
    {
    }

    GroundedAction(const std::string &name, const std::list<std::string> &arg_values)
        : name(name), arg_values(arg_values.begin(), arg_values.end())
    {
    }

    const std::string &get_name() const
    {
        return name;
    }

    std::list<std::string> get_arg_values() const
    {
        return std::list<std::string>(arg_values.begin(), arg_values.end());
    }

    const std::vector<std::string> &get_arg_vector() const
    {
        return arg_values;
    }

    bool operator==(const GroundedAction &rhs) const
    {
        return name == rhs.name && arg_values == rhs.arg_values;
    }

    std::string toString() const
    {
        std::string temp;
        temp += name;
        temp += "(";
        for (size_t i = 0; i < arg_values.size(); ++i)
        {
            if (i > 0)
                temp += ",";
            temp += arg_values[i];
        }
        temp += ")";
        return temp;
    }
};

inline std::ostream &operator<<(std::ostream &os, const GroundedAction &gac)
{
    os << gac.toString() << " ";
    return os;
}

#endif
