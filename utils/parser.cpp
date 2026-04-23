#include "symbolic_planner/parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace
{
std::string remove_whitespace(std::string line)
{
    line.erase(std::remove_if(line.begin(), line.end(),
                              [](unsigned char c) { return std::isspace(c); }),
               line.end());
    return line;
}

std::vector<Condition> parse_conditions(const std::string &line, const std::regex &condition_regex)
{
    std::vector<Condition> conditions;
    auto begin = std::sregex_iterator(line.begin(), line.end(), condition_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it)
    {
        std::string predicate = (*it)[1].str();
        std::vector<std::string> args = parse_symbols((*it)[2].str());
        bool truth = true;

        if (!predicate.empty() && predicate[0] == '!')
        {
            predicate = predicate.substr(1);
            truth = false;
        }

        conditions.emplace_back(predicate, args, truth);
    }

    return conditions;
}

void add_grounded_conditions(Env *env,
                             const std::vector<Condition> &conditions,
                             bool goal_conditions)
{
    for (const Condition &condition : conditions)
    {
        GroundedCondition grounded(condition.get_predicate(),
                                   condition.get_arg_vector(),
                                   condition.get_truth());
        if (goal_conditions)
            env->add_goal_condition(grounded);
        else
            env->add_initial_condition(grounded);
    }
}
} // namespace

std::vector<std::string> parse_symbols(const std::string &symbols_str)
{
    std::vector<std::string> symbols;
    size_t start = 0;

    while (start <= symbols_str.size())
    {
        size_t comma = symbols_str.find(',', start);
        std::string symbol = symbols_str.substr(start, comma - start);
        if (!symbol.empty())
            symbols.push_back(symbol);

        if (comma == std::string::npos)
            break;
        start = comma + 1;
    }

    return symbols;
}

Env *create_env(char *filename)
{
    return create_env(static_cast<const char *>(filename));
}

Env *create_env(const char *filename)
{
    std::ifstream input_file(filename);
    if (!input_file.is_open())
        throw std::runtime_error(std::string("Unable to open environment file: ") + filename);

    Env *env = new Env();

    std::regex symbol_state_regex("symbols:", std::regex::icase);
    std::regex symbol_regex("([a-zA-Z0-9_,]+)");
    std::regex initial_condition_regex("initialconditions:(.*)", std::regex::icase);
    std::regex condition_regex("(!?[A-Z][a-zA-Z0-9_]*)\\(([a-zA-Z0-9_,]*)\\)");
    std::regex goal_condition_regex("goalconditions:(.*)", std::regex::icase);
    std::regex action_regex("actions:", std::regex::icase);
    std::regex action_definition_regex("([A-Z][a-zA-Z0-9_]*)\\(([a-zA-Z0-9_,]*)\\)");
    std::regex precondition_regex("preconditions:(.*)", std::regex::icase);
    std::regex effect_regex("effects:(.*)", std::regex::icase);

    enum ParserState
    {
        SYMBOLS,
        INITIAL,
        GOAL,
        ACTIONS,
        ACTION_DEFINITION,
        ACTION_PRECONDITION,
        ACTION_EFFECT
    };

    ParserState parser = SYMBOLS;
    std::vector<Condition> preconditions;
    std::vector<Condition> effects;
    std::string action_name;
    std::vector<std::string> action_args;

    std::string raw_line;
    while (std::getline(input_file, raw_line))
    {
        std::string line = remove_whitespace(raw_line);
        if (line.empty())
            continue;

        if (parser == SYMBOLS)
        {
            if (!std::regex_search(line, symbol_state_regex))
                throw std::runtime_error("Symbols are not specified correctly.");

            line = line.substr(8);
            std::smatch match;
            if (std::regex_search(line, match, symbol_regex))
                env->add_symbols(parse_symbols(match.str()));
            parser = INITIAL;
        }
        else if (parser == INITIAL)
        {
            if (!std::regex_match(line, initial_condition_regex))
                throw std::runtime_error("Initial conditions are not specified correctly.");

            add_grounded_conditions(env, parse_conditions(line, condition_regex), false);
            parser = GOAL;
        }
        else if (parser == GOAL)
        {
            if (!std::regex_match(line, goal_condition_regex))
                throw std::runtime_error("Goal conditions are not specified correctly.");

            add_grounded_conditions(env, parse_conditions(line, condition_regex), true);
            parser = ACTIONS;
        }
        else if (parser == ACTIONS)
        {
            if (!std::regex_match(line, action_regex))
                throw std::runtime_error("Actions are not specified correctly.");

            parser = ACTION_DEFINITION;
        }
        else if (parser == ACTION_DEFINITION)
        {
            std::smatch match;
            if (!std::regex_match(line, match, action_definition_regex))
                throw std::runtime_error("Action is not specified correctly: " + line);

            action_name = match[1].str();
            action_args = parse_symbols(match[2].str());
            parser = ACTION_PRECONDITION;
        }
        else if (parser == ACTION_PRECONDITION)
        {
            if (!std::regex_match(line, precondition_regex))
                throw std::runtime_error("Precondition is not specified correctly.");

            preconditions = parse_conditions(line, condition_regex);
            parser = ACTION_EFFECT;
        }
        else if (parser == ACTION_EFFECT)
        {
            if (!std::regex_match(line, effect_regex))
                throw std::runtime_error("Effects are not specified correctly.");

            effects = parse_conditions(line, condition_regex);
            env->add_action(Action(action_name, action_args, preconditions, effects));

            preconditions.clear();
            effects.clear();
            action_name.clear();
            action_args.clear();
            parser = ACTION_DEFINITION;
        }
    }

    return env;
}
