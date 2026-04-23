#include "symbolic_planner/heuristics.hpp"
#include "symbolic_planner/parser.hpp"
#include "symbolic_planner/planners/astar_planner.hpp"
#include "symbolic_planner/planners/breadth_first_planner.hpp"
#include "symbolic_planner/planners/greedy_best_first_planner.hpp"
#include "symbolic_planner/search.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <utility>

#ifndef ENVS_DIR
#define ENVS_DIR "../envs"
#endif

bool print_status = true;

namespace
{
std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string env_string(const char *name, const std::string &default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    return lower_copy(value);
}

int env_int(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (!value)
        return default_value;
    int parsed = std::atoi(value);
    return parsed > 0 ? parsed : default_value;
}

void print_stats(const SearchStats &stats)
{
    std::cerr << "Planner: " << stats.planner_name
              << " | solved=" << (stats.solved ? "yes" : "no")
              << " | plan_length=" << stats.plan_length
              << " | expanded=" << stats.expanded_states
              << " | generated=" << stats.generated_states
              << " | grounded_actions=" << stats.grounded_actions
              << " | time_ms=" << stats.elapsed_ms
              << std::endl;
}

std::shared_ptr<Heuristic> make_heuristic(const std::string &name)
{
    if (name == "zero" || name == "blind")
        return std::make_shared<ZeroHeuristic>();
    if (name == "goals" || name == "goal" || name == "unsatisfied" || name == "unsatisfied-goals")
        return std::make_shared<UnsatisfiedGoalHeuristic>();
    if (name == "hmax" || name == "delete-hmax" || name == "max")
        return std::make_shared<DeleteRelaxationHeuristic>(DeleteRelaxationHeuristic::Variant::HMax);
    if (name == "hadd" || name == "delete-hadd" || name == "add")
        return std::make_shared<DeleteRelaxationHeuristic>(DeleteRelaxationHeuristic::Variant::HAdd);
    return std::make_shared<DeleteRelaxationHeuristic>(DeleteRelaxationHeuristic::Variant::RelaxedPlan);
}

std::shared_ptr<Heuristic> heuristic_for_mode(const std::string &mode)
{
    if (mode == "astar_goal" || mode == "astar_goals")
        return make_heuristic("goals");
    if (mode == "astar_hmax" || mode == "optimal")
        return make_heuristic("hmax");
    if (mode == "astar_hadd" || mode == "strong" || mode == "best")
        return make_heuristic("hadd");
    return make_heuristic(env_string("SYMBOLIC_PLANNER_HEURISTIC", "ff"));
}
} // namespace

std::list<GroundedAction> plan_with_breadth_first_search(Env *env, SearchStats *stats = nullptr)
{
    BreadthFirstPlanner planner;
    return planner.plan(*env, stats);
}

std::list<GroundedAction> plan_with_a_star_search(Env *env, SearchStats *stats = nullptr)
{
    AStarPlanner planner(std::make_shared<DeleteRelaxationHeuristic>(DeleteRelaxationHeuristic::Variant::RelaxedPlan));
    return planner.plan(*env, stats);
}

std::list<GroundedAction> plan_with_a_star_search(Env *env,
                                                  std::shared_ptr<Heuristic> heuristic,
                                                  int heuristic_weight,
                                                  SearchStats *stats = nullptr)
{
    AStarPlanner planner(std::move(heuristic), heuristic_weight);
    return planner.plan(*env, stats);
}

std::list<GroundedAction> plan_with_greedy_best_first_search(Env *env,
                                                             std::shared_ptr<Heuristic> heuristic,
                                                             SearchStats *stats = nullptr)
{
    GreedyBestFirstPlanner planner(std::move(heuristic));
    return planner.plan(*env, stats);
}

std::list<GroundedAction> planner(Env *env)
{
    SearchStats stats;
    std::list<GroundedAction> actions;
    std::string mode = env_string("SYMBOLIC_PLANNER_MODE", "bfs");

    if (mode == "astar" || mode == "a_star" || mode == "heuristic" ||
        mode == "astar_ff" || mode == "astar_goal" || mode == "astar_goals" ||
        mode == "astar_hmax" || mode == "optimal" || mode == "astar_hadd" ||
        mode == "strong" || mode == "best")
    {
        actions = plan_with_a_star_search(env, heuristic_for_mode(mode), 1, &stats);
    }
    else if (mode == "weighted" || mode == "weighted_astar" || mode == "weighted_ff" || mode == "wastar")
    {
        actions = plan_with_a_star_search(env, heuristic_for_mode(mode),
                                          env_int("SYMBOLIC_PLANNER_WEIGHT", 5), &stats);
    }
    else if (mode == "greedy" || mode == "gbfs" || mode == "greedy_ff")
    {
        actions = plan_with_greedy_best_first_search(env, heuristic_for_mode(mode), &stats);
    }
    else
    {
        actions = plan_with_breadth_first_search(env, &stats);
    }

    print_stats(stats);
    return actions;
}

int main(int argc, char *argv[])
{
    // DO NOT CHANGE THIS FUNCTION
    const char *env_file = "example.txt";
    if (argc > 1)
        env_file = argv[1];
    std::string envsDirPath = ENVS_DIR;
    char *filename = new char[envsDirPath.length() + strlen(env_file) + 2];
    strcpy(filename, envsDirPath.c_str());
    strcat(filename, "/");
    strcat(filename, env_file);

    std::cout << "Environment: " << filename << std::endl;
    Env *env = create_env(filename);
    if (print_status)
    {
        std::cout << *env;
    }

    std::list<GroundedAction> actions = planner(env);

    std::cout << "\nPlan: " << std::endl;
    for (const GroundedAction &gac : actions)
    {
        std::cout << gac << std::endl;
    }

    delete[] filename;
    delete env;
    return 0;
}
