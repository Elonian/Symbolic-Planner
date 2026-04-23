#ifndef SYMBOLIC_PLANNER_GREEDY_BEST_FIRST_PLANNER_HPP
#define SYMBOLIC_PLANNER_GREEDY_BEST_FIRST_PLANNER_HPP

#include "symbolic_planner/heuristics.hpp"
#include "symbolic_planner/search.hpp"

#include <memory>

class GreedyBestFirstPlanner : public PlannerBase
{
    std::shared_ptr<Heuristic> heuristic;

public:
    explicit GreedyBestFirstPlanner(std::shared_ptr<Heuristic> heuristic);

    std::string name() const override;
    std::list<GroundedAction> plan(const Env &env, SearchStats *stats = nullptr) const override;
};

#endif
