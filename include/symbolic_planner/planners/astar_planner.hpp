#ifndef SYMBOLIC_PLANNER_ASTAR_PLANNER_HPP
#define SYMBOLIC_PLANNER_ASTAR_PLANNER_HPP

#include "symbolic_planner/heuristics.hpp"
#include "symbolic_planner/search.hpp"

#include <memory>

class AStarPlanner : public PlannerBase
{
    std::shared_ptr<Heuristic> heuristic;
    int heuristic_weight = 1;

public:
    explicit AStarPlanner(std::shared_ptr<Heuristic> heuristic, int heuristic_weight = 1);

    std::string name() const override;
    std::list<GroundedAction> plan(const Env &env, SearchStats *stats = nullptr) const override;
};

#endif
