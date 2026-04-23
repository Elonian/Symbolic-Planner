#ifndef SYMBOLIC_PLANNER_BREADTH_FIRST_PLANNER_HPP
#define SYMBOLIC_PLANNER_BREADTH_FIRST_PLANNER_HPP

#include "symbolic_planner/search.hpp"

class BreadthFirstPlanner : public PlannerBase
{
public:
    std::string name() const override;
    std::list<GroundedAction> plan(const Env &env, SearchStats *stats = nullptr) const override;
};

#endif
