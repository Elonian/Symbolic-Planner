#ifndef SYMBOLIC_PLANNER_HEURISTICS_HPP
#define SYMBOLIC_PLANNER_HEURISTICS_HPP

#include "symbolic_planner/grounding.hpp"
#include "symbolic_planner/state.hpp"

#include <string>
#include <vector>

class Heuristic
{
public:
    virtual ~Heuristic() = default;
    virtual std::string name() const = 0;
    virtual int estimate(const State &state, const Env &env) const = 0;
};

class ZeroHeuristic : public Heuristic
{
public:
    std::string name() const override;
    int estimate(const State &state, const Env &env) const override;
};

class UnsatisfiedGoalHeuristic : public Heuristic
{
public:
    std::string name() const override;
    int estimate(const State &state, const Env &env) const override;
};

class DeleteRelaxationHeuristic : public Heuristic
{
public:
    enum class Variant
    {
        HMax,
        HAdd,
        RelaxedPlan
    };

    explicit DeleteRelaxationHeuristic(Variant variant = Variant::RelaxedPlan);

    std::string name() const override;
    int estimate(const State &state, const Env &env) const override;

private:
    Variant variant;
    mutable const Env *cached_env = nullptr;
    mutable std::vector<GroundedOperator> cached_operators;

    const std::vector<GroundedOperator> &operators_for(const Env &env) const;
};

#endif
