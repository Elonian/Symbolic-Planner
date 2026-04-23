#ifndef SYMBOLIC_PLANNER_SEARCH_HPP
#define SYMBOLIC_PLANNER_SEARCH_HPP

#include "symbolic_planner/types.hpp"

#include <chrono>
#include <list>
#include <string>
#include <vector>

struct SearchStats
{
    std::string planner_name;
    size_t expanded_states = 0;
    size_t generated_states = 0;
    size_t grounded_actions = 0;
    size_t plan_length = 0;
    double elapsed_ms = 0.0;
    bool solved = false;
};

class PlannerBase
{
public:
    virtual ~PlannerBase() = default;
    virtual std::string name() const = 0;
    virtual std::list<GroundedAction> plan(const Env &env, SearchStats *stats = nullptr) const = 0;
};

template <typename Clock = std::chrono::steady_clock>
class ScopedTimer
{
    typename Clock::time_point start;
    double *elapsed_ms;

public:
    explicit ScopedTimer(double *elapsed_ms)
        : start(Clock::now()), elapsed_ms(elapsed_ms)
    {
    }

    ~ScopedTimer()
    {
        if (elapsed_ms)
        {
            auto end = Clock::now();
            *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        }
    }
};

#endif
