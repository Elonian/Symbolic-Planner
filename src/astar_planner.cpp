#include "symbolic_planner/planners/astar_planner.hpp"

#include "symbolic_planner/grounding.hpp"
#include "symbolic_planner/state.hpp"
#include "symbolic_planner/transition.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>

namespace
{
struct SearchNode
{
    State state;
    std::string key;
    int parent = -1;
    int g = 0;
    GroundedAction action;
};

struct OpenItem
{
    int f = 0;
    int g = 0;
    int index = 0;
    size_t sequence = 0;
};

struct OpenItemGreater
{
    bool operator()(const OpenItem &lhs, const OpenItem &rhs) const
    {
        if (lhs.f != rhs.f)
            return lhs.f > rhs.f;
        if (lhs.g != rhs.g)
            return lhs.g < rhs.g;
        return lhs.sequence > rhs.sequence;
    }
};

std::list<GroundedAction> reconstruct_plan(const std::vector<SearchNode> &nodes, int goal_index)
{
    std::list<GroundedAction> plan;
    for (int index = goal_index; index >= 0 && nodes[index].parent >= 0; index = nodes[index].parent)
        plan.push_front(nodes[index].action);
    return plan;
}

int weighted_priority(int g, int h, int weight)
{
    const int limit = std::numeric_limits<int>::max() / 4;
    if (h >= limit || h > (limit - g) / weight)
        return limit;
    return g + weight * h;
}
} // namespace

AStarPlanner::AStarPlanner(std::shared_ptr<Heuristic> heuristic, int heuristic_weight)
    : heuristic(std::move(heuristic)), heuristic_weight(std::max(1, heuristic_weight))
{
}

std::string AStarPlanner::name() const
{
    if (heuristic_weight == 1)
        return "A Star Search [" + heuristic->name() + "]";
    return "Weighted A Star Search [w=" + std::to_string(heuristic_weight) + ", " + heuristic->name() + "]";
}

std::list<GroundedAction> AStarPlanner::plan(const Env &env, SearchStats *stats) const
{
    SearchStats local_stats;
    if (!stats)
        stats = &local_stats;
    *stats = SearchStats();
    stats->planner_name = name();
    ScopedTimer<> timer(&stats->elapsed_ms);

    std::vector<GroundedOperator> operators = generate_grounded_operators(env);
    stats->grounded_actions = operators.size();

    std::vector<SearchNode> nodes;
    std::priority_queue<OpenItem, std::vector<OpenItem>, OpenItemGreater> open;
    std::unordered_map<std::string, int> best_g;
    size_t sequence = 0;

    SearchNode start;
    start.state = env.get_initial_conditions();
    start.key = state_key(start.state);
    nodes.push_back(start);
    best_g[start.key] = 0;
    open.push({weighted_priority(0, heuristic->estimate(start.state, env), heuristic_weight), 0, 0, sequence++});

    while (!open.empty())
    {
        OpenItem item = open.top();
        open.pop();

        const std::string current_key = nodes[item.index].key;
        const int current_g = nodes[item.index].g;
        auto best_it = best_g.find(current_key);
        if (best_it != best_g.end() && current_g != best_it->second)
            continue;

        if (goals_satisfied(nodes[item.index].state, env.get_goal_conditions()))
        {
            std::list<GroundedAction> plan = reconstruct_plan(nodes, item.index);
            stats->plan_length = plan.size();
            stats->solved = true;
            return plan;
        }

        ++stats->expanded_states;
        State current_state = nodes[item.index].state;

        for (const GroundedOperator &op : operators)
        {
            if (!operator_applicable(current_state, op))
                continue;

            State next_state = apply_operator(current_state, op);
            std::string next_key = state_key(next_state);
            int next_g = current_g + 1;

            auto previous = best_g.find(next_key);
            if (previous != best_g.end() && previous->second <= next_g)
                continue;

            SearchNode child;
            child.state = std::move(next_state);
            child.key = next_key;
            child.parent = item.index;
            child.g = next_g;
            child.action = op.action;

            int child_index = static_cast<int>(nodes.size());
            int h = heuristic->estimate(child.state, env);
            nodes.push_back(std::move(child));
            best_g[next_key] = next_g;
            open.push({weighted_priority(next_g, h, heuristic_weight), next_g, child_index, sequence++});
            ++stats->generated_states;
        }
    }

    return {};
}
