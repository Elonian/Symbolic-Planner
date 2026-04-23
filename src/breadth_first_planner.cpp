#include "symbolic_planner/planners/breadth_first_planner.hpp"

#include "symbolic_planner/grounding.hpp"
#include "symbolic_planner/state.hpp"
#include "symbolic_planner/transition.hpp"

#include <queue>
#include <unordered_map>

namespace
{
struct SearchNode
{
    State state;
    int parent = -1;
    GroundedAction action;
};

std::list<GroundedAction> reconstruct_plan(const std::vector<SearchNode> &nodes, int goal_index)
{
    std::list<GroundedAction> plan;
    for (int index = goal_index; index >= 0 && nodes[index].parent >= 0; index = nodes[index].parent)
        plan.push_front(nodes[index].action);
    return plan;
}
} // namespace

std::string BreadthFirstPlanner::name() const
{
    return "Breadth First Search";
}

std::list<GroundedAction> BreadthFirstPlanner::plan(const Env &env, SearchStats *stats) const
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
    std::queue<int> open;
    std::unordered_map<std::string, int> visited;

    SearchNode start;
    start.state = env.get_initial_conditions();
    nodes.push_back(start);
    open.push(0);
    visited[state_key(start.state)] = 0;

    if (goals_satisfied(start.state, env.get_goal_conditions()))
    {
        stats->solved = true;
        return {};
    }

    while (!open.empty())
    {
        int node_index = open.front();
        open.pop();
        State current_state = nodes[node_index].state;
        ++stats->expanded_states;

        for (const GroundedOperator &op : operators)
        {
            if (!operator_applicable(current_state, op))
                continue;

            State next_state = apply_operator(current_state, op);
            std::string key = state_key(next_state);
            if (visited.find(key) != visited.end())
                continue;

            SearchNode child;
            child.state = std::move(next_state);
            child.parent = node_index;
            child.action = op.action;

            int child_index = static_cast<int>(nodes.size());
            nodes.push_back(std::move(child));
            visited[key] = child_index;
            open.push(child_index);
            ++stats->generated_states;

            if (goals_satisfied(nodes[child_index].state, env.get_goal_conditions()))
            {
                std::list<GroundedAction> plan = reconstruct_plan(nodes, child_index);
                stats->plan_length = plan.size();
                stats->solved = true;
                return plan;
            }
        }
    }

    return {};
}
