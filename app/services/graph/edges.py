from langgraph.graph import StateGraph, END
from app.services.graph.stategraph import JobSearchState
from app.services.graph.nodes import (
    retrieve_jobs, retrieve_profile, filter_jobs, rank_jobs, refine_query,
    supervisor, dummy_agent,
)

graph = StateGraph(JobSearchState)
print("=== Adding Graph Nodes ====")
graph.add_node('retrieve_profile',retrieve_profile)
graph.add_node('retrieve_jobs',retrieve_jobs)
graph.add_node('filter_jobs',filter_jobs)
graph.add_node('rank_jobs',rank_jobs)
graph.add_node('refine_query',refine_query)
graph.add_node('supervisor',supervisor)
graph.add_node('dummy_agent',dummy_agent)

### EDGE ###
graph.add_edge('retrieve_profile','retrieve_jobs')
graph.add_edge('retrieve_jobs','filter_jobs')
# retry loop: after the strategist rewrites the query, search again with it.
graph.add_edge('refine_query','retrieve_jobs')
# Search Agent hands control back to the Supervisor instead of ending directly.
graph.add_edge('rank_jobs','supervisor')
# Dummy Agent also hands back to the Supervisor so it can decide to stop.
graph.add_edge('dummy_agent','supervisor')

graph.set_entry_point('supervisor')
#### END ##


def should_retry(state: JobSearchState):
    print("====ENTERED IN SHOULD RETRY======")
    filtered = state['filtered_jobs']
    profile = state['profile']
    retry_count = state.get('retry_count', 0)

    # Safety net — profile is normally fetched once at the entry point.
    if not profile:
        return 'retrieve_profile'

    # No jobs cleared filtering → have the strategist rewrite the query and retry,
    # but only until the cap, otherwise the graph would loop forever. Past the cap,
    # give up gracefully and rank whatever we have.
    if len(filtered) == 0:
        if retry_count > 5:
            return 'rank_jobs'
        return 'refine_query'

    # Enough jobs → proceed to gap analysis.
    return 'rank_jobs'

graph.add_conditional_edges(
    'filter_jobs',
    should_retry,
    {
        'retrieve_profile':'retrieve_profile',
        'refine_query':'refine_query',
        'rank_jobs':'rank_jobs'
    }
)


def route_supervisor(state: JobSearchState):
    print("==== SUPERVISOR ROUTING ====")
    # Pure lookup now — the actual judgment already happened in the supervisor
    # node (an LLM call, or the trivial first-visit shortcut), and got stored
    # in state. This router just relays that decision.
    return state['supervisor_decision']

graph.add_conditional_edges(
    'supervisor',
    route_supervisor,
    {
        'retrieve_profile':'retrieve_profile',
        'dummy_agent':'dummy_agent',
        'end':END,
    }
)

graph_app = graph.compile()



