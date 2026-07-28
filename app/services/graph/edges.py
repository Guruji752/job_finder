from langgraph.graph import StateGraph, END
from app.services.graph.stategraph import JobSearchState
from app.services.graph.nodes import retrieve_jobs, retrieve_profile, filter_jobs, rank_jobs, refine_query

graph = StateGraph(JobSearchState)
print("=== Adding Graph Nodes ====")
graph.add_node('retrieve_profile',retrieve_profile)
graph.add_node('retrieve_jobs',retrieve_jobs)
graph.add_node('filter_jobs',filter_jobs)
graph.add_node('rank_jobs',rank_jobs)
graph.add_node('refine_query',refine_query)

### EDGE ###
graph.add_edge('retrieve_profile','retrieve_jobs')
graph.add_edge('retrieve_jobs','filter_jobs')
# retry loop: after the strategist rewrites the query, search again with it.
graph.add_edge('refine_query','retrieve_jobs')
graph.add_edge('rank_jobs',END)

graph.set_entry_point('retrieve_profile')
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


graph_app = graph.compile()



