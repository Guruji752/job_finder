from langgraph.graph import StateGraph, END
from app.services.graph.stategraph import JobSearchState
from app.services.graph.nodes import retrieve_jobs, retrieve_profile, filter_jobs, rank_jobs

graph = StateGraph(JobSearchState)
print("=== Adding Graph Nodes ====")
graph.add_node('retrieve_profile',retrieve_profile)
graph.add_node('retrieve_jobs',retrieve_jobs)
graph.add_node('filter_jobs',filter_jobs)
graph.add_node('rank_jobs',rank_jobs)

### EDGE ###
graph.add_edge('retrieve_profile','retrieve_jobs')
graph.add_edge('retrieve_jobs','filter_jobs')
# graph.add_edge('filter_jobs','rank_jobs')
# graph.add_edge('rank_jobs',END)
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

    # No jobs cleared filtering → retry the search, but only until the cap,
    # otherwise the graph would loop forever. Past the cap, give up gracefully
    # and rank whatever we have.
    if len(filtered) == 0:
        if retry_count > 5:
            return 'rank_jobs'
        return 'retrieve_jobs'

    # Enough jobs → proceed to gap analysis.
    return 'rank_jobs'

graph.add_conditional_edges(
    'filter_jobs',
    should_retry,
    {
        'retrieve_profile':'retrieve_profile',
        'retrieve_jobs':'retrieve_jobs',
        'rank_jobs':'rank_jobs'
    }
)

graph.add_edge('rank_jobs',END)

graph_app = graph.compile()



