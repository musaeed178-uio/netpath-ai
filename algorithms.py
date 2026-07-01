"""
algorithms.py - NetPath AI Core Search Engine
Contains graph data structures, UCS, A* (admissible & non-admissible),
and Hill Climbing for load balancing.
"""

import heapq
import math
import time


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class Node:
    """Represents a router/server in the network."""
    def __init__(self, node_id: int, x: float, y: float, label: str = ""):
        self.id = node_id
        self.x = x
        self.y = y
        self.label = label or f"R{node_id}"

    def __repr__(self):
        return f"Node({self.label}, x={self.x:.0f}, y={self.y:.0f})"


class Graph:
    """Weighted undirected graph using an adjacency list."""

    def __init__(self):
        self.nodes: dict[int, Node] = {}          # node_id -> Node
        self.adj: dict[int, list[tuple]] = {}     # node_id -> [(neighbour_id, weight), ...]
        self._next_id = 0

    def add_node(self, x: float, y: float, label: str = "") -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(nid, x, y, label)
        self.adj[nid] = []
        return nid

    def add_edge(self, u: int, v: int, weight: float):
        """Add (or update) an undirected edge between u and v."""
        # Remove existing edge in both directions first
        self.adj[u] = [(nb, w) for nb, w in self.adj[u] if nb != v]
        self.adj[v] = [(nb, w) for nb, w in self.adj[v] if nb != u]
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))

    def update_edge_weight(self, u: int, v: int, new_weight: float):
        """Dynamically update an edge weight (simulate congestion)."""
        self.add_edge(u, v, new_weight)

    def get_edge_weight(self, u: int, v: int) -> float | None:
        for nb, w in self.adj.get(u, []):
            if nb == v:
                return w
        return None

    def remove_node(self, nid: int):
        if nid not in self.nodes:
            return
        del self.nodes[nid]
        del self.adj[nid]
        for uid in self.adj:
            self.adj[uid] = [(nb, w) for nb, w in self.adj[uid] if nb != nid]

    def clear(self):
        self.nodes.clear()
        self.adj.clear()
        self._next_id = 0

    def euclidean(self, u: int, v: int) -> float:
        a, b = self.nodes[u], self.nodes[v]
        return math.hypot(a.x - b.x, a.y - b.y)

    def all_edges(self):
        """Return each undirected edge once as (u, v, weight)."""
        seen = set()
        for u in self.adj:
            for v, w in self.adj[u]:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    yield u, v, w


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

HEURISTIC_ADMISSIBLE = "admissible"
HEURISTIC_NONADMISSIBLE = "nonadmissible"


def heuristic(graph: Graph, node: int, goal: int, mode: str) -> float:
    if node not in graph.nodes or goal not in graph.nodes:
        return 0.0
    d = graph.euclidean(node, goal)
    if mode == HEURISTIC_ADMISSIBLE:
        # Straight-line Euclidean distance — admissible (never overestimates)
        return d
    else:
        # Distance squared / scaled — non-admissible, leads to suboptimal paths
        return (d ** 2) / 50.0


# ---------------------------------------------------------------------------
# Search Result
# ---------------------------------------------------------------------------

class SearchResult:
    """Container for the output of a search run."""
    def __init__(self):
        self.path: list[int] = []           # Node IDs on the final path
        self.cost: float = float("inf")
        self.nodes_expanded: int = 0
        self.elapsed_ms: float = 0.0
        self.history: list[dict] = []       # Step-by-step animation data
        self.found: bool = False
        self.algorithm: str = ""
        self.heuristic_mode: str = ""

    def __repr__(self):
        return (f"SearchResult(found={self.found}, cost={self.cost:.2f}, "
                f"expanded={self.nodes_expanded}, time={self.elapsed_ms:.2f}ms)")


# ---------------------------------------------------------------------------
# Uniform Cost Search
# ---------------------------------------------------------------------------

def uniform_cost_search(graph: Graph, start: int, goal: int) -> SearchResult:
    """
    UCS: Expand cheapest-cost node first.
    Guaranteed optimal. May explore many nodes.
    """
    result = SearchResult()
    result.algorithm = "UCS"
    t0 = time.perf_counter()

    if start not in graph.nodes or goal not in graph.nodes:
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # Priority queue: (cost, node_id, path)
    pq = [(0.0, start, [start])]
    visited: dict[int, float] = {}   # node -> best g cost seen

    while pq:
        cost, node, path = heapq.heappop(pq)

        # Record step for animation
        frontier_ids = [item[1] for item in pq]
        result.history.append({
            "expanded": node,
            "visited": list(visited.keys()),
            "frontier": frontier_ids,
            "path_so_far": list(path),
            "cost_so_far": cost,
        })

        # Skip if we already found a cheaper route to this node
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
        result.nodes_expanded += 1

        if node == goal:
            result.found = True
            result.path = path
            result.cost = cost
            break

        for neighbour, weight in graph.adj.get(node, []):
            new_cost = cost + weight
            if neighbour not in visited or visited[neighbour] > new_cost:
                heapq.heappush(pq, (new_cost, neighbour, path + [neighbour]))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# A* Search
# ---------------------------------------------------------------------------

def a_star_search(
    graph: Graph,
    start: int,
    goal: int,
    heuristic_mode: str = HEURISTIC_ADMISSIBLE,
) -> SearchResult:
    """
    A*: f(n) = g(n) + h(n).
    Admissible heuristic -> optimal path.
    Non-admissible -> faster but possibly suboptimal.
    """
    result = SearchResult()
    result.algorithm = f"A* ({heuristic_mode})"
    result.heuristic_mode = heuristic_mode
    t0 = time.perf_counter()

    if start not in graph.nodes or goal not in graph.nodes:
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    h_start = heuristic(graph, start, goal, heuristic_mode)
    # Priority queue: (f, g, node_id, path)
    pq = [(h_start, 0.0, start, [start])]
    visited: dict[int, float] = {}

    while pq:
        f, g, node, path = heapq.heappop(pq)

        frontier_ids = [item[2] for item in pq]
        result.history.append({
            "expanded": node,
            "visited": list(visited.keys()),
            "frontier": frontier_ids,
            "path_so_far": list(path),
            "cost_so_far": g,
            "f": f,
        })

        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        result.nodes_expanded += 1

        if node == goal:
            result.found = True
            result.path = path
            result.cost = g
            break

        for neighbour, weight in graph.adj.get(node, []):
            new_g = g + weight
            if neighbour not in visited or visited[neighbour] > new_g:
                new_f = new_g + heuristic(graph, neighbour, goal, heuristic_mode)
                heapq.heappush(pq, (new_f, new_g, neighbour, path + [neighbour]))

    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Hill Climbing – Load Balancing
# ---------------------------------------------------------------------------

def hill_climbing_load_balance(
    graph: Graph,
    congestion_threshold: float = 15.0,
    penalty_factor: float = 1.5,
) -> list[tuple[int, int, float, float]]:
    """
    Local search to reduce congestion.
    Identifies edges whose weight exceeds `congestion_threshold` and
    'reroutes' traffic by increasing their weight (simulating router
    load-balancing: penalising overloaded links so future searches avoid them).

    Returns a list of (u, v, old_weight, new_weight) tuples for logging.
    """
    changes = []
    for u, v, w in list(graph.all_edges()):
        if w >= congestion_threshold:
            new_w = w * penalty_factor
            graph.update_edge_weight(u, v, new_w)
            changes.append((u, v, w, new_w))
    return changes
