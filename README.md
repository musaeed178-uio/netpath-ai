# NetPath AI – Heuristic Traffic Routing Simulator

NetPath AI is an interactive, visual simulation tool designed to explore and compare network traffic routing algorithms. Built with Python and Pygame, it allows users to construct custom network topologies, introduce congestion, and observe how different pathfinding strategies perform in real-time.

---

## 🚀 Features

- **Interactive Graph Editor:** Create, move, and manage network nodes and edges in real-time.
- **Pathfinding Algorithms:** Compare the performance of:
  - **UCS (Uniform Cost Search):** Guaranteed optimal path based on cost.
  - **A* (Admissible):** Optimal path using straight-line Euclidean distance as the heuristic.
  - **A* (Non-Admissible):** Faster, heuristic-driven search that may produce suboptimal paths.
- **Traffic Simulation:** Dynamically toggle edge congestion to observe how algorithms reroute traffic.
- **Load Balancing:** Built-in Hill Climbing algorithm to automatically detect congested links and penalize them, incentivizing traffic to spread across the network.
- **Step-through Visualization:** Analyze the search process step-by-step with intuitive animations showing the frontier, visited nodes, and final paths.
- **Dark Cyberpunk UI:** A responsive, high-contrast dashboard for monitoring search metrics and network state.

---

## 🛠 Prerequisites

Ensure you have Python 3 installed. The project depends on the `pygame` library.

```bash
pip install pygame
```

---

## 🖥 How to Run

1. Clone the repository.
2. Navigate to the project directory.
3. Launch the application:

```bash
python main.py
```

---

## 🎮 Controls

| Action | Key |
| :--- | :--- |
| **Add Node Mode** | `N` |
| **Add Edge Mode** | `E` |
| **Set Start Node** | `S` |
| **Set Goal Node** | `G` |
| **Delete Node** | `D` |
| **Toggle Congestion** | `C` |
| **Load Balance** | `L` |
| **Load Demo Network** | `P` |
| **Clear Graph** | `Ctrl + Z` |
| **Run Algorithm** | `R` |
| **Step / Auto-play** | `Space` |
| **Navigate Steps** | `←` / `→` |
| **Show Final Path** | `Enter` |
| **Cancel / Clear Results** | `Esc` |

---

## 📂 Project Structure

- `main.py`: Entry point, Pygame event loop, and UI state management.
- `algorithms.py`: Core search engine (UCS, A*, Hill Climbing) and graph data structures.
- `gui.py`: Rendering engine for nodes, edges, panels, and UI widgets.

---

## 💡 How it works

The simulator uses a graph-based representation where nodes act as routers/servers and weighted edges act as network links. Congestion is simulated by doubling the edge weight, which directly impacts the cost calculation in both UCS and A* algorithms. The load balancing tool uses a local search (Hill Climbing) to identify edges above a threshold and apply a penalty factor, helping demonstrate how networks mitigate bottlenecks.
