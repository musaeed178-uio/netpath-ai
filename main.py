"""
main.py - NetPath AI Entry Point
Orchestrates the Pygame event loop, graph state, and algorithm execution.

Controls:
  N       – Add Node mode
  E       – Add Edge mode
  S       – Set Start mode
  G       – Set Goal mode
  D       – Delete Node mode
  C       – Toggle Congestion on an edge (click near midpoint)
  L       – Run Hill Climbing load balancer
  R       – Run selected algorithm(s)
  Space   – Step through animation
  →/←     – Next/Prev step
  Enter   – Finish step-through, show final path
  Esc     – Clear results / cancel
  Ctrl+Z  – Clear entire graph
  P       – Load a preset demo network
"""

import sys
import pygame
import math
import time

from algorithms import (
    Graph, SearchResult,
    uniform_cost_search, a_star_search,
    hill_climbing_load_balance,
    HEURISTIC_ADMISSIBLE, HEURISTIC_NONADMISSIBLE,
)
from gui import (
    Renderer, Button,
    COL_ACCENT, COL_WHITE, COL_TEXT_LO, COL_SUCCESS,
    COL_ERROR, COL_WARNING, BG_PANEL, COL_NODE_START, COL_NODE_GOAL,
)

# ---------------------------------------------------------------------------
# Window / Layout constants
# ---------------------------------------------------------------------------
WIN_W, WIN_H = 1200, 750
FPS          = 60
STEP_DELAY   = 0.12   # seconds between auto-play steps


# ---------------------------------------------------------------------------
# Preset demo network
# ---------------------------------------------------------------------------
def load_preset(graph: Graph):
    graph.clear()
    # Nodes  (x, y) within canvas width ~940
    coords = [
        (100, 350), (220, 150), (220, 550), (400, 100),
        (400, 350), (400, 600), (600, 200), (600, 500),
        (780, 350), (900, 150), (900, 550),
    ]
    for i, (x, y) in enumerate(coords):
        graph.add_node(x, y, label=f"R{i}")

    edges = [
        (0, 1,  8), (0, 2,  5),
        (1, 3,  6), (1, 4, 10),
        (2, 4,  7), (2, 5,  9),
        (3, 6,  4), (4, 6,  8), (4, 7, 12),
        (5, 7,  6),
        (6, 8,  5), (7, 8,  7),
        (8, 9,  9), (8,10, 11),
        (9,10, 20),
    ]
    for u, v, w in edges:
        graph.add_edge(u, v, w)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("NetPath AI – Heuristic Traffic Routing Simulator")
    clock  = pygame.time.Clock()

    # Fonts (fall back to system default if google fonts unavailable)
    try:
        font_sm = pygame.font.SysFont("Segoe UI", 13)
        font_md = pygame.font.SysFont("Segoe UI", 16, bold=True)
        font_lg = pygame.font.SysFont("Segoe UI", 22, bold=True)
    except Exception:
        font_sm = pygame.font.SysFont(None, 14)
        font_md = pygame.font.SysFont(None, 18)
        font_lg = pygame.font.SysFont(None, 24)

    renderer = Renderer(screen, font_sm, font_md, font_lg)

    # ------------------------------------------------------------------
    # Application State
    # ------------------------------------------------------------------
    graph   = Graph()
    START   = None   # node id
    GOAL    = None   # node id
    mode    = "ADD_NODE"    # current interaction mode

    result  : SearchResult | None = None
    result2 : SearchResult | None = None
    algo_label  = ""
    algo2_label = ""

    step            = -1   # -1 = showing final result; >= 0 = step-through index
    auto_play       = False
    last_step_time  = 0.0

    pending_edge_src: int | None = None   # first node selected for edge
    alert: str | None = None
    alert_timer: float = 0.0

    # Selected algorithm
    ALGOS = ["UCS", "A* Admissible", "A* Non-Admissible", "Comparison"]
    algo_idx = 0

    # Heuristic
    heuristic_mode = HEURISTIC_ADMISSIBLE

    # ------------------------------------------------------------------
    # Build right-panel buttons
    # ------------------------------------------------------------------
    PX = WIN_W - Renderer.PANEL_W + 10
    BW = Renderer.PANEL_W - 20
    BH = 28

    def make_btn(y, label, color=COL_ACCENT):
        return Button((PX, y, BW, BH), label, color=color)

    # Mode buttons
    btn_add_node  = make_btn(60,  "📍 Add Node  [N]")
    btn_add_edge  = make_btn(95,  "🔗 Add Edge  [E]")
    btn_set_start = make_btn(130, "🟢 Set Start [S]", color=(40, 140, 80))
    btn_set_goal  = make_btn(165, "🔴 Set Goal  [G]", color=(160, 40, 40))
    btn_delete    = make_btn(200, "🗑 Delete Node [D]", color=(100, 30, 30))
    btn_congestion= make_btn(235, "⚡ Congestion [C]", color=(150, 80, 20))
    btn_balancer  = make_btn(270, "⚖ Load Balance [L]", color=(60, 80, 150))
    btn_preset    = make_btn(305, "📋 Load Preset [P]", color=(60, 80, 60))
    btn_clear     = make_btn(340, "🗑 Clear Graph [Ctrl+Z]", color=(90, 20, 20))

    # Algorithm selector buttons
    btn_algo = [make_btn(390 + i*32, ALGOS[i], color=(30, 50, 100))
                for i in range(len(ALGOS))]
    btn_algo[algo_idx].active = True

    btn_run      = make_btn(528, "▶  Run Algorithm [R]", color=(30, 110, 60))
    btn_step     = make_btn(562, "⏭  Step Mode [Space]", color=(50, 50, 100))
    btn_prev     = make_btn(596, "◀  Prev Step [←]",     color=(50, 50, 100))
    btn_next     = make_btn(630, "▶  Next Step [→]",     color=(50, 50, 100))
    btn_finish   = make_btn(664, "✔  Show Path [Enter]",  color=(30, 110, 60))

    all_buttons = [
        btn_add_node, btn_add_edge, btn_set_start, btn_set_goal,
        btn_delete, btn_congestion, btn_balancer, btn_preset, btn_clear,
        *btn_algo,
        btn_run, btn_step, btn_prev, btn_next, btn_finish,
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def set_alert(msg: str, duration: float = 3.0):
        nonlocal alert, alert_timer
        alert = msg
        alert_timer = time.time() + duration

    def clear_results():
        nonlocal result, result2, step, auto_play
        result = result2 = None
        step   = -1
        auto_play = False

    def set_mode(new_mode: str):
        nonlocal mode, pending_edge_src
        mode = new_mode
        pending_edge_src = None

    def select_algo(idx: int):
        nonlocal algo_idx, heuristic_mode
        algo_idx = idx
        for i, b in enumerate(btn_algo):
            b.active = (i == idx)
        heuristic_mode = HEURISTIC_ADMISSIBLE

    def run_algorithms():
        nonlocal result, result2, step, algo_label, algo2_label, heuristic_mode
        clear_results()
        if START is None or GOAL is None:
            set_alert("Set START and GOAL nodes first!")
            return
        if START not in graph.nodes or GOAL not in graph.nodes:
            set_alert("START or GOAL node was deleted!")
            return

        choice = ALGOS[algo_idx]
        if choice == "UCS":
            result     = uniform_cost_search(graph, START, GOAL)
            algo_label = "UCS"
            result2    = None
        elif choice == "A* Admissible":
            result     = a_star_search(graph, START, GOAL, HEURISTIC_ADMISSIBLE)
            algo_label = "A* (Admissible)"
            result2    = None
        elif choice == "A* Non-Admissible":
            result     = a_star_search(graph, START, GOAL, HEURISTIC_NONADMISSIBLE)
            algo_label = "A* (Non-Admissible)"
            result2    = None
        else:  # Comparison
            result      = uniform_cost_search(graph, START, GOAL)
            result2     = a_star_search(graph, START, GOAL, HEURISTIC_ADMISSIBLE)
            algo_label  = "UCS"
            algo2_label = "A* (Admissible)"

        if result and not result.found:
            set_alert("⚠ Network Timeout: No path found!")

    def advance_step(delta: int):
        nonlocal step
        if result is None or not result.history:
            return
        step = max(0, min(step + delta, len(result.history) - 1))

    def finish_step():
        """Exit step-through, display final path."""
        nonlocal step, auto_play
        step      = -1
        auto_play = False

    def find_edge_near(pos, threshold=18) -> tuple | None:
        """Return (u, v) of edge whose midpoint is nearest pos within threshold."""
        x, y = pos
        best_dist = threshold
        best_edge = None
        for u, v, w in graph.all_edges():
            nu = graph.nodes[u]; nv = graph.nodes[v]
            mx = (nu.x + nv.x) / 2; my = (nu.y + nv.y) / 2
            d  = math.hypot(x - mx, y - my)
            if d < best_dist:
                best_dist = d
                best_edge = (u, v)
        return best_edge

    # ------------------------------------------------------------------
    # Load preset on startup
    # ------------------------------------------------------------------
    load_preset(graph)

    # ------------------------------------------------------------------
    # Event Loop
    # ------------------------------------------------------------------
    running = True
    while running:
        dt = clock.tick(FPS)

        # Alert timeout
        if alert and time.time() > alert_timer:
            alert = None

        # Auto-play step-through
        if auto_play and result and step >= 0:
            if time.time() - last_step_time > STEP_DELAY:
                if step < len(result.history) - 1:
                    step += 1
                    last_step_time = time.time()
                else:
                    auto_play = False

        # Hover update
        mpos = pygame.mouse.get_pos()
        for btn in all_buttons:
            btn.update_hover(mpos)

        # ------ Events ------------------------------------------------
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            # ---- Mouse click ----------------------------------------
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos

                # Panel button clicks
                if not renderer.canvas_contains(pos):
                    if btn_add_node.is_clicked(pos):   set_mode("ADD_NODE")
                    elif btn_add_edge.is_clicked(pos): set_mode("ADD_EDGE")
                    elif btn_set_start.is_clicked(pos):set_mode("SET_START")
                    elif btn_set_goal.is_clicked(pos): set_mode("SET_GOAL")
                    elif btn_delete.is_clicked(pos):   set_mode("DELETE")
                    elif btn_congestion.is_clicked(pos):set_mode("CONGESTION")
                    elif btn_balancer.is_clicked(pos):
                        changes = hill_climbing_load_balance(graph)
                        clear_results()
                        if changes:
                            set_alert(f"⚖ Load Balanced {len(changes)} link(s)")
                        else:
                            set_alert("No congested links found.")
                    elif btn_preset.is_clicked(pos):
                        load_preset(graph)
                        clear_results()
                        START = None; GOAL = None
                    elif btn_clear.is_clicked(pos):
                        graph.clear(); clear_results()
                        START = None; GOAL = None
                    elif btn_run.is_clicked(pos):   run_algorithms()
                    elif btn_step.is_clicked(pos):
                        if result and result.history:
                            step = 0
                            auto_play = not auto_play
                            last_step_time = time.time()
                    elif btn_prev.is_clicked(pos):  advance_step(-1)
                    elif btn_next.is_clicked(pos):  advance_step(+1)
                    elif btn_finish.is_clicked(pos):finish_step()
                    else:
                        for i, b in enumerate(btn_algo):
                            if b.is_clicked(pos):
                                select_algo(i); break

                # Canvas clicks
                else:
                    clicked_node = renderer.find_node_at(pos, graph)

                    if mode == "ADD_NODE":
                        graph.add_node(pos[0], pos[1])
                        clear_results()

                    elif mode == "ADD_EDGE":
                        if clicked_node is not None:
                            if pending_edge_src is None:
                                pending_edge_src = clicked_node
                            else:
                                if pending_edge_src != clicked_node:
                                    w = renderer.prompt_weight(
                                        f"Weight for edge "
                                        f"{graph.nodes[pending_edge_src].label}"
                                        f" → {graph.nodes[clicked_node].label}:"
                                    )
                                    if w is not None:
                                        graph.add_edge(pending_edge_src, clicked_node, w)
                                        clear_results()
                                pending_edge_src = None

                    elif mode == "SET_START":
                        if clicked_node is not None:
                            START = clicked_node
                            clear_results()

                    elif mode == "SET_GOAL":
                        if clicked_node is not None:
                            GOAL = clicked_node
                            clear_results()

                    elif mode == "DELETE":
                        if clicked_node is not None:
                            if clicked_node == START: START = None
                            if clicked_node == GOAL:  GOAL  = None
                            graph.remove_node(clicked_node)
                            clear_results()

                    elif mode == "CONGESTION":
                        edge = find_edge_near(pos)
                        if edge:
                            u, v = edge
                            old_w = graph.get_edge_weight(u, v) or 1
                            new_w = old_w * 2
                            graph.update_edge_weight(u, v, new_w)
                            clear_results()
                            set_alert(
                                f"⚡ Congestion: {graph.nodes[u].label}↔"
                                f"{graph.nodes[v].label}  "
                                f"{old_w:.0f} → {new_w:.0f}"
                            )

            # ---- Keyboard -------------------------------------------
            elif event.type == pygame.KEYDOWN:
                ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

                if event.key == pygame.K_n: set_mode("ADD_NODE")
                elif event.key == pygame.K_e: set_mode("ADD_EDGE")
                elif event.key == pygame.K_s: set_mode("SET_START")
                elif event.key == pygame.K_g: set_mode("SET_GOAL")
                elif event.key == pygame.K_d: set_mode("DELETE")
                elif event.key == pygame.K_c: set_mode("CONGESTION")
                elif event.key == pygame.K_l:
                    changes = hill_climbing_load_balance(graph)
                    clear_results()
                    set_alert(f"⚖ Load Balanced {len(changes)} link(s)" if changes
                              else "No congested links found.")
                elif event.key == pygame.K_p:
                    load_preset(graph); clear_results()
                    START = None; GOAL = None
                elif event.key == pygame.K_z and ctrl:
                    graph.clear(); clear_results()
                    START = None; GOAL = None
                elif event.key == pygame.K_r:
                    run_algorithms()
                elif event.key == pygame.K_SPACE:
                    if result and result.history:
                        if step < 0: step = 0
                        auto_play = not auto_play
                        last_step_time = time.time()
                elif event.key == pygame.K_RIGHT:  advance_step(+1)
                elif event.key == pygame.K_LEFT:   advance_step(-1)
                elif event.key == pygame.K_RETURN: finish_step()
                elif event.key == pygame.K_ESCAPE:
                    clear_results()
                    pending_edge_src = None

        # ------ Compose state dict for renderer ----------------------
        state = {
            "graph":         graph,
            "start":         START,
            "goal":          GOAL,
            "mode":          mode,
            "result":        result,
            "result2":       result2,
            "algo_label":    algo_label,
            "algo2_label":   algo2_label,
            "step":          step,
            "pending_edge_src": pending_edge_src,
            "alert":         alert,
            "buttons":       all_buttons,
        }

        renderer.draw_frame(state)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()