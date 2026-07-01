"""
gui.py - NetPath AI Pygame Rendering & Interaction Layer
Handles drawing the canvas, panels, user interactions, and animation.
"""

import pygame
import math
from algorithms import (
    Graph, SearchResult,
    HEURISTIC_ADMISSIBLE, HEURISTIC_NONADMISSIBLE,
)

# ---------------------------------------------------------------------------
# Colour Palette  (dark cyberpunk theme)
# ---------------------------------------------------------------------------
BG_CANVAS      = (10,  14,  30)
BG_PANEL       = (16,  20,  42)
PANEL_BORDER   = (50,  60, 120)

COL_NODE       = (60, 140, 240)      # default node
COL_NODE_START = (50, 220, 120)      # source
COL_NODE_GOAL  = (240, 70,  70)      # target
COL_NODE_FRONT = (255, 210,  50)     # frontier / open list
COL_NODE_VISIT = (180,  60, 200)     # visited / closed
COL_NODE_PATH  = (50,  220, 200)     # final path

COL_EDGE       = (50,  60, 100)
COL_EDGE_PATH  = (50,  220, 200)
COL_EDGE_CONG  = (240, 100,  40)     # congested edge (high weight)

COL_TEXT_HI    = (220, 230, 255)
COL_TEXT_LO    = (110, 120, 160)
COL_ACCENT     = ( 80, 160, 255)
COL_SUCCESS    = ( 50, 220, 120)
COL_ERROR      = (240,  70,  70)
COL_WARNING    = (255, 180,  40)
COL_WHITE      = (255, 255, 255)

NODE_RADIUS    = 18
CONGESTION_THR = 15.0   # edges with weight >= this get orange tint


# ---------------------------------------------------------------------------
# Button helper
# ---------------------------------------------------------------------------
class Button:
    def __init__(self, rect, label, color=COL_ACCENT, text_color=COL_WHITE,
                 active=False, active_color=COL_SUCCESS):
        self.rect        = pygame.Rect(rect)
        self.label       = label
        self.color       = color
        self.text_color  = text_color
        self.active      = active          # toggle state
        self.active_color = active_color
        self.hovered     = False

    def draw(self, surface, font):
        base = self.active_color if self.active else self.color
        col  = tuple(min(255, c + 30) for c in base) if self.hovered else base
        pygame.draw.rect(surface, col,  self.rect, border_radius=6)
        pygame.draw.rect(surface, COL_WHITE, self.rect, 1, border_radius=6)
        lbl = font.render(self.label, True, self.text_color)
        lx  = self.rect.centerx - lbl.get_width()  // 2
        ly  = self.rect.centery - lbl.get_height() // 2
        surface.blit(lbl, (lx, ly))

    def is_clicked(self, pos) -> bool:
        return self.rect.collidepoint(pos)

    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class Renderer:
    """
    Manages all Pygame drawing and maintains UI widget state.
    main.py calls these methods each frame.
    """

    PANEL_W = 260   # width of right-hand control panel

    def __init__(self, screen: pygame.Surface, font_sm, font_md, font_lg):
        self.screen   = screen
        self.font_sm  = font_sm
        self.font_md  = font_md
        self.font_lg  = font_lg
        W, H = screen.get_size()
        self.canvas_rect = pygame.Rect(0, 0, W - self.PANEL_W, H)
        self.panel_rect  = pygame.Rect(W - self.PANEL_W, 0, self.PANEL_W, H)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def canvas_contains(self, pos) -> bool:
        return self.canvas_rect.collidepoint(pos)

    # ------------------------------------------------------------------
    # Main draw entry-point
    # ------------------------------------------------------------------
    def draw_frame(self, state: dict):
        self.screen.fill(BG_CANVAS, self.canvas_rect)
        self._draw_grid()
        self._draw_edges(state)
        self._draw_nodes(state)
        self._draw_panel(state)
        self._draw_status_bar(state)
        self._draw_alert(state)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    def _draw_grid(self):
        W, H = self.canvas_rect.width, self.canvas_rect.height
        for x in range(0, W, 40):
            pygame.draw.line(self.screen, (18, 24, 48), (x, 0), (x, H))
        for y in range(0, H, 40):
            pygame.draw.line(self.screen, (18, 24, 48), (0, y), (W, y))

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------
    def _draw_edges(self, state: dict):
        graph: Graph             = state["graph"]
        result: SearchResult | None = state.get("result")
        result2: SearchResult | None = state.get("result2")
        step: int                = state.get("step", -1)

        path_edges  = set()
        path_edges2 = set()

        def collect_path_edges(res):
            edges = set()
            if res and res.found and res.path:
                p = res.path
                for i in range(len(p) - 1):
                    edges.add((min(p[i], p[i+1]), max(p[i], p[i+1])))
            return edges

        if result  and step < 0: path_edges  = collect_path_edges(result)
        if result2 and step < 0: path_edges2 = collect_path_edges(result2)

        for u, v, w in graph.all_edges():
            n1 = graph.nodes[u]
            n2 = graph.nodes[v]
            key = (min(u, v), max(u, v))

            if key in path_edges or key in path_edges2:
                col   = COL_EDGE_PATH
                thick = 4
            elif w >= CONGESTION_THR:
                col   = COL_EDGE_CONG
                thick = 2
            else:
                col   = COL_EDGE
                thick = 1

            pygame.draw.line(self.screen, col,
                             (int(n1.x), int(n1.y)),
                             (int(n2.x), int(n2.y)), thick)

            # Weight label
            mx = int((n1.x + n2.x) / 2)
            my = int((n1.y + n2.y) / 2)
            lbl = self.font_sm.render(f"{w:.0f}", True, COL_TEXT_LO)
            bg  = pygame.Surface((lbl.get_width()+4, lbl.get_height()+2), pygame.SRCALPHA)
            bg.fill((10, 14, 30, 180))
            self.screen.blit(bg,  (mx - lbl.get_width()//2 - 2, my - lbl.get_height()//2 - 1))
            self.screen.blit(lbl, (mx - lbl.get_width()//2,     my - lbl.get_height()//2))

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    def _node_color(self, nid: int, state: dict) -> tuple:
        start  = state.get("start")
        goal   = state.get("goal")
        result = state.get("result")
        step   = state.get("step", -1)

        if nid == start: return COL_NODE_START
        if nid == goal:  return COL_NODE_GOAL

        if result:
            if step >= 0:
                # step-through mode
                idx = min(step, len(result.history) - 1)
                h   = result.history[idx]
                if nid == h["expanded"]:         return COL_NODE_VISIT
                if nid in h.get("frontier", []): return COL_NODE_FRONT
                if nid in h.get("visited",  []): return COL_NODE_VISIT
            else:
                if result.found and nid in result.path: return COL_NODE_PATH

        return COL_NODE

    def _draw_nodes(self, state: dict):
        graph: Graph = state["graph"]
        pending_edge = state.get("pending_edge_src")
        mode         = state.get("mode", "")

        for nid, node in graph.nodes.items():
            col = self._node_color(nid, state)

            # Glow ring
            glow_surf = pygame.Surface((NODE_RADIUS*4, NODE_RADIUS*4), pygame.SRCALPHA)
            glow_col  = (*col, 40)
            pygame.draw.circle(glow_surf, glow_col,
                               (NODE_RADIUS*2, NODE_RADIUS*2), NODE_RADIUS*2)
            self.screen.blit(glow_surf,
                             (int(node.x) - NODE_RADIUS*2,
                              int(node.y) - NODE_RADIUS*2))

            # Outer ring
            pygame.draw.circle(self.screen, col, (int(node.x), int(node.y)), NODE_RADIUS)
            pygame.draw.circle(self.screen, COL_WHITE, (int(node.x), int(node.y)), NODE_RADIUS, 2)

            # Highlight pending source
            if nid == pending_edge:
                pygame.draw.circle(self.screen, COL_WARNING,
                                   (int(node.x), int(node.y)), NODE_RADIUS + 4, 2)

            # Label
            lbl = self.font_sm.render(node.label, True, COL_WHITE)
            self.screen.blit(lbl, (int(node.x) - lbl.get_width()//2,
                                   int(node.y) - lbl.get_height()//2))

    # ------------------------------------------------------------------
    # Panel
    # ------------------------------------------------------------------
    def _draw_panel(self, state: dict):
        s = self.screen
        r = self.panel_rect
        s.fill(BG_PANEL, r)
        pygame.draw.line(s, PANEL_BORDER, r.topleft, r.bottomleft, 2)

        y = r.y + 12
        # Title
        title = self.font_lg.render("NetPath AI", True, COL_ACCENT)
        s.blit(title, (r.x + 10, y)); y += 32
        sub = self.font_sm.render("Traffic Routing Simulator", True, COL_TEXT_LO)
        s.blit(sub,   (r.x + 10, y)); y += 24

        self._divider(y); y += 14

        # Buttons drawn externally; we just draw their labels here
        for btn in state.get("buttons", []):
            btn.draw(s, self.font_sm)

        # Metrics
        y = r.y + 390
        self._divider(y); y += 12
        self._label(s, "METRICS", r.x + 10, y, COL_ACCENT); y += 22

        result  = state.get("result")
        result2 = state.get("result2")
        algo    = state.get("algo_label", "—")
        algo2   = state.get("algo2_label", "—")

        if result:
            self._metric_block(s, r.x + 10, y, algo, result); y += 90
        if result2:
            self._metric_block(s, r.x + 10, y, algo2, result2); y += 90

        if not result and not result2:
            self._label(s, "No results yet.", r.x + 10, y, COL_TEXT_LO)

        # Mode hint
        mode_hint = {
            "ADD_NODE":   "Click canvas to place node",
            "ADD_EDGE":   "Click source, then target node",
            "SET_START":  "Click a node to set as START",
            "SET_GOAL":   "Click a node to set as GOAL",
            "CONGESTION": "Click an edge midpoint to boost weight",
            "DELETE":     "Click a node to delete it",
        }.get(state.get("mode", ""), "")
        if mode_hint:
            hint = self.font_sm.render(mode_hint, True, COL_WARNING)
            s.blit(hint, (r.x + 8, r.bottom - 50))

    def _metric_block(self, s, x, y, label, res: SearchResult):
        col = COL_SUCCESS if res.found else COL_ERROR
        self._label(s, label, x, y, col);           y += 18
        found_txt = "✔ Path Found" if res.found else "✘ No Path (Timeout)"
        self._label(s, found_txt, x, y, COL_TEXT_HI); y += 16
        self._label(s, f"  Cost       : {res.cost:.2f}" if res.found else "  Cost: N/A",
                    x, y, COL_TEXT_HI); y += 16
        self._label(s, f"  Expanded   : {res.nodes_expanded}", x, y, COL_TEXT_HI); y += 16
        self._label(s, f"  Time       : {res.elapsed_ms:.2f} ms", x, y, COL_TEXT_HI)

    def _label(self, s, text, x, y, color=COL_TEXT_HI):
        surf = self.font_sm.render(text, True, color)
        s.blit(surf, (x, y))

    def _divider(self, y):
        r = self.panel_rect
        pygame.draw.line(self.screen, PANEL_BORDER,
                         (r.x + 4, y), (r.right - 4, y), 1)

    # ------------------------------------------------------------------
    # Status bar at bottom of canvas
    # ------------------------------------------------------------------
    def _draw_status_bar(self, state: dict):
        bar = pygame.Rect(0, self.canvas_rect.height - 26,
                          self.canvas_rect.width, 26)
        pygame.draw.rect(self.screen, (14, 18, 36), bar)
        step  = state.get("step", -1)
        total = 0
        res   = state.get("result")
        if res:
            total = len(res.history)
        if step >= 0 and total:
            txt = f"Step {step+1}/{total}  |  Press → / ← to navigate  |  Enter to finish"
            col = COL_WARNING
        else:
            txt = "NetPath AI  |  Place nodes → connect edges → set START/GOAL → Run"
            col = COL_TEXT_LO
        lbl = self.font_sm.render(txt, True, col)
        self.screen.blit(lbl, (8, self.canvas_rect.height - 20))

    # ------------------------------------------------------------------
    # Alert overlay
    # ------------------------------------------------------------------
    def _draw_alert(self, state: dict):
        alert = state.get("alert")
        if not alert:
            return
        W, H = self.screen.get_size()
        overlay = pygame.Surface((W - self.PANEL_W, 60), pygame.SRCALPHA)
        overlay.fill((240, 70, 70, 200))
        self.screen.blit(overlay, (0, H // 2 - 30))
        lbl = self.font_md.render(alert, True, COL_WHITE)
        self.screen.blit(lbl, (
            (W - self.PANEL_W) // 2 - lbl.get_width() // 2,
            H // 2 - lbl.get_height() // 2,
        ))

    # ------------------------------------------------------------------
    # Node-click detection
    # ------------------------------------------------------------------
    @staticmethod
    def find_node_at(pos, graph: Graph) -> int | None:
        x, y = pos
        for nid, node in graph.nodes.items():
            if math.hypot(node.x - x, node.y - y) <= NODE_RADIUS + 4:
                return nid
        return None

    # ------------------------------------------------------------------
    # Weight input dialog (blocking, but lightweight)
    # ------------------------------------------------------------------
    def prompt_weight(self, prompt_text: str = "Edge weight:") -> float | None:
        """
        Small in-window text input for edge weight.
        Returns float or None if cancelled.
        """
        clock  = pygame.time.Clock()
        text   = ""
        active = True
        W, H   = self.screen.get_size()
        box    = pygame.Rect(W // 2 - 150, H // 2 - 40, 300, 80)

        while active:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_RETURN:
                        active = False
                    elif ev.key == pygame.K_ESCAPE:
                        return None
                    elif ev.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif ev.unicode in "0123456789.":
                        text += ev.unicode

            # Draw overlay
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            pygame.draw.rect(self.screen, BG_PANEL, box, border_radius=10)
            pygame.draw.rect(self.screen, COL_ACCENT, box, 2, border_radius=10)

            p_surf = self.font_sm.render(prompt_text, True, COL_TEXT_LO)
            self.screen.blit(p_surf, (box.x + 12, box.y + 10))

            input_rect = pygame.Rect(box.x + 12, box.y + 34, box.width - 24, 28)
            pygame.draw.rect(self.screen, (25, 30, 60), input_rect, border_radius=4)
            pygame.draw.rect(self.screen, COL_ACCENT, input_rect, 1, border_radius=4)
            t_surf = self.font_md.render(text + "|", True, COL_WHITE)
            self.screen.blit(t_surf, (input_rect.x + 6, input_rect.y + 3))

            hint = self.font_sm.render("Enter=confirm  Esc=cancel", True, COL_TEXT_LO)
            self.screen.blit(hint, (box.x + 12, box.y + 64))

            pygame.display.flip()
            clock.tick(60)

        try:
            val = float(text)
            return val if val >= 0 else None
        except ValueError:
            return None
