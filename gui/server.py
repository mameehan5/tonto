"""Local web GUI for playing and evaluating Tonto.

Usage: python gui/server.py [--port 8000] [--no-browser]
"""
import argparse
import json
import sys
import warnings
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.engine import TontoGame
from game.bots import BOTS, make_bot


class Session:
    def __init__(self):
        self.new(players=4, decks=1, rounds=2, cards=6, seats=["human", "greedy", "greedy", "greedy"], seed=None)

    def new(self, players, decks, rounds, cards, seats, seed):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.game = TontoGame(players, decks, rounds, cards, seed=seed)
        self.warning = str(w[0].message) if w else None
        self.seats = (list(seats) + ["greedy"] * players)[:players]
        self.bots = {i: make_bot(s) for i, s in enumerate(self.seats) if s != "human"}
        self.history = []

    def _save(self):
        self.history.append(self.game.to_dict())

    def act(self, action):
        self._save()
        self.game.step(action)

    def bot(self):
        pid = self.game.game_state.current_player
        self._save()
        self.game.step(self.bots[pid].act(self.game, pid))

    def next_round(self):
        self._save()
        self.game.next_round()

    def undo(self):
        if self.history:
            self.game = TontoGame.from_dict(self.history.pop())

    def state(self, reveal):
        humans = [i for i, s in enumerate(self.seats) if s == "human"]
        viewer = None if reveal or not humans else humans[0]
        models = [f"ppo:{p.relative_to(ROOT)}" for p in sorted(ROOT.glob("models/**/*.zip"))]
        return {"view": self.game.view(viewer), "seats": self.seats, "can_undo": bool(self.history),
                "seat_options": ["human"] + [b for b in BOTS if b != "greedy_v0"] + ["greedy_v0"] + models,
                "warning": self.warning, "config": self.game.to_dict()["config"]}


session = Session()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send(200, session.state("reveal=0" not in self.path))
        else:
            self._send(200, (ROOT / "gui" / "index.html").read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            cmd = self.path.split("/")[-1]
            if cmd == "new":
                session.new(int(body["players"]), int(body["decks"]), int(body["rounds"]), int(body["cards"]),
                            body["seats"], int(body["seed"]) if body.get("seed") not in (None, "") else None)
            elif cmd == "act":
                session.act(int(body["action"]))
            elif cmd == "bot":
                session.bot()
            elif cmd == "next_round":
                session.next_round()
            elif cmd == "undo":
                session.undo()
            self._send(200, session.state(body.get("reveal", True)))
        except Exception as e:  # surface config/model errors in the GUI
            self._send(400, {"error": str(e) or type(e).__name__})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://localhost:{a.port}"
    print(f"Tonto GUI running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    HTTPServer(("localhost", a.port), Handler).serve_forever()
