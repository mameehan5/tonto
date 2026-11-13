"""End-to-end server test: two humans + a bot play full games over websockets.
Checks each player only ever sees their own hidden cards. Usage: python tests/test_server.py"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server.app as srv
from fastapi.testclient import TestClient

srv.BOT_DELAY = 0
client = TestClient(srv.app)
client.__enter__()  # one event loop shared by all connections, like the real server


last = {}  # websocket -> most recent state message


def latest(ws, until):
    """Return the most recent state if it satisfies `until`, else read messages until one does."""
    if ws in last and until(last[ws]):
        return last[ws]
    while True:
        m = json.loads(ws.receive_text())
        if m["type"] == "error":
            raise AssertionError(m["message"])
        last[ws] = m
        if until(m):
            return m


def check_private(m):
    me = m["you"]
    for p in m["view"]["players"]:
        if p["id"] != me:
            assert all(c["card"] is None for c in p["cards"] if not c["face_up"]), "leaked a hidden card"
            assert p["known_score"] is None
    for e in m["view"]["log"]:
        assert e["player"] == me or e["detail"] is None, "leaked a private log detail"


def play_game(seed):
    random.seed(seed)
    r = client.post("/api/rooms", json={"name": "Mike", "players": 3, "cards": 6, "rounds": 2}).json()
    code, host = r["code"], r["token"]
    assert client.post(f"/api/rooms/{code}/join", json={"name": "Ana"}).status_code == 200
    room = srv.rooms[code]
    friend = room.seats[1]["token"]
    assert client.post(f"/api/rooms/ZZZZ/join", json={"name": "x"}).status_code == 404

    with client.websocket_connect(f"/ws/{code}?token={host}") as wh, \
         client.websocket_connect(f"/ws/{code}?token={friend}") as wf:
        latest(wh, lambda m: True)
        wh.send_text(json.dumps({"type": "add_bot", "level": "medium"}))
        latest(wh, lambda m: m["seats"][2] is not None)
        wh.send_text(json.dumps({"type": "start"}))
        socks = {0: wh, 1: wf}
        current = lambda cur, n: latest(socks[cur], lambda m: m["view"] and len(m["view"]["log"]) == n and m["view"]["legal_actions"])
        latest(wh, lambda m: m["view"])
        # Out-of-turn moves are rejected (game starts with seat 0)
        wf.send_text(json.dumps({"type": "action", "action": 0}))
        try:
            latest(wf, lambda m: False)
        except AssertionError as e:
            assert "not your turn" in str(e)
        moves = 0
        while not room.game.game_state.complete:
            g = room.game
            if g.round_state.complete:
                wh.send_text(json.dumps({"type": "next_round"}))
                time.sleep(0.05)
                continue
            cur = g.game_state.current_player
            if cur == 2:  # bot's turn
                time.sleep(0.01)
                continue
            n = len(g.log)  # match the state message for exactly this position
            m = current(cur, n)
            check_private(m)
            socks[cur].send_text(json.dumps({"type": "action", "action": random.choice(m["view"]["legal_actions"])["action"]}))
            while len(room.game.log) == n:
                time.sleep(0.005)
            moves += 1
        m = latest(wh, lambda m: m["view"]["game_complete"])
        check_private(m)
        return code, moves, m["view"]["winner"]


for seed in range(5):
    code, moves, winner = play_game(seed)
    # Saved rooms reload with identical state
    reloaded = srv.Room.load(srv.SAVE_DIR / f"{code}.json")
    assert reloaded.game.to_dict() == srv.rooms[code].game.to_dict()
    (srv.SAVE_DIR / f"{code}.json").unlink()
    print(f"game {seed}: {moves} human moves, winner seat {winner}, save/reload ok")
print("PASS")
