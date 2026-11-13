"""Tonto multiplayer server: rooms by code, server-authoritative game state, bots fill empty seats.

Run: uvicorn server.app:app --host 0.0.0.0 --port 8000
Each player's browser only ever receives that player's view of the game (TontoGame.view(seat)).
Rooms are saved to rooms/<CODE>.json after every change, so a restart doesn't lose games.
"""
import asyncio
import json
import secrets
import sys
import warnings
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game.bots import make_bot
from game.engine import TontoGame

STATIC = Path(__file__).parent / "static"
SAVE_DIR = ROOT / "rooms"
SAVE_DIR.mkdir(exist_ok=True)
BOT_DELAY = 0.7  # seconds between bot moves, so humans can follow along
BOTS = {"easy": ("Easy bot", "greedy_v0"), "medium": ("Medium bot", "greedy"), "hard": ("Hard bot", "search:24")}
CODE_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I/O, easy to read aloud


class Room:
    def __init__(self, code, settings):
        self.code = code
        self.settings = settings  # players, cards, decks, rounds
        self.seats = [None] * settings["players"]  # {"name", "token"} or {"name", "bot"}
        self.host = None  # host token
        self.game = None
        self.sockets = {}  # token -> set of websockets
        self.lock = asyncio.Lock()
        self.bot_task = None
        self.bots = {}

    # Seats ----------------------------------------------------------------

    def seat_of(self, token):
        return next((i for i, s in enumerate(self.seats) if s and s.get("token") == token), None)

    def add_human(self, name):
        i = self.seats.index(None)
        token = secrets.token_urlsafe(16)
        self.seats[i] = {"name": name[:20], "token": token}
        self.host = self.host or token
        return i, token

    def bot_for(self, i):
        if i not in self.bots:
            self.bots[i] = make_bot(BOTS[self.seats[i]["bot"]][1])
        return self.bots[i]

    # State ----------------------------------------------------------------

    def state_for(self, token):
        seat = self.seat_of(token)
        return {
            "type": "state",
            "code": self.code,
            "settings": self.settings,
            "you": seat,
            "is_host": token == self.host,
            "seats": [None if s is None else {
                "name": s["name"], "bot": s.get("bot"),
                "online": "token" in s and bool(self.sockets.get(s["token"])),
            } for s in self.seats],
            "bot_levels": {k: v[0] for k, v in BOTS.items()},
            "view": self.game.view(seat) if self.game else None,
        }

    async def broadcast(self):
        self.save()
        for token, socks in list(self.sockets.items()):
            msg = json.dumps(self.state_for(token))
            for ws in list(socks):
                try:
                    await ws.send_text(msg)
                except Exception:
                    socks.discard(ws)

    def save(self):
        data = {"code": self.code, "settings": self.settings, "seats": self.seats, "host": self.host,
                "game": self.game.to_dict() if self.game else None}
        (SAVE_DIR / f"{self.code}.json").write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        r = cls(d["code"], d["settings"])
        r.seats, r.host = d["seats"], d["host"]
        r.game = TontoGame.from_dict(d["game"]) if d["game"] else None
        return r

    # Play -----------------------------------------------------------------

    def auto_forced(self):
        """Apply moves that have only one option (e.g. ending a turn) for human players."""
        g = self.game
        while not g.round_state.complete and len(g.legal_actions()) == 1 \
                and "token" in self.seats[g.game_state.current_player]:
            g.step(g.legal_actions()[0])

    async def run_bots(self):
        """Play bot turns one move at a time, broadcasting each move."""
        g = self.game
        while g and not g.round_state.complete and "bot" in self.seats[g.game_state.current_player]:
            await asyncio.sleep(BOT_DELAY)
            async with self.lock:
                pid = g.game_state.current_player
                bot = self.bot_for(pid)
                action = await asyncio.to_thread(bot.act, g, pid)
                g.step(action)
                self.auto_forced()
            await self.broadcast()

    def kick_bots(self):
        if self.bot_task is None or self.bot_task.done():
            self.bot_task = asyncio.create_task(self.run_bots())


rooms: dict[str, Room] = {}
for f in SAVE_DIR.glob("*.json"):
    try:
        rooms[f.stem] = Room.load(f)
    except Exception as e:  # a corrupt save shouldn't stop the server
        print(f"Skipping {f.name}: {e}")

app = FastAPI(title="Tonto")


class CreateReq(BaseModel):
    name: str
    players: int = 4
    cards: int = 6
    decks: int = 1
    rounds: int = 3


class JoinReq(BaseModel):
    name: str


def validate(s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            TontoGame(s["players"], s["decks"], s["rounds"], s["cards"], seed=0)
        except AssertionError as e:
            raise HTTPException(400, str(e) or "Those settings don't make a valid game")


@app.post("/api/rooms")
async def create_room(req: CreateReq):
    settings = {"players": req.players, "cards": req.cards, "decks": req.decks, "rounds": req.rounds}
    validate(settings)
    code = "".join(secrets.choice(CODE_LETTERS) for _ in range(4))
    while code in rooms:
        code = "".join(secrets.choice(CODE_LETTERS) for _ in range(4))
    room = rooms[code] = Room(code, settings)
    seat, token = room.add_human(req.name or "Host")
    room.save()
    return {"code": code, "token": token, "seat": seat}


@app.post("/api/rooms/{code}/join")
async def join_room(code: str, req: JoinReq):
    room = rooms.get(code.upper())
    if not room:
        raise HTTPException(404, f"No table with code {code.upper()}. Check the code with the host.")
    if room.game:
        raise HTTPException(409, "That game has already started.")
    if None not in room.seats:
        raise HTTPException(409, "That table is full.")
    async with room.lock:
        seat, token = room.add_human(req.name or f"Player {len(room.seats)}")
    await room.broadcast()
    return {"code": room.code, "token": token, "seat": seat}


@app.websocket("/ws/{code}")
async def play(ws: WebSocket, code: str, token: str):
    room = rooms.get(code.upper())
    await ws.accept()
    if not room or room.seat_of(token) is None:
        await ws.send_text(json.dumps({"type": "error", "message": "This seat no longer exists.", "fatal": True}))
        await ws.close()
        return
    room.sockets.setdefault(token, set()).add(ws)
    await room.broadcast()
    if room.game:
        room.kick_bots()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            try:
                await handle(room, token, msg)
            except (ValueError, KeyError, IndexError, AssertionError) as e:
                await ws.send_text(json.dumps({"type": "error", "message": str(e) or "That move isn't allowed."}))
    except WebSocketDisconnect:
        room.sockets.get(token, set()).discard(ws)
        await room.broadcast()


async def handle(room, token, msg):
    kind = msg["type"]
    seat = room.seat_of(token)
    host = token == room.host
    async with room.lock:
        if kind == "action":
            g = room.game
            if g is None or g.game_state.current_player != seat:
                raise ValueError("It's not your turn.")
            g.step(int(msg["action"]))
            room.auto_forced()
        elif kind == "next_round":
            if room.game.round_state.complete and not room.game.game_state.complete:
                room.game.next_round()
                room.auto_forced()
        elif kind == "settings" and host and not room.game:
            s = {k: int(msg[k]) for k in ("players", "cards", "decks", "rounds")}
            validate(s)
            humans = [x for x in room.seats if x]
            if s["players"] < len(humans):
                raise ValueError("Remove a player before making the table smaller.")
            room.settings = s
            room.seats = (humans + [None] * s["players"])[:s["players"]]
            room.bots = {}
        elif kind == "add_bot" and host and not room.game:
            i = room.seats.index(None)
            room.seats[i] = {"name": BOTS[msg["level"]][0], "bot": msg["level"]}
        elif kind == "remove" and host and not room.game:
            i = int(msg["seat"])
            if room.seats[i] and room.seats[i].get("token") != room.host:
                room.seats[i] = None
                room.bots.pop(i, None)
        elif kind == "start" and host and not room.game:
            if None in room.seats:
                raise ValueError("Fill every seat (add bots or wait for friends) before starting.")
            s = room.settings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                room.game = TontoGame(s["players"], s["decks"], s["rounds"], s["cards"])
            room.auto_forced()
        elif kind == "rematch" and host and room.game and room.game.game_state.complete:
            room.game = None
            room.bots = {}
    await room.broadcast()
    if room.game:
        room.kick_bots()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
