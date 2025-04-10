from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import asyncio
import discord
import uuid
from dotenv import load_dotenv
import os

app = FastAPI()

# Allow frontend connection0
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI classifier
load_dotenv()
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1", token=os.getenv("HF_TOKEN"))
labels = ["hiring freelancer", 
    "general discussion",
    "spam/advertisement"]

clients = {}  # track running bots per WebSocket

# Message filter
def is_freelance_related(text):
    result = classifier(text, candidate_labels=labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    # Return True only if it's a likely freelance job offer
    return top_label in ["hiring freelancer"] and top_score >= 0.8

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    clients[client_id] = {"websocket": websocket, "bot": None}

    try:
        while True:
            data = await websocket.receive_json()
            token = data.get("token")
            keywords = data.get("keywords", [])

            # Start the Discord bot task
            bot = DiscordBot(token, keywords, websocket)
            clients[client_id]["bot"] = bot
            asyncio.create_task(bot.start_bot())

    except WebSocketDisconnect:
        if client_id in clients:
            bot = clients[client_id].get("bot")
            if bot:
                await bot.stop()
            del clients[client_id]

# Discord self-bot logic
class DiscordBot:
    def __init__(self, token, keywords, websocket):
        self.token = token
        self.keywords = [k.lower() for k in keywords]
        self.websocket = websocket
        self.client = discord.Client()

        @self.client.event
        async def on_ready():
            print(f"Bot ready: {self.client.user}")

        @self.client.event
        async def on_message(message):
            if message.author.id == self.client.user.id:
                return

            content = message.content.lower()
            match_count = sum(1 for k in self.keywords if k in content)

            if match_count >= 2 or (is_freelance_related(content)):
                await self.websocket.send_json({
                    "message": message.content,
                    "server": message.guild.name if message.guild else "DM",
                    "channel": message.channel.name if hasattr(message.channel, 'name') else "Unknown",
                    "author": str(message.author),
                })

    async def start_bot(self):
        try:
            await self.client.start(self.token)
        except Exception as e:
            await self.websocket.send_json({"error": str(e)})

    async def stop(self):
        await self.client.close()
