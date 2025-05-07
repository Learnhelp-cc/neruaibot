import discord
import json
import os
import aiohttp
import asyncio
import random # For emoji reactions
import re
import urllib.parse # For search query encoding
import subprocess # For shell commands
from datetime import datetime, timedelta
from discord.ext import commands
from discord import app_commands
from typing import Optional, Dict, List, Any

# Define paths for persistent data - ENSURE THESE DIRECTORIES ARE WRITABLE
# It's good practice to place these in a dedicated data directory.
DATA_DIR = os.path.join(os.path.dirname(__file__), "bot_data") # Assumes bot_data folder in the same dir as the cog
DEFAULT_MEMORY_PATH = os.path.join(DATA_DIR, "mind.json")
DEFAULT_HISTORY_PATH = os.path.join(DATA_DIR, "ai_conversation_history_neru.json")
DEFAULT_MANUAL_CONTEXT_PATH = os.path.join(DATA_DIR, "ai_manual_context.json")
DEFAULT_DYNAMIC_LEARNING_PATH = os.path.join(DATA_DIR, "ai_dynamic_learning_neru.json")
DEFAULT_CONFIG_PATH = os.path.join(DATA_DIR, "ai_configs.json")

class AICog(commands.Cog, name="AI"):
    """Cog for interacting with an AI model, adapted for Meta Llama Preview API."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # --- API Configuration ---
        # IMPORTANT: Ensure AI_API_KEY environment variable is set to your Meta Llama Preview API key.
        self.api_key = os.getenv("AI_API_KEY")
        if not self.api_key:
            print("WARNING: AI_API_KEY environment variable not found. AI features will be disabled.")
        # IMPORTANT: Verify this is the correct endpoint for the Meta Llama Preview API.
        # It might have a version prefix, a 'preview' segment, or a different subdomain.
        self.api_url = os.getenv("LLAMA_API_URL", "https://api.llama.com/v1/chat/completions") # Placeholder
        # -------------------------

        self.security_code = os.getenv("SERVICE_CODE") # For privileged commands

        # Ensure data directory exists
        if not os.path.exists(DATA_DIR):
            try:
                os.makedirs(DATA_DIR)
                print(f"Created data directory: {DATA_DIR}")
            except OSError as e:
                print(f"FATAL: Could not create data directory {DATA_DIR}. Persistent data will fail. Error: {e}")


        # --- Memory Setup ---
        self.memory_file_path = os.getenv("BOT_MEMORY_PATH", DEFAULT_MEMORY_PATH)
        self.user_memory: Dict[str, List[str]] = {}
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.manual_context: List[str] = []
        self.dynamic_learning: List[str] = []

        self.load_memory()
        self.history_file_path = os.getenv("BOT_HISTORY_PATH", DEFAULT_HISTORY_PATH)
        self.load_history()
        self.manual_context_file_path = os.getenv("BOT_MANUAL_CONTEXT_PATH", DEFAULT_MANUAL_CONTEXT_PATH)
        self.load_manual_context()
        self.dynamic_learning_file_path = os.getenv("BOT_DYNAMIC_LEARNING_PATH", DEFAULT_DYNAMIC_LEARNING_PATH)
        self.load_dynamic_learning()
        # --------------------

        # --- Default AI Configuration ---
        # IMPORTANT: Verify these parameters and their names/values are compatible with the Meta Llama Preview API.
        self.default_config = {
            "model": "meta-llama/Llama-3-70b-chat-hf", # Example, ensure this is a valid model for the Meta Llama Preview API
            "temperature": 0.75,
            "max_tokens": 1500, # Meta Llama API might use 'max_gen_len' or 'max_new_tokens'
            "top_p": 0.9,
            "frequency_penalty": 0.1, # Verify if supported
            "presence_penalty": 0.1,  # Verify if supported
        }
        
        self.user_configs = {}
        self.config_file = os.getenv("BOT_CONFIG_PATH", DEFAULT_CONFIG_PATH)
        self.load_configs()
        
        self.active_channels = set() # Example: for channel-specific AI listening

        # --- System Prompt ---
        self.system_prompt_template = (
            "You are Kasane Teto, a cheerful and helpful UTAUloid and Discord bot. "
            "You are interacting with users on Discord. Be friendly, a bit quirky, and always try your best to assist. "
            "You have access to tools to remember facts about users and run safe shell commands. "
            "Current date: {current_date}\n\n"
            "MANUAL CONTEXT (Important information to always consider):\n{manual_context}\n\n"
            "DYNAMIC LEARNING EXAMPLES (Preferred ways to respond or act based on past interactions):\n{dynamic_learning_context}\n\n"
            "USER-SPECIFIC MEMORY (Facts you remember about the current user, {user_name} (ID: {user_id})):\n{user_memory_context}"
        )
        # ---------------------------

        # --- Tool Definitions ---
        # IMPORTANT: Verify the tool definition format and calling mechanism with Meta Llama Preview API documentation.
        # This format is based on OpenAI's function calling. Meta Llama might have a different schema.
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_safe_shell_command",
                    "description": "Executes a simple, safe, read-only shell command if necessary to answer a user's question (e.g., get current date, list files, check uptime). Prohibited commands include file modification, cat, sudo, rm, mv, cp, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The safe shell command to execute (e.g., 'date', 'ls -l', 'uptime', 'ping -c 1 google.com').",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remember_fact_about_user",
                    "description": "Stores a concise fact learned about the user during the conversation (e.g., 'likes pineapple pizza', 'favorite color is blue', 'has a dog named Sparky'). Only use this for explicit facts stated by the user or directly implied. Do not make assumptions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "The Discord User ID of the user the fact pertains to. This should be the ID of the user currently interacting with you.",
                            },
                             "fact": {
                                "type": "string",
                                "description": "The specific, concise fact to remember about the user.",
                            }
                        },
                        "required": ["user_id", "fact"],
                    },
                },
            }
            # Add more tools here if needed, e.g., for web search, calendar, etc.
            # Ensure each tool has a corresponding handler method in this class.
        ]
        # ------------------------

    # --- Helper for ensuring directory exists ---
    def _ensure_dir_exists(self, file_path: str) -> bool:
        """Ensures the directory for a given file_path exists. Creates it if not."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
                return True
            except OSError as e:
                print(f"ERROR: Could not create directory {directory}. Error: {e}")
                return False
        return True

    # --- Memory Management ---
    def load_memory(self):
        if not self._ensure_dir_exists(self.memory_file_path):
            print("Memory loading aborted as directory couldn't be ensured.")
            self.user_memory = {}
            return
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                    self.user_memory = json.load(f)
                print(f"Loaded memory for {len(self.user_memory)} users from {self.memory_file_path}")
            else:
                print(f"Memory file not found at {self.memory_file_path}. Starting with empty memory.")
                self.user_memory = {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from memory file {self.memory_file_path}: {e}. Starting with empty memory.")
            self.user_memory = {}
        except Exception as e:
            print(f"Error loading memory from {self.memory_file_path}: {e}. Starting with empty memory.")
            self.user_memory = {}

    def save_memory(self):
        if not self._ensure_dir_exists(self.memory_file_path):
            print("Memory saving aborted as directory couldn't be ensured.")
            return
        try:
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_memory, f, indent=4, ensure_ascii=False)
            # print(f"Saved memory to {self.memory_file_path}")
        except Exception as e:
            print(f"Error saving memory to {self.memory_file_path}: {e}")

    def add_user_fact(self, user_id: str, fact: str):
        user_id_str = str(user_id)
        fact = fact.strip()
        if not fact: return

        if user_id_str not in self.user_memory:
            self.user_memory[user_id_str] = []
        
        if not any(fact.lower() == existing_fact.lower() for existing_fact in self.user_memory[user_id_str]):
            self.user_memory[user_id_str].append(fact)
            print(f"Added fact for user {user_id_str}: '{fact}'")
            self.save_memory()

    def get_user_facts(self, user_id: str) -> List[str]:
        return self.user_memory.get(str(user_id), [])

    # --- History Management ---
    def load_history(self):
        if not self._ensure_dir_exists(self.history_file_path):
            self.conversation_history = {}
            return
        try:
            if os.path.exists(self.history_file_path):
                with open(self.history_file_path, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                print(f"Loaded history for {len(self.conversation_history)} users from {self.history_file_path}")
            else:
                print(f"History file not found at {self.history_file_path}. Starting empty.")
                self.conversation_history = {}
                self.save_history() # Create if not exists
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from history file {self.history_file_path}: {e}. Starting empty.")
            self.conversation_history = {}
        except Exception as e:
            print(f"Error loading history from {self.history_file_path}: {e}. Starting empty.")
            self.conversation_history = {}

    def save_history(self):
        if not self._ensure_dir_exists(self.history_file_path):
            return
        try:
            with open(self.history_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=4, ensure_ascii=False)
            # print(f"Saved history to {self.history_file_path}")
        except Exception as e:
            print(f"Error saving history to {self.history_file_path}: {e}")

    def add_to_history(self, user_id: str, role: str, content: str):
        user_id_str = str(user_id)
        if user_id_str not in self.conversation_history:
            self.conversation_history[user_id_str] = []

        self.conversation_history[user_id_str].append({"role": role, "content": content})
        max_history_messages = 20 # e.g., 10 turns
        if len(self.conversation_history[user_id_str]) > max_history_messages:
            self.conversation_history[user_id_str] = self.conversation_history[user_id_str][-max_history_messages:]
        self.save_history()

    def get_user_history(self, user_id: str) -> List[Dict[str, str]]:
        return self.conversation_history.get(str(user_id), [])

    # --- Manual Context Management ---
    def load_manual_context(self):
        if not self._ensure_dir_exists(self.manual_context_file_path):
            self.manual_context = []
            return
        try:
            if os.path.exists(self.manual_context_file_path):
                with open(self.manual_context_file_path, 'r', encoding='utf-8') as f:
                    self.manual_context = json.load(f)
                print(f"Loaded {len(self.manual_context)} manual context entries from {self.manual_context_file_path}")
            else:
                self.manual_context = []
                self.save_manual_context()
        except json.JSONDecodeError as e:
            print(f"Error decoding manual context: {e}. Starting empty.")
            self.manual_context = []
        except Exception as e:
            print(f"Error loading manual context: {e}. Starting empty.")
            self.manual_context = []

    def save_manual_context(self):
        if not self._ensure_dir_exists(self.manual_context_file_path):
            return
        try:
            with open(self.manual_context_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.manual_context, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving manual context: {e}")

    def add_manual_context(self, text: str) -> bool:
        text = text.strip()
        if text and text not in self.manual_context:
            self.manual_context.append(text)
            self.save_manual_context()
            print(f"Added manual context: '{text[:50]}...'")
            return True
        return False

    # --- Dynamic Learning Management ---
    def load_dynamic_learning(self):
        if not self._ensure_dir_exists(self.dynamic_learning_file_path):
            self.dynamic_learning = []
            return
        try:
            if os.path.exists(self.dynamic_learning_file_path):
                with open(self.dynamic_learning_file_path, 'r', encoding='utf-8') as f:
                    self.dynamic_learning = json.load(f)
                print(f"Loaded {len(self.dynamic_learning)} dynamic learning entries from {self.dynamic_learning_file_path}")
            else:
                self.dynamic_learning = []
                self.save_dynamic_learning()
        except json.JSONDecodeError as e:
            print(f"Error decoding dynamic learning: {e}. Starting empty.")
            self.dynamic_learning = []
        except Exception as e:
            print(f"Error loading dynamic learning: {e}. Starting empty.")
            self.dynamic_learning = []

    def save_dynamic_learning(self):
        if not self._ensure_dir_exists(self.dynamic_learning_file_path):
            return
        try:
            with open(self.dynamic_learning_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.dynamic_learning, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving dynamic learning: {e}")

    def add_dynamic_learning(self, text: str) -> bool:
        text = text.strip()
        if text and text not in self.dynamic_learning:
            self.dynamic_learning.append(text)
            self.save_dynamic_learning()
            print(f"Added dynamic learning example: '{text[:50]}...'")
            return True
        return False

    # --- Config Management ---
    def load_configs(self):
        if not self._ensure_dir_exists(self.config_file):
            self.user_configs = {}
            return
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_configs = json.load(f)
                    for user_id, config in loaded_configs.items():
                        # Ensure loaded configs inherit from default and then update
                        self.user_configs[user_id] = self.default_config.copy()
                        self.user_configs[user_id].update(config)
            else:
                self.user_configs = {}
        except json.JSONDecodeError as e:
            print(f"Error loading configurations (invalid JSON): {e}")
            self.user_configs = {}
        except Exception as e:
            print(f"Error loading configurations: {e}")
            self.user_configs = {}

    def save_configs(self):
        if not self._ensure_dir_exists(self.config_file):
            return
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.user_configs, f, indent=4)
        except Exception as e:
            print(f"Error saving configurations: {e}")
    
    def get_user_config(self, user_id: str) -> Dict:
        return self.user_configs.get(str(user_id), self.default_config).copy()

    # --- Tool Implementations ---
    def is_safe_command(self, command: str) -> bool:
        """Checks if a shell command is on an allowlist and not on a denylist."""
        if not command: return False
        command_lower = command.lower().strip()
        # Simple allowlist (expand as needed)
        allowed_commands = ["date", "uptime", "ls", "ping", "echo", "hostname", "uname", "df", "free"]
        # Simple denylist (expand as needed)
        disallowed_keywords = ["sudo", "rm", "mv", "cp", "mkfs", "shutdown", "reboot", "kill", "cat", ">", "|", ";", "&", "`", "$(", "wget", "curl"] # Be cautious with these

        # Check if the base command is allowed
        base_command = command_lower.split(" ")[0]
        if base_command not in allowed_commands:
            print(f"Command base '{base_command}' not in allowed list.")
            return False

        # Check for disallowed keywords/characters anywhere in the command
        for keyword in disallowed_keywords:
            if keyword in command_lower: # Check if keyword is a substring
                # Allow 'ping -c 1 google.com' but not 'echo something; rm -rf /'
                # This is a basic check, more sophisticated parsing might be needed for complex cases.
                # For instance, ensure keywords are standalone or part of arguments, not part of file names if ls is used.
                # A safer approach for commands like ls is to sanitize arguments or not allow arguments.
                if keyword == "cat" and base_command == "ls": # e.g. ls "file with cat in name"
                    pass # Allow if 'cat' is part of a filename for 'ls'
                elif keyword in [">", "|", ";", "&", "`", "$("] and base_command in ["echo", "ls"]: # Allow for specific safe uses if any
                     pass # Example: echo "hello > world" is okay if echo is sandboxed
                else:
                    print(f"Disallowed keyword '{keyword}' found in command.")
                    return False
        
        # Specific check for ping to ensure it has -c (count) to prevent flooding
        if base_command == "ping" and "-c" not in command_lower:
            print("Ping command must include a count (-c) parameter.")
            return False

        return True

    async def run_shell_command(self, command: str) -> str:
        """Executes a shell command safely and returns its output."""
        if not self.is_safe_command(command):
            return f"Error: Command '{command}' is not allowed for safety reasons."
        try:
            # Use asyncio.create_subprocess_shell for non-blocking execution
            # Split command into list for Popen if not using shell=True, but shell=True is often easier for simple commands.
            # For security, if shell=True, the command itself must be heavily sanitized, which is_safe_command aims to do.
            # Consider using shlex.split(command) if not using shell=True and command has arguments.
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15.0) # 15 sec timeout

            if process.returncode == 0:
                return stdout.decode('utf-8', errors='replace').strip()
            else:
                return f"Error executing command (Code {process.returncode}): {stderr.decode('utf-8', errors='replace').strip()}"
        except asyncio.TimeoutError:
            return "Error: Command execution timed out."
        except Exception as e:
            return f"Error running shell command: {e}"

    async def search_internet(self, query: str) -> str:
        """Placeholder for an internet search function. Implement with a search API."""
        # Example: You might use Google Custom Search API, DuckDuckGo API, etc.
        # For now, it returns a placeholder.
        print(f"AI requested internet search for: {query}")
        # This is where you'd make an API call to a search engine.
        # For example, using a hypothetical search tool:
        # results = await some_search_library.search(query, api_key="YOUR_SEARCH_API_KEY", num_results=3)
        # formatted_results = "\n".join([f"- {res.title}: {res.snippet} ({res.url})" for res in results])
        # return f"Search results for '{query}':\n{formatted_results}"
        
        # Using a simple DuckDuckGo link as a placeholder
        encoded_query = urllib.parse.quote_plus(query)
        return (f"I would search for '{query}'. "
                f"You can try searching on DuckDuckGo: https://duckduckgo.com/?q={encoded_query}\n"
                f"[System Note: This is a placeholder. Actual search results would be injected here.]")


    async def timeout_user(self, guild_id: int, user_id: int, duration_minutes: int) -> bool:
        """Times out a user in a specific guild."""
        guild = self.bot.get_guild(guild_id)
        if not guild: return False
        member = guild.get_member(user_id)
        if not member: return False

        # Bot needs 'Moderate Members' permission
        if not guild.me.guild_permissions.moderate_members:
            print(f"Bot lacks 'Moderate Members' permission in guild {guild_id} to timeout user {user_id}.")
            return False
        
        # Cannot timeout users with higher roles or self
        if member == guild.me or guild.me.top_role <= member.top_role:
            print(f"Cannot timeout user {user_id} due to role hierarchy or self-timeout attempt.")
            return False

        try:
            delta = timedelta(minutes=duration_minutes)
            await member.timeout(delta, reason="AI initiated timeout via command.")
            print(f"Successfully timed out user {user_id} in guild {guild_id} for {duration_minutes} minutes.")
            return True
        except discord.Forbidden:
            print(f"Forbidden to timeout user {user_id} in guild {guild_id}. Check permissions and role hierarchy.")
        except discord.HTTPException as e:
            print(f"HTTPException while timing out user {user_id}: {e}")
        return False

    # --- Core AI Response Generation ---
    async def generate_response(self, user_id: str, user_name: str, prompt: str, source_message: Optional[discord.Message] = None, source_interaction: Optional[discord.Interaction] = None) -> str:
        if not self.api_key:
            return "Sorry, the AI API key is not configured. I cannot generate a response right now. Please tell my owner!"

        # Determine guild and channel context
        guild = source_message.guild if source_message else (source_interaction.guild if source_interaction else None)
        channel = source_message.channel if source_message else (source_interaction.channel if source_interaction else None)
        guild_id = guild.id if guild else None
        # channel_id = channel.id if channel else None # Not used directly in this version of generate_response

        config = self.get_user_config(user_id)
        user_id_str = str(user_id)

        # --- Regex Command Handling (Example: Timeout, Search) ---
        # These could also be implemented as tools if the AI is reliable in calling them.
        timeout_match = re.search(r"timeout\s+<@!?(\d+)>(?:\s+for\s+(\d+)\s*(minute|minutes|min|mins|hour|hours|day|days))?", prompt, re.IGNORECASE)
        search_match = re.search(r"\b(?:search|google|find|look up)\b(?:\s+for)?\s+(.+?)(?:\s+on\s+the\s+internet)?$", prompt, re.IGNORECASE)
        
        if timeout_match and guild_id and channel: # Ensure channel context for permissions
            # Check if the bot has permission to timeout in this channel/guild
            # This is a simplified check; real permission check might be more complex
            if guild.me.guild_permissions.moderate_members:
                target_id_str = timeout_match.group(1)
                duration_str = timeout_match.group(2) or "5" # Default 5 minutes
                unit = (timeout_match.group(3) or "minutes").lower()
                try:
                    target_id = int(target_id_str)
                    duration = int(duration_str)
                    if unit.startswith("hour"): duration *= 60
                    elif unit.startswith("day"): duration *= 1440
                    duration = min(duration, 28 * 24 * 60) # Discord max timeout is 28 days
                    
                    result = await self.timeout_user(guild_id, target_id, duration)
                    if result:
                        if duration >= 1440: timeout_str_fmt = f"{duration // 1440} day(s)"
                        elif duration >= 60: timeout_str_fmt = f"{duration // 60} hour(s)"
                        else: timeout_str_fmt = f"{duration} minute(s)"
                        return f"Okay~! I've timed out <@{target_id}> for {timeout_str_fmt}! Tee-hee! ✨"
                    else:
                        return "Aww, I couldn't timeout that user... 😥 Maybe I don't have the 'Moderate Members' permission, or they have a higher role than me, or they're not in the server?"
                except ValueError:
                    return "Hmm, that doesn't look like a valid number for the timeout duration or user ID."
            else:
                return "I'd love to help with that, but I don't seem to have the 'Moderate Members' permission here! 😟"

        elif search_match:
            query = search_match.group(1).strip()
            search_results_text = await self.search_internet(query)
            # Modify prompt to include search results for the AI to synthesize
            prompt += f"\n\n[System Note: I just searched the internet for '{query}'. Use the following information (if relevant and reliable) to answer the user's request naturally as Kasane Teto. Do not just repeat the results verbatim. If the search didn't yield useful info, say so or try to answer from your own knowledge.]\nSearch Information:\n{search_results_text}"
            # Let the normal AI generation process handle the response synthesis

        # --- Prepare context for AI ---
        user_facts = self.get_user_facts(user_id_str)
        user_memory_str = "You don't seem to remember any specific facts about this user yet."
        if user_facts:
            facts_list = "\n".join([f"- {fact}" for fact in user_facts])
            user_memory_str = f"Here's what you remember about {user_name} (User ID: {user_id_str}):\n{facts_list}"

        manual_context_str = "\n".join([f"- {item}" for item in self.manual_context]) if self.manual_context else "None provided."
        dynamic_learning_str = "\n".join([f"- {item}" for item in self.dynamic_learning]) if self.dynamic_learning else "None provided."
        
        current_date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        system_context = self.system_prompt_template.format(
            current_date=current_date_str,
            manual_context=manual_context_str,
            dynamic_learning_context=dynamic_learning_str,
            user_memory_context=user_memory_str,
            user_name=user_name,
            user_id=user_id_str
        )

        history_messages = self.get_user_history(user_id_str)
        
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_context}]
        messages.extend(history_messages)
        # Add current prompt, ensuring it's not a duplicate if it was modified (e.g., by search)
        # The prompt variable already contains modifications if search_match was true
        current_user_message = {"role": "user", "content": f"{user_name} (User ID: {user_id_str}): {prompt}"}
        messages.append(current_user_message)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Meta Llama API might require specific headers, e.g., for user agent or referrals.
            # "HTTP-Referer": "https://your-discord-bot-project-url.com", # Optional, verify if needed
            # "X-Title": "Kasane Teto Discord Bot" # Optional, verify if needed
        }

        max_tool_iterations = 5 # Prevent infinite loops of tool calls
        for iteration in range(max_tool_iterations):
            payload = {
                "model": config["model"],
                "messages": messages,
                # IMPORTANT: Verify how Meta Llama API expects tools. It might be 'tools' or 'functions'.
                # Also, the structure of each tool definition needs to match their schema.
                "tools": self.tools if self.tools else None, # Send tools if defined
                "temperature": config.get("temperature"),
                # IMPORTANT: Meta Llama API might use 'max_gen_len' or 'max_tokens' or similar.
                "max_tokens": config.get("max_tokens"), 
                "top_p": config.get("top_p"),
                # Verify if these penalties are supported and their names are correct
                "frequency_penalty": config.get("frequency_penalty"),
                "presence_penalty": config.get("presence_penalty"),
                # "repetition_penalty": config.get("repetition_penalty"), # If supported
                # "tool_choice": "auto", # Or "none", or specific tool. Verify Meta Llama's equivalent.
            }
            # Remove None values from payload, as some APIs are strict
            payload = {k: v for k, v in payload.items() if v is not None}
            if not payload.get("tools"): # If no tools, remove the key
                payload.pop("tools", None)


            print(f"--- Iteration {iteration + 1} ---")
            # print(f"Sending payload to AI: {json.dumps(payload, indent=2)}") # For debugging

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.api_url, headers=headers, json=payload, timeout=90.0) as response: # Increased timeout
                        if response.status == 200:
                            data = await response.json()
                            # print(f"Received data from AI: {json.dumps(data, indent=2)}") # For debugging

                            if not data.get("choices") or not data["choices"][0].get("message"):
                                print(f"API Error: Unexpected response format. Data: {data}")
                                return f"Sorry {user_name}, I got an unexpected response from the AI. Maybe try again?"
                            
                            response_message = data["choices"][0]["message"]
                            # IMPORTANT: Verify how Meta Llama API indicates finish reason (e.g., 'stop', 'tool_calls', 'length')
                            finish_reason = data["choices"][0].get("finish_reason", "stop") # Default to 'stop' if not provided

                            # Append assistant's current thought/response to messages list for context,
                            # even if it's a tool call request.
                            messages.append(response_message)

                            # --- Tool Call Handling ---
                            # IMPORTANT: Verify how Meta Llama API returns tool calls.
                            # It might be `response_message.get("tool_calls")` or `response_message.get("function_call")`.
                            # The structure of tool_calls objects also needs to be verified.
                            if response_message.get("tool_calls") and (finish_reason == "tool_calls" or finish_reason == "tool_use"): # Adjust finish_reason as per API docs
                                tool_calls = response_message["tool_calls"]
                                print(f"AI requested tool calls: {tool_calls}")

                                for tool_call in tool_calls:
                                    # IMPORTANT: Verify the structure of a tool_call object.
                                    # It might be `tool_call.function.name` and `tool_call.function.arguments`.
                                    function_name = tool_call.get("function", {}).get("name")
                                    tool_call_id = tool_call.get("id") # OpenAI style
                                    
                                    if not function_name or not tool_call_id:
                                        print(f"Invalid tool call structure: {tool_call}")
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call_id or "unknown_tool_id",
                                            "content": "Error: Received an invalid tool call structure from AI."
                                        })
                                        continue # Next tool call or next iteration

                                    try:
                                        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                                        arguments = json.loads(arguments_str)
                                        
                                        tool_result_content = f"Error: Tool '{function_name}' not implemented or failed."
                                        if function_name == "run_safe_shell_command":
                                            command_to_run = arguments.get("command")
                                            if command_to_run:
                                                tool_result_content = await self.run_shell_command(command_to_run)
                                            else:
                                                tool_result_content = "Error: No command provided for run_safe_shell_command."
                                        
                                        elif function_name == "remember_fact_about_user":
                                            fact_user_id = arguments.get("user_id")
                                            fact_to_remember = arguments.get("fact")
                                            # Security: Ensure AI is trying to remember for the *current* interacting user,
                                            # or handle appropriately if it's for someone else (might require confirmation).
                                            if fact_user_id == user_id_str and fact_to_remember:
                                                self.add_user_fact(fact_user_id, fact_to_remember)
                                                tool_result_content = f"Successfully remembered fact for user {fact_user_id} ({user_name}): '{fact_to_remember}'"
                                            elif not fact_user_id or not fact_to_remember:
                                                tool_result_content = "Error: Missing user_id or fact for remember_fact_about_user."
                                            else: # AI trying to remember for a different user ID
                                                tool_result_content = f"Error: Tool 'remember_fact_about_user' was called with user ID '{fact_user_id}', but current interaction is with '{user_id_str}'. This action was blocked for safety. Please confirm the correct user ID."
                                        else:
                                            tool_result_content = f"Error: Unknown tool function '{function_name}' requested by AI."

                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call_id,
                                            "name": function_name, # Some APIs might need 'name' here too
                                            "content": str(tool_result_content), # Ensure content is string
                                        })
                                    except json.JSONDecodeError:
                                        print(f"Error decoding tool arguments for {function_name}: {arguments_str}")
                                        messages.append({
                                            "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                                            "content": f"Error: Invalid JSON arguments for {function_name}."})
                                    except Exception as e_tool:
                                        print(f"Error executing tool {function_name}: {e_tool}")
                                        messages.append({
                                            "role": "tool", "tool_call_id": tool_call_id, "name": function_name,
                                            "content": f"Error: An unexpected error occurred while running tool {function_name}: {e_tool}"})
                                # After processing all tool calls in this batch, continue to the next API call iteration
                                continue # To the top of the for loop for next API call

                            # --- No Tool Calls, or Finished After Tool Calls ---
                            elif response_message.get("content"):
                                final_response = response_message["content"].strip()
                                print(f"AI Response for {user_name} (iter {iteration+1}): {final_response[:150]}...")

                                # Add interaction to history (both user prompt and AI final response)
                                # The user prompt added to history should be the original one, or the one augmented by search.
                                # current_user_message['content'] holds the full prompt sent to the AI in this turn.
                                self.add_to_history(user_id_str, "user", current_user_message['content'])
                                self.add_to_history(user_id_str, "assistant", final_response)
                                return final_response
                            
                            else: # No content and no tool calls, or unexpected finish_reason
                                print(f"API Error: No content and no tool calls in response, or unexpected finish_reason '{finish_reason}'. Data: {data}")
                                if iteration < max_tool_iterations -1 and finish_reason != "stop": # If not stop, maybe it needs to continue
                                    messages.append({"role":"user", "content":"[System Note] Please provide a textual response or use a tool correctly."})
                                    continue # Try one more time if not the last iteration
                                return "Hmm, I seem to have lost my train of thought... Can you ask again, or maybe rephrase?"

                        elif response.status == 401: # Unauthorized
                             print(f"API Error: Unauthorized (401). Check your API Key. Response: {await response.text()}")
                             return "Oh dear! It seems my connection to the AI brain is not authorized. My owner needs to check the API key! 🔑"
                        elif response.status == 429: # Rate limit
                             print(f"API Error: Rate limited (429). Response: {await response.text()}")
                             await asyncio.sleep(5) # Wait before allowing another try
                             return "Woah there, too fast! I'm getting a bit overwhelmed. Please wait a moment and try again. 🐢"
                        else: # Handle other HTTP errors
                            error_text = await response.text()
                            print(f"API Error: {response.status} - {error_text}")
                            return f"Oops! I encountered an issue trying to connect to my AI brain (Error {response.status}). Please try again later. If this persists, my owner should check the logs!"
            
            except aiohttp.ClientConnectorError as e:
                print(f"Network Error: Could not connect to API at {self.api_url}. Error: {e}")
                return "I'm having trouble reaching my AI brain right now due to a network issue. Please check your connection or try again later."
            except asyncio.TimeoutError:
                print(f"API call timed out after 90 seconds for user {user_name}.")
                return "Phew, that took a while and I couldn't get a response in time! Could you try asking something a bit simpler, or try again in a moment?"
            except Exception as e:
                print(f"An unexpected error occurred during AI response generation: {e}")
                # Log the full traceback for debugging
                import traceback
                traceback.print_exc()
                return "Something went unexpectedly haywire while I was thinking! 🤯 My owner should check my console for the gory details."

        # If loop finishes without returning (e.g., too many tool iterations)
        print(f"Max tool iterations ({max_tool_iterations}) reached for user {user_name}.")
        self.add_to_history(user_id_str, "user", current_user_message['content']) # Log user prompt
        self.add_to_history(user_id_str, "assistant", "[System Error: Max tool iterations reached]") # Log error
        return "I tried my best with the tools, but it got a bit complicated. Could you simplify your request or try a different approach?"

    # --- Discord Slash Commands ---
    @app_commands.command(name="chat", description="Chat with Kasane Teto (AI)!")
    @app_commands.describe(message="Your message to Teto.")
    async def chat_command(self, interaction: discord.Interaction, message: str):
        """Handles the /chat command."""
        if not self.api_key:
            await interaction.response.send_message("AI features are currently disabled as the API key is not configured. Please contact the bot owner.", ephemeral=True)
            return

        await interaction.response.defer(thinking=True) # Indicate bot is working
        
        user_id = str(interaction.user.id)
        user_name = interaction.user.display_name

        try:
            response_text = await self.generate_response(user_id, user_name, message, source_interaction=interaction)
            
            # Send response in chunks if too long
            if len(response_text) > 2000:
                for i in range(0, len(response_text), 1990): # 1990 to be safe with "..."
                    chunk = response_text[i:i+1990]
                    if i == 0:
                        await interaction.followup.send(chunk)
                    else:
                        await interaction.channel.send(chunk) # Send subsequent chunks as new messages
            else:
                await interaction.followup.send(response_text)

        except Exception as e:
            print(f"Error in /chat command for user {user_name} ({user_id}): {e}")
            import traceback
            traceback.print_exc()
            await interaction.followup.send("Oh no! Something went wrong while I was trying to respond. My circuits are buzzing! 😵 Please try again.")

    # --- Discord Event Listeners ---
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handles messages to respond when the bot is pinged."""
        # Ignore messages from the bot itself
        if message.author == self.bot.user:
            return

        # Check if the bot is mentioned in the message
        # message.mentions contains a list of User/Member objects mentioned
        if self.bot.user in message.mentions:
            # Extract the message content after the mention
            # This regex finds the bot mention at the start and captures the rest
            mention_regex = re.compile(rf"<@!?{self.bot.user.id}>\s*")
            content = mention_regex.sub("", message.content).strip()

            if not content:
                # If only the bot was mentioned with no text, maybe send a default greeting or ignore
                # For now, let's ignore empty messages after mention
                return

            # Process the message content using the AI
            user_id = str(message.author.id)
            user_name = message.author.display_name

            # Indicate bot is thinking (optional for message listener, but good practice)
            # await message.channel.trigger_typing() # This can be spammy in busy channels

            try:
                response_text = await self.generate_response(user_id, user_name, content, source_message=message)

                # Send response in chunks if too long
                if len(response_text) > 2000:
                    for i in range(0, len(response_text), 1990):
                        chunk = response_text[i:i+1990]
                        await message.channel.send(chunk)
                else:
                    await message.channel.send(response_text)

            except Exception as e:
                print(f"Error in on_message for user {user_name} ({user_id}): {e}")
                import traceback
                traceback.print_exc()
                await message.channel.send("Oh no! Something went wrong while I was trying to respond to your mention. 😵")

        # Allow other message processing (e.g., commands) to continue
        await self.bot.process_commands(message)


    # --- Discord Slash Commands ---
    @app_commands.command(name="setaiconfig", description="Set your personal AI model configuration (e.g., temperature).")
    @app_commands.describe(
        model="Model identifier (optional, consult bot owner for available models)",
        temperature="Model temperature (0.0-2.0, e.g., 0.7)",
        max_tokens="Max tokens for response (e.g., 1000)",
        top_p="Top P sampling (0.0-1.0, e.g., 0.9)"
    )
    async def set_ai_config_command(self, interaction: discord.Interaction, 
                                    model: Optional[str] = None, 
                                    temperature: Optional[app_commands.Range[float, 0.0, 2.0]] = None,
                                    max_tokens: Optional[app_commands.Range[int, 50, 4000]] = None, # Adjust range as per API limits
                                    top_p: Optional[app_commands.Range[float, 0.0, 1.0]] = None):
        """Allows users to set their AI configuration."""
        user_id = str(interaction.user.id)
        current_config = self.get_user_config(user_id) # Gets a copy

        updated = False
        if model is not None:
            # Potentially validate model against a list of allowed/known models if you want to restrict this
            current_config["model"] = model
            updated = True
        if temperature is not None:
            current_config["temperature"] = temperature
            updated = True
        if max_tokens is not None:
            # IMPORTANT: Ensure 'max_tokens' is the correct parameter name for Meta Llama API
            current_config["max_tokens"] = max_tokens 
            updated = True
        if top_p is not None:
            current_config["top_p"] = top_p
            updated = True

        if updated:
            self.user_configs[user_id] = current_config # Update the stored config
            self.save_configs()
            await interaction.response.send_message("Your AI configuration has been updated! ✨", ephemeral=True)
        else:
            await interaction.response.send_message("No changes were made. Provide at least one parameter to update.", ephemeral=True)

    @app_commands.command(name="getaiconfig", description="View your current AI model configuration.")
    async def get_ai_config_command(self, interaction: discord.Interaction):
        """Shows the user their current AI configuration."""
        user_id = str(interaction.user.id)
        config = self.get_user_config(user_id)
        config_str = "\n".join([f"**{k}**: {v}" for k, v in config.items()])
        embed = discord.Embed(title=f"{interaction.user.display_name}'s AI Configuration", description=config_str, color=discord.Color.blue())
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="addfact", description="Manually add a fact for the AI to remember about you.")
    @app_commands.describe(fact="The fact to remember (e.g., 'My favorite food is bread').")
    async def add_fact_command(self, interaction: discord.Interaction, fact: str):
        """Allows a user to manually add a fact to their own memory."""
        user_id = str(interaction.user.id)
        user_name = interaction.user.display_name
        self.add_user_fact(user_id, fact)
        await interaction.response.send_message(f"Okay, {user_name}, I'll try to remember that: '{fact}'", ephemeral=True)

    @app_commands.command(name="viewfacts", description="View the facts the AI remembers about you.")
    async def view_facts_command(self, interaction: discord.Interaction):
        """Allows a user to view their remembered facts."""
        user_id = str(interaction.user.id)
        user_name = interaction.user.display_name
        facts = self.get_user_facts(user_id)
        if facts:
            facts_str = "\n".join([f"- {f}" for f in facts])
            embed = discord.Embed(title=f"Facts I Remember About {user_name}", description=facts_str, color=discord.Color.green())
        else:
            embed = discord.Embed(title=f"Facts I Remember About {user_name}", description="I don't seem to have any specific facts stored for you yet!", color=discord.Color.orange())
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(name="addcontext", description="[Admin] Add a global manual context string for the AI.")
    @app_commands.describe(text="The context string to add.")
    @app_commands.checks.has_permissions(administrator=True) # Example permission
    async def add_manual_context_command(self, interaction: discord.Interaction, text: str):
        if self.add_manual_context(text):
            await interaction.response.send_message(f"Added to manual context: '{text[:100]}...'", ephemeral=True)
        else:
            await interaction.response.send_message("Context already exists or is empty.", ephemeral=True)
    
    @add_manual_context_command.error
    async def add_manual_context_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("Sorry, you don't have permission to use this command!", ephemeral=True)
        else:
            await interaction.response.send_message(f"An error occurred: {error}", ephemeral=True)


    @app_commands.command(name="adddynamic", description="[Admin] Add a global dynamic learning example for the AI.")
    @app_commands.describe(text="The dynamic learning example (e.g., 'User asks for help -> Offer assistance politely').")
    @app_commands.checks.has_permissions(administrator=True) # Example permission
    async def add_dynamic_learning_command(self, interaction: discord.Interaction, text: str):
        if self.add_dynamic_learning(text):
            await interaction.response.send_message(f"Added to dynamic learning: '{text[:100]}...'", ephemeral=True)
        else:
            await interaction.response.send_message("Dynamic learning example already exists or is empty.", ephemeral=True)

    @add_dynamic_learning_command.error
    async def add_dynamic_learning_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message("Sorry, you don't have permission to use this command!", ephemeral=True)
        else:
            await interaction.response.send_message(f"An error occurred: {error}", ephemeral=True)


async def setup(bot: commands.Bot):
    # Ensure data directory exists upon setup
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"Created data directory during setup: {DATA_DIR}")
        except OSError as e:
            print(f"FATAL during setup: Could not create data directory {DATA_DIR}. Error: {e}")
            # Depending on strictness, you might prevent the cog from loading
            # raise e 
    
    # Create default files if they don't exist, with empty JSON structures
    default_files_to_check = {
        DEFAULT_MEMORY_PATH: {},
        DEFAULT_HISTORY_PATH: {},
        DEFAULT_MANUAL_CONTEXT_PATH: [],
        DEFAULT_DYNAMIC_LEARNING_PATH: [],
        DEFAULT_CONFIG_PATH: {}
    }
    for file_path, default_content in default_files_to_check.items():
        if not os.path.exists(file_path):
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(default_content, f, indent=4)
                print(f"Created default empty file: {file_path}")
            except Exception as e:
                print(f"Error creating default file {file_path}: {e}")

    await bot.add_cog(AICog(bot))
    print("AICog loaded successfully.")
