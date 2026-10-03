"""NCP-AAI W1 D5 (AAI Day 1): one tool, one call, one answer.

The smallest possible agent step: the model DECIDES to call get_alert,
your code ACTS, the model reads the OBSERVATION.
Source: https://docs.nvidia.com/nim/large-language-models/1.15.0/function-calling.html
"""
import json, os
from pathlib import Path
from openai import OpenAI

# Reuse the repo's shared key file (gitignored) unless the var is already set.
_ENV = Path(__file__).resolve().parents[2] / "00-environment" / ".env"
if _ENV.exists():
    for line in _ENV.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

client = OpenAI(base_url=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                api_key=os.environ["NVIDIA_API_KEY"])
MODEL = os.environ.get("AGENT_MODEL", "nvidia/nemotron-3-super-120b-a12b")

ALERTS = {"A-1001": {"customer": "Northwind Trading", "pattern": "cash deposits",
                     "deposits": [9500, 9400, 9600], "window_days": 7}}   # synthetic
TOOLS = [{"type": "function", "function": {
    "name": "get_alert",
    "description": "Fetch a transaction-monitoring alert by id. Call it whenever the user mentions an alert id.",
    "parameters": {"type": "object",
                   "properties": {"alert_id": {"type": "string", "description": "e.g. A-1001"}},
                   "required": ["alert_id"]}}}]

question = [{"role": "user", "content": "Summarise alert A-1001 in one line."}]
msg = client.chat.completions.create(model=MODEL, messages=question, tools=TOOLS, tool_choice="auto",
                                     temperature=1.0, top_p=0.95).choices[0].message
print("tool_calls:", msg.tool_calls)

if msg.tool_calls:                                   # act, then observe
    call = msg.tool_calls[0]
    alert_id = json.loads(call.function.arguments)["alert_id"]
    obs = json.dumps(ALERTS.get(alert_id, {"error": "not found"}))
    final = client.chat.completions.create(
        model=MODEL, tools=TOOLS, temperature=1.0, top_p=0.95,
        messages=question + [msg, {"role": "tool", "tool_call_id": call.id, "content": obs}]).choices[0].message
    print("answer:", final.content)
