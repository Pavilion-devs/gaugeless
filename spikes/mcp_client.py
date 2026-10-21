import asyncio, json, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    p = StdioServerParameters(command=".venv/bin/python", args=["-W", "ignore", "-m", "gaugeless.mcp_server"])
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as c:
            await c.initialize()
            tools = await c.list_tools(); print("tools:", [t.name for t in tools.tools])
            def j(res): return json.loads(res.content[0].text)
            sid = j(await c.call_tool("open_session", {"site": "lamorge"}))["session_id"]
            for name, args in [("load_clip", {}), ("diagnose_clip", {}), ("measure_velocity", {"method": "klt"}),
                               ("compute_discharge", {}), ("publish", {})]:
                res = j(await c.call_tool(name, {"session_id": sid, "args": args}))
                print(name, {k: v for k, v in res.items() if k != "sections"})
asyncio.run(main())
