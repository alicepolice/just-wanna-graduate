#!/usr/bin/env python3
"""
测试 claude-agent-sdk 替换 _call_claude 的可行性。
用简单 prompt 验证流式文字 + 工具调用事件。
"""
import asyncio
import sys

from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import (
    AssistantMessage, StreamEvent, ResultMessage, SystemMessage,
    TextBlock,
)

# 模拟 skill_runner 的参数
SYSTEM = "你是一个助手，回答要简短。"
PROMPT = "用 python 写一行打印 hello world 的代码"
CWD = "/home/exp/just-wanna-graduate"


async def main():
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM,
        cwd=CWD,
        permission_mode="auto",
        model="claude-sonnet-4-6",
        effort="high",
        include_partial_messages=True,
    )

    full_text = []

    async for msg in query(prompt=PROMPT, options=options):
        if isinstance(msg, StreamEvent):
            event = msg.event
            etype = event.get("type", "")

            # 流式文字
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    sys.stdout.write(delta.get("text", ""))
                    sys.stdout.flush()

            # 工具调用开始
            elif etype == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    sys.stdout.write(f"\n[tool:{block.get('name')}] ")
                    sys.stdout.flush()

            # 工具 input 流式拼接（可选打印）
            elif etype == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "input_json_delta":
                    sys.stdout.write(delta.get("partial_json", ""))
                    sys.stdout.flush()

        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    full_text.append(block.text)

        elif isinstance(msg, ResultMessage):
            print(f"\n\n[done] turns={msg.num_turns} cost=${msg.total_cost_usd:.4f}")

    print("\n完整输出:", "".join(full_text)[:300])


asyncio.run(main())
