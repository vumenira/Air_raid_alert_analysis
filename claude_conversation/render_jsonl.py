"""Render raw Claude Code session files (~/.claude/projects/<project>/<session>.jsonl) as plain text.

Unlike a terminal copy, this keeps every tool call and its full output (long outputs are capped),
so the numbers the assistant was looking at stay visible. Thinking blocks are left out.

Usage: python claude_conversation/render_jsonl.py SESSION.jsonl [SESSION2.jsonl ...] -o OUT.txt [--max-lines 150]
"""
import argparse
import json
from pathlib import Path


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b["text"])
        elif b.get("type") == "image":
            parts.append("[image]")
    return "\n".join(parts)


def cap(s: str, max_lines: int) -> str:
    lines = s.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + [f"[... {len(lines) - max_lines} more lines truncated]"])


def indent(s: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in s.splitlines())


def tool_call(b: dict) -> str:
    name, inp = b["name"], b.get("input", {})
    if name == "Bash":
        body = inp.get("command", "")
    elif name in ("Write",):
        body = f"{inp.get('file_path')}\n{inp.get('content', '')}"
    elif name == "Edit":
        body = f"{inp.get('file_path')}\n--- old\n{inp.get('old_string', '')}\n+++ new\n{inp.get('new_string', '')}"
    else:
        body = json.dumps(inp, ensure_ascii=False, indent=1)
    return f"[tool call: {name}]\n{indent(body)}"


def render(path: Path, max_lines: int) -> list[str]:
    out = [f"######## session {path.stem} ########"]
    for line in path.open():
        d = json.loads(line)
        if d.get("type") not in ("user", "assistant") or d.get("isSidechain"):
            continue
        msg, ts = d.get("message", {}), d.get("timestamp", "")[:19].replace("T", " ") + " UTC"
        content = msg.get("content")
        if d["type"] == "user":
            if d.get("isMeta"):
                continue
            if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
                for b in content:
                    if b.get("type") == "tool_result":
                        res = text_of(b.get("content")) if not isinstance(b.get("content"), str) else b["content"]
                        tag = "tool error" if b.get("is_error") else "tool output"
                        out.append(f"[{tag}]\n{indent(cap(res, max_lines))}")
                continue
            text = text_of(content).strip()
            if text and not text.startswith("<local-command") and not text.startswith("<command-"):
                out.append(f"\n==== USER  ({ts}) ====\n{text}")
            elif "<command-name>" in text:
                out.append(f"\n==== USER  ({ts}) ====\n{text.split('<command-name>')[1].split('<')[0]}")
        else:
            for b in content or []:
                if b.get("type") == "text" and b["text"].strip():
                    out.append(f"\n==== CLAUDE ({ts}) ====\n{b['text'].strip()}")
                elif b.get("type") == "tool_use":
                    out.append(tool_call(b))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--max-lines", type=int, default=150)
    args = ap.parse_args()
    lines = [f"Rendered from raw Claude Code session files by claude_conversation/render_jsonl.py "
             f"(tool outputs capped at {args.max_lines} lines; thinking blocks omitted).\n"]
    for p in args.sessions:
        lines += render(p, args.max_lines) + [""]
    args.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
