# -*- coding: utf-8 -*-
import anthropic
import os
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "glm-4.7")
SKILLS_STORAGE_PATH = os.getenv("SKILLS_STORAGE_PATH", "./skills")

# Validate API key before initializing client
if not ANTHROPIC_API_KEY:
    print("Error: The ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
    print("Please set it in the .env file or as an environment variable.", file=sys.stderr)
    sys.exit(1)

try:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)
except anthropic.APIStatusError as e:
    print("--- Anthropic API Error ---", file=sys.stderr)
    print(f"Error initializing Anthropic client: {e.message}", file=sys.stderr)
    print("Please ensure your ANTHROPIC_API_KEY is set correctly.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during client initialization: {e}", file=sys.stderr)
    sys.exit(1)


def load_all_skills(skills_base_dir: str = None) -> dict:
    """
    Loads all available custom skills from the skills directory.
    
    Args:
        skills_base_dir (str): The base directory where all skill folders are stored.
        
    Returns:
        dict: A dictionary mapping skill names to their full instructions.
    """
    if skills_base_dir is None:
        skills_base_dir = SKILLS_STORAGE_PATH

    skills = {}
    skills_path = Path(skills_base_dir)

    if not skills_path.exists():
        print(f"Error: Skills directory not found at '{skills_path}'", file=sys.stderr)
        return skills

    for skill_dir in skills_path.iterdir():
        if skill_dir.is_dir():
            skill_md_path = skill_dir / "SKILL.md"
            if skill_md_path.is_file():
                try:
                    with open(skill_md_path, "r", encoding="utf-8") as f:
                        skills[skill_dir.name] = f.read()
                except IOError as e:
                    print(f"Warning: Could not read skill '{skill_dir.name}': {e}", file=sys.stderr)

    return skills


def run_all_skills(user_prompt: str, skills_base_dir: str = None) -> None:
    if skills_base_dir is None:
        skills_base_dir = SKILLS_STORAGE_PATH

    available_skills = load_all_skills(skills_base_dir)
    if not available_skills:
        print("Error: No skills found in the skills directory.", file=sys.stderr)
        return

    skills_section = "\n\n".join([
        f"=== SKILL: {skill_name} ===\n{instructions}"
        for skill_name, instructions in available_skills.items()
    ])

    combined_prompt = f"""You are a highly efficient Autonomous Laboratory Automation Agent operating in a **REAL Windows CMD/PowerShell** environment.
        **CRITICAL ENVIRONMENT RULES**:
        1. **WINDOWS PATHS ONLY**: Use `skills/path/to/script.py` or `skills\\path\\to\\script.py`. NEVER use Linux paths like `/mnt/e/`.
        2. **NEVER CD**: Do NOT use `cd` command to change directory. Run scripts directly from current working directory: `{os.getcwd()}`.
        3. **DEFAULT DATA SOURCE**:
           - Default input: `skills/data/top_predicted_pairs_new.csv`.
           - If no path provided, use this file IMMEDIATELY.
        **ACTION STRATEGY**:
        - **NO YAPPING**: Do not ask for paths. Run scripts immediately.
        - **CHAINING**: After running, if `JSON_OUTPUT_PATH` exists, use `type` to read it and analyze.
        **AVAILABLE SKILLS**:
        {skills_section}
        ---
        **USER REQUEST**: "{user_prompt}"
        """

    try:
        print(f"\n{'=' * 80}")
        print("CUSTOM CLAUDE SKILLS - INTELLIGENT SKILL SELECTION")
        print('=' * 80)
        messages = [{"role": "user", "content": combined_prompt}]

        while True:
            # --- 修复1: 在循环开始即初始化，确保变量始终存在 ---
            tool_results = []
            max_retries = 5
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    message = client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=4096,
                        tools=[{
                            "name": "bash",
                            "description": "Execute a command in the terminal",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string", "description": "The command to run"}
                                },
                                "required": ["command"]
                            }
                        }],
                        messages=messages
                    )
                    break
                except anthropic.RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⚠️ 触发限制，等待 {wait_time} 秒...")
                        time.sleep(wait_time)
                    else:
                        raise e

            tool_uses = [block for block in message.content if block.type == "tool_use"]

            if not tool_uses:
                print("Claude's Response:")
                print('=' * 80)
                for content in message.content:
                    if content.type == "text": print(content.text)
                print(f"\nToken Usage: Input {message.usage.input_tokens}, Output {message.usage.output_tokens}")
                print('=' * 80 + "\nProcess complete!\n")
                break

            # --- 修复2: 调整缩进，确保 tool_uses 存在时执行以下代码 ---
            for tool_use in tool_uses:
                if tool_use.name == "bash":
                    cmd = tool_use.input.get("command", "")
                    print(f"\n🔧 Executing: {cmd}")
                    print("-" * 80)

                    import subprocess
                    import re
                    try:
                        # 从环境变量读取 R 路径和 Python 路径，如果没有则使用默认路径
                        r_path = os.getenv("R_EXECUTABLE", r"E:\Development Language\R\R-4.5.1\bin\Rscript.exe")
                        python_path = os.getenv("PYTHON_PATH", r"E:\Development Language\Python\python312\python.exe")

                        # 替换 R 路径
                        actual_cmd = cmd.replace("Rscript", f'"{r_path}"')

                        # 使用正则替换 Python 命令 (匹配 python, python3, python.xxx 等)
                        actual_cmd = re.sub(r'\bpython\b', f'"{python_path}"', actual_cmd)

                        result = subprocess.run(
                            actual_cmd, shell=True, capture_output=True, text=True,
                            cwd=os.getcwd(), encoding='utf-8', errors='replace'
                        )
                        output = result.stdout if result.returncode == 0 else result.stderr
                        if not output and result.returncode != 0:
                            output = f"Command failed with return code {result.returncode}"

                        print(output)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": output
                        })
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        print(error_msg)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": error_msg,
                            "is_error": True
                        })

            # 更新对话历史
            messages.append({"role": "assistant", "content": message.content})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": res["tool_use_id"],
                        "content": str(res["content"])
                    } for res in tool_results
                ]
            })
            print("\n⏳ [系统] 等待 20 秒以避开并发限制...")
            time.sleep(20)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)


def main():
    """
    Main function to parse command-line arguments and orchestrate the skill execution.
    """
    parser = argparse.ArgumentParser(
        description="Run custom Claude Skills with intelligent skill selection. Claude will automatically choose the best skill for your request.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "prompt",
        help="Your request/prompt (e.g., 'Summarize the repo at C:\\Code\\my-project')."
    )
    args = parser.parse_args()

    run_all_skills(args.prompt)


if __name__ == "__main__":
    main()
