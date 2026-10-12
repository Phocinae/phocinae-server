#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_wheel_contents.py — 在发布**之前**检查刚构建出来的 wheel 里到底有什么。

**为什么必须单独做这一步**：四个 release workflow 都是"跑源码树里的测试 →
构建 → 上传"。而本项目栽过的三个缺陷**全部**是"源码树全绿、装上去是坏的"：

  1. `phocinae-mcp 0.2.2` —— wheel 里没带 L0 表，`pip install` 后每条命令都被拒
     （源码树里旁边有 `../phocinae-guard/guard.py`，所以源码测试永远看不到）
  2. `phocinae-cli 0.1.1` —— wheel 里是手抄的 guard 0.1.0，四个 shell 绕过敞着
  3. `phocinae-guard 0.1.3` —— 白名单里只有 `python3 -m pytest`（Windows 上没有）

这三个都不是"逻辑错"，而是**打出来的包里装的东西不对**。所以发布前必须有人
**打开那个 wheel 看一眼**。`tools/verify_published.py` 做的是同一件事，但它要等
包已经上了 PyPI —— 那时已经晚了。

本脚本检查（全部基于 wheel 内容，不需要网络、不装依赖）：

  * wheel 能被解析，含 `.dist-info/METADATA`
  * `Name` / `Version` 与期望一致（防止 tag 与 pyproject 不一致）
  * **上一版曾经缺失的关键数据文件存在**（每个包用 `REQUIRE` 声明）
  * 没有把测试、缓存、`.pyc` 打进包里
  * 纯 Python 包不应带平台特定 tag（`py3-none-any` 之外要说明理由）

每个包的 `REQUIRE` 就是"这个包最容易漏掉的那个文件"，来自上面三次事故。
"""
import argparse
import glob
import io
import os
import re
import sys
import zipfile

# 包名 -> 必须存在的内部路径（相对 wheel 根）。
# 这些是**踩过的坑**，不是泛泛的完整性检查：
#   phocinae_mcp/l0_table.json  —— 0.2.2 漏了它，装完每条命令都被拒
REQUIRE = {
    "phocinae-mcp": ["phocinae_mcp/l0_table.json", "phocinae_mcp/l0.py"],
    "phocinae-guard": ["phocinae_guard/guard.py"],
    "phocinae-cli": ["phocinae_cli/l0.py"],
    "phocinae-server": ["phocinae/engine.py"],
}

FORBID = ("/tests/", "test_", "__pycache__", ".pyc", ".egg-info",
          "/.github/", ".pytest_cache", "/poc/", "/examples/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wheel", help="path to the built .whl (globs allowed)")
    ap.add_argument("--expect-name", required=True)
    ap.add_argument("--expect-version", default=os.environ.get("GITHUB_REF_NAME", ""),
                    help="defaults to GITHUB_REF_NAME with a leading v stripped")
    args = ap.parse_args()

    paths = glob.glob(args.wheel)
    if not paths:
        print("FAIL no wheel matched %r" % args.wheel)
        return 1
    whl = paths[0]
    print("wheel: %s" % os.path.basename(whl))

    expect_ver = args.expect_version.lstrip("v") if args.expect_version else ""
    # 手工 workflow_dispatch 时 github.ref_name 是**分支名**（如 "main"），不是版本号。
    # 把它当版本比较会在每次手工重发时稳定误报"版本不匹配"—— 而手工重发正是修好
    # 凭据后要走的路。识别出这种情况就跳过比较并说明，而不是报错。
    if expect_ver and not re.match(r"^\d+\.\d+", expect_ver):
        print("  NOTE --expect-version %r is not a version (a manual dispatch passes the"
              " branch name);" % expect_ver)
        print("       skipping the tag/version comparison. A tag push compares it.")
        expect_ver = ""
    fails = []

    def check(name, cond, detail=""):
        print(("  PASS " if cond else "  FAIL ") + name + ("" if cond else "  " + str(detail)))
        if not cond:
            fails.append(name)

    try:
        z = zipfile.ZipFile(whl)
        names = z.namelist()
    except zipfile.BadZipFile as e:
        print("FAIL not a readable zip: %s" % e)
        return 1

    metas = [n for n in names if n.endswith(".dist-info/METADATA")]
    check("wheel contains METADATA", len(metas) == 1, metas)
    if not metas:
        return 1
    meta = z.read(metas[0]).decode("utf-8", "replace")
    f_name = ""
    f_ver = ""
    for line in meta.split("\n"):
        if line.lower().startswith("name: "):
            f_name = line[6:].strip()
        elif line.lower().startswith("version: "):
            f_ver = line[9:].strip()
    check("METADATA Name == %s" % args.expect_name, f_name == args.expect_name,
          "got %r" % f_name)
    if expect_ver:
        check("METADATA Version == %s" % expect_ver, f_ver == expect_ver,
              "got %r  (a tag that does not match pyproject publishes the wrong version)"
              % f_ver)
    else:
        print("  NOTE no expected version supplied; skipping the tag/version comparison")

    # 关键数据文件 —— 这一条就是 0.2.2 那次事故的守卫
    for need in REQUIRE.get(args.expect_name, []):
        check("ships %s" % need, need in names,
              "not in the wheel; this is the packaging defect that shipped once already")

    bad = [n for n in names if any(f in n for f in FORBID)]
    check("no tests / caches / bytecode inside", not bad, bad[:6])

    tags = os.path.basename(whl).split("-")[-3:]
    check("pure-python tag (py3-none-any)", "py3-none-any" in os.path.basename(whl),
          "tags=%s  (a pure-python package should not be platform-tagged)" % tags)

    print("\n  entries: %d" % len(names))
    print()
    if fails:
        print("verify_wheel_contents: FAILED %d check(s)" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("verify_wheel_contents: the assembled wheel is what we intend to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
