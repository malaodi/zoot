from __future__ import annotations

import sys

from zoot.cli import build_agent, build_arg_parser
from zoot.tui.app import ZootTuiApp


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.prompt:
        print("zoot-tui does not accept one-shot prompts; start the TUI and type there.", file=sys.stderr)
        return 2
    agent = build_agent(args)
    ZootTuiApp(agent).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
