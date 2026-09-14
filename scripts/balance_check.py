import re, sys

src = open(sys.argv[1], encoding='utf-8').read()
s = re.sub(r'//[^\n]*', '', src)
s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
s = re.sub(r'"(?:\\.|[^"\\])*"', '""', s)
s = re.sub(r"'(?:\\.|[^'\\])*'", "''", s)
ok = True
for a, b in [('{', '}'), ('(', ')'), ('[', ']')]:
    ca, cb = s.count(a), s.count(b)
    print(a + b, ca, cb, 'OK' if ca == cb else 'MISMATCH')
    ok = ok and ca == cb
print('lines', len(src.splitlines()))
sys.exit(0 if ok else 1)
