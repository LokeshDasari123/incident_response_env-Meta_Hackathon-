"""Fix Windows cp1252-incompatible Unicode in train.py and before_after_report.py."""
import pathlib

REPLACEMENTS = {
    '\u2192': '->',    # rightwards arrow (Scan->Analyze)
    '\u2550': '=',     # double horizontal box drawing
    '\u2500': '-',     # single horizontal box drawing
    '\u2014': '--',    # em dash
    '\u2013': '-',     # en dash
    '\u2018': "'",     # left single quote
    '\u2019': "'",     # right single quote
    '\u201c': '"',     # left double quote
    '\u201d': '"',     # right double quote
    '\u2265': '>=',    # greater than or equal
    '\u2264': '<=',    # less than or equal
    '\u2022': '*',     # bullet
    '\u2026': '...',   # ellipsis
    '\u221e': 'inf',   # infinity
    '\u2714': 'OK',    # heavy check mark
    '\u2713': 'OK',    # check mark
    '\u25ba': '>>',    # black right-pointing pointer
    '\u2248': '~=',    # almost equal
    '\u2560': '|',     # heavy double vertical and right
    '\u2563': '|',     # heavy double vertical and left
    '\u2566': '+',     # heavy double down and horizontal
    '\u2569': '+',     # heavy double up and horizontal
    '\u2551': '|',     # double vertical line
    '\u2554': '+',     # box drawing double down-right
    '\u2557': '+',     # box drawing double down-left
    '\u255a': '+',     # box drawing double up-right
    '\u255d': '+',     # box drawing double up-left
    '\u254c': '-',     # box drawing light double dash horizontal
    '\u2502': '|',     # box drawing light vertical
    '\u250c': '+',     # box drawing light down and right
    '\u2510': '+',     # box drawing light down and left
    '\u2514': '+',     # box drawing light up and right
    '\u2518': '+',     # box drawing light up and left
    '\u251c': '+',     # box drawing light vertical and right
    '\u2524': '+',     # box drawing light vertical and left
    '\u252c': '+',     # box drawing light down and horizontal
    '\u2534': '+',     # box drawing light up and horizontal
    '\u253c': '+',     # box drawing light vertical and horizontal
    '\u2550': '=',     # double horizontal
    '\u2571': '/',     # diagonal
    '\u2572': '\\',    # diagonal
}

TARGET_FILES = [
    'training/train.py',
    'training/before_after_report.py',
    'agents/chain_of_thought.py',
    'agents/hybrid_router.py',
    'agents/progressive_memory.py',
]

for filepath in TARGET_FILES:
    path = pathlib.Path(filepath)
    if not path.exists():
        print(f'SKIP (not found): {filepath}')
        continue
    content = path.read_text(encoding='utf-8')
    original = content
    for uni, asc in REPLACEMENTS.items():
        content = content.replace(uni, asc)
    
    # Verify cp1252 encodable
    bad_lines = []
    for i, line in enumerate(content.splitlines(), 1):
        try:
            line.encode('cp1252')
        except UnicodeEncodeError as e:
            bad_lines.append((i, repr(e.object[e.start]), line[:60]))
    
    if bad_lines:
        print(f'WARNING: {filepath} still has {len(bad_lines)} bad lines:')
        for ln, ch, snippet in bad_lines[:5]:
            print(f'  L{ln}: {ch}: {snippet}')
    else:
        changed = content != original
        path.write_text(content, encoding='utf-8')
        print(f'OK  {filepath}{"  (modified)" if changed else "  (no change)"}')

print('\nDone.')
