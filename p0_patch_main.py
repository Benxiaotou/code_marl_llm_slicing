"""Safely patch just the sample-file selection in original main.py.

No environment maths, reward, MARL training logic, or test logic are changed.
Backup main.py.p0-original-backup is created, and rerunning is idempotent.
"""
from pathlib import Path


def patch_main(path=Path('main.py')):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Run this in the original project folder containing main.py')
    source = path.read_bytes()
    if b'p0_sample_file' in source and b'p0_sample_cnt' in source:
        print('main.py already patched; no action')
        return False
    old_count = b'self.sample_cnt = 600  #'
    new_count = b'self.sample_cnt = int(getattr(config, "p0_sample_cnt", 600))  #'
    if source.count(old_count) != 1:
        raise ValueError('Expected exactly one original sample_cnt declaration; no change')
    # The 4-line filename expression ends at the sample_save.pkl literal.
    anchor = b"+ '_sample_save.pkl'"
    field = b'self.sample_dict_save_file = self.data_dir'
    if source.count(field) != 1:
        raise ValueError('Cannot locate sample filename field uniquely; no change')
    eol = b'\r\n' if b'\r\n' in source else b'\n'
    source_new = source.replace(old_count, new_count)
    field_pos = source_new.index(field)
    pos = source_new.index(anchor, field_pos) + len(anchor)
    if pos - field_pos > 400:
        raise ValueError('Unexpected distance between filename field and suffix')
    if source_new[pos:pos+len(eol)] != eol:
        raise ValueError('Unexpected end of sample filename expression; no change')
    insert = eol + b'        self.sample_dict_save_file = getattr(config, "p0_sample_file", None) or self.sample_dict_save_file'
    source_new = source_new[:pos] + insert + source_new[pos:]
    backup = path.with_name(path.name + '.p0-original-backup')
    if backup.exists():
        raise FileExistsError('Backup already exists; check manually: ' + str(backup))
    backup.write_bytes(source)
    try:
        path.write_bytes(source_new)
    except Exception:
        path.write_bytes(source)
        raise
    print('main.py patched; original saved as', backup)
    return True


if __name__ == '__main__':
    patch_main()
