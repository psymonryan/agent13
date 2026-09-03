"""Shared test helpers for process spawning.

Uses pexpect.spawn on Unix (full terminal emulation) and
pexpect.popen_spawn.PopenSpawn on Windows (pipe-based, no pty).
"""


def spawn_process(command, args=None, env=None, encoding="utf-8", timeout=30,
                  dimensions=(50, 200), maxread=4096):
    """Spawn a process using pexpect.spawn on Unix, PopenSpawn on Windows.

    PopenSpawn doesn't support dimensions or maxread -- those are silently
    dropped on Windows.  The returned object supports .expect(), .sendline(),
    .before, .close(), and .timeout on both platforms.
    """
    import pexpect
    import sys
    if sys.platform == "win32":
        from pexpect.popen_spawn import PopenSpawn
        cmd_str = f"{command} {' '.join(args or [])}"
        proc = PopenSpawn(cmd_str, encoding=encoding, codec_errors="replace",
                          timeout=timeout, env=env)
        # PopenSpawn lacks .close() and .sendcontrol() that pexpect.spawn has.
        # .close() -> send EOF then terminate the underlying subprocess.
        # .sendcontrol(c) -> send the control character (Ctrl+C = \x03, Ctrl+D = \x04, etc).
        proc.close = lambda: (proc.sendeof(), proc.proc.terminate())
        _orig_sendcontrol = getattr(proc, 'sendcontrol', None)
        if _orig_sendcontrol is None:
            def _sendcontrol(char):
                code = ord(char.lower()) - ord('a') + 1
                proc.send(chr(code))
            proc.sendcontrol = _sendcontrol
    else:
        # codec_errors="replace": macOS ttys interleave kernel-echoed input
        # bytes with app output under load (fast typing during redraws),
        # which can split a multi-byte UTF-8 char across writes -> invalid
        # stream. Real terminals survive it (next redraw overwrites); the
        # strict decoder must not crash the test. Same policy as Windows.
        proc = pexpect.spawn(
            command, args=args, env=env, encoding=encoding,
            codec_errors="replace",
            timeout=timeout, dimensions=dimensions, maxread=maxread,
        )
    proc.timeout = timeout
    return proc
