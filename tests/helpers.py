"""Shared test helpers for process spawning.

Uses pexpect.spawn on Unix (full terminal emulation) and
pexpect.popen_spawn.PopenSpawn on Windows (pipe-based, no pty).
"""

# The FIRST spawn of a pytest session pays the box's cold-start cost, and on a
# CI checkout that cost is large: ci_remote.sh wipes the remote tree (including
# the venv) and ships a tarball with __pycache__/*.pyc excluded, then `uv sync`
# writes a brand-new venv - so the first spawned REPL compiles the whole import
# graph from source while Defender scans freshly written files. Warm, a spawn
# reaches its prompt in ~7s; cold on a loaded box it can exceed a fixed 30s
# budget (the intermittent Windows failure in test_auto_context_integration.py,
# the first file in the suite to spawn a process - every later spawn is warm).
# The cost is a property of the first spawn, not of any test, so only that
# first spawn gets the generous ceiling and every later spawn keeps its own
# budget. expect() returns as soon as its needle arrives, so a bigger timeout
# is a ceiling, not a sleep: this costs nothing on a warm box.
FIRST_SPAWN_TIMEOUT = 120
_first_spawn_done = False


def spawn_process(command, args=None, env=None, encoding="utf-8", timeout=30,
                  dimensions=(50, 200), maxread=4096):
    """Spawn a process using pexpect.spawn on Unix, PopenSpawn on Windows.

    PopenSpawn doesn't support dimensions or maxread -- those are silently
    dropped on Windows.  The returned object supports .expect(), .sendline(),
    .before, .close(), .terminate(), and .timeout on both platforms.
    """
    global _first_spawn_done
    if not _first_spawn_done:
        # First spawn of the session: pay the cold box at a generous ceiling
        # (see FIRST_SPAWN_TIMEOUT above). Every later spawn keeps the
        # timeout its caller asked for.
        timeout = max(timeout, FIRST_SPAWN_TIMEOUT)
        _first_spawn_done = True
    import pexpect
    import sys
    if sys.platform == "win32":
        from pexpect.popen_spawn import PopenSpawn
        cmd_str = f"{command} {' '.join(args or [])}"
        proc = PopenSpawn(cmd_str, encoding=encoding, codec_errors="replace",
                          timeout=timeout, env=env)

        # PopenSpawn lacks .close() and .sendcontrol() that pexpect.spawn has.
        # .close(force) mirrors pexpect.spawn.close(force=False):
        #   force=False -> send EOF then terminate the underlying subprocess.
        #   force=True  -> terminate immediately (no EOF).
        # .sendcontrol(c) -> send the control character (Ctrl+C = \x03, Ctrl+D = \x04, etc).
        def _close(force=False):
            if not force:
                proc.sendeof()
            proc.proc.terminate()
        proc.close = _close

        # PopenSpawn also lacks .terminate() that pexpect.spawn has (tests call
        # proc.terminate(force=True) in finally blocks). Mirror pexpect.spawn's
        # terminate: stop the underlying subprocess.
        def _terminate(force=False):
            proc.proc.terminate()
        proc.terminate = _terminate
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
