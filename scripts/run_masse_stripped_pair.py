"""Detached sequential train/eval/decode/plot for the stripped Masse-LIF pair.

Runs on a local working directory so Nutstore/Y: checkpoint writes cannot kill
the job. Copies completed artifacts back to the repo results tree. Retries a
crashed train command so last.pt resume can continue.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"Y:\python_project\Net_torch")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.experiments.masse_delayed_cue_lif.artifacts import (  # noqa: E402
    command_is_fresh,
    require_identical_trial_tables,
)

PYTHON = Path(r"S:\pycharm\Anaconda\envs\torch_env\python.exe")
PYTHONW = Path(r"S:\pycharm\Anaconda\envs\torch_env\pythonw.exe")
WORK_ROOT = Path(r"S:\masse_delayed_cue_lif_work")
DEST_ROOT = REPO / "results" / "masse_delayed_cue_lif"
LOG_PATH = WORK_ROOT / "pair_pipeline.log"
PROFILES = ("stripped_stsp", "stripped_no_stsp")
COMMANDS = ("train", "evaluate", "decode", "plot")
TRAIN_RETRIES = 8
OTHER_RETRIES = 3
HEARTBEAT_S = 30

DETACH_FLAGS = 0
CHILD_FLAGS = 0
if os.name == "nt":
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    # pythonw + DETACHED_PROCESS; CREATE_NO_WINDOW is ignored with DETACHED_PROCESS.
    DETACH_FLAGS = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    CHILD_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | no_window


def _ignore_console_ctrl() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)
    except OSError:
        pass


def _hidden_startupinfo() -> subprocess.STARTUPINFO | None:
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0
    return info


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(message: str) -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{_stamp()} {message}\n"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    sys.stderr.write(line)
    sys.stderr.flush()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _status(workdir: Path) -> str:
    return str(_read_json(workdir / "summary.json").get("status", ""))


def _copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _prepare_workdir(profile: str) -> Path:
    dest = DEST_ROOT / profile
    work = WORK_ROOT / profile
    work.mkdir(parents=True, exist_ok=True)
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    trials_src = dest / "data" / "trials.csv"
    config_src = dest / "run_config.json"
    if not trials_src.is_file() or not config_src.is_file():
        raise FileNotFoundError(f"missing trials or run_config for {profile} at {dest}")
    shutil.copy2(trials_src, work / "data" / "trials.csv")
    shutil.copy2(config_src, work / "run_config.json")
    return work


def _sync_progress(profile: str, work: Path) -> None:
    dest = DEST_ROOT / profile
    if dest.name == "formal":
        raise RuntimeError("refusing to write formal/")
    (dest / "data").mkdir(parents=True, exist_ok=True)
    (dest / "logs").mkdir(parents=True, exist_ok=True)
    history = work / "data" / "train_history.json"
    if history.is_file():
        shutil.copy2(history, dest / "data" / "train_history.json")
    if LOG_PATH.is_file():
        shutil.copy2(LOG_PATH, dest / "logs" / "formal_pipeline.log")
    heartbeat = {
        "time": _stamp(),
        "profile": profile,
        "work": str(work),
        "status": _status(work),
    }
    (dest / "logs" / "supervisor_heartbeat.json").write_text(
        json.dumps(heartbeat, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_python(args: list[str], retries: int, *, profile: str, work: Path) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    last_rc = 1
    for attempt in range(retries + 1):
        log(f"RUN attempt {attempt + 1}/{retries + 1}: {' '.join(args)}")
        handle = LOG_PATH.open("a", encoding="utf-8")
        try:
            handle.write(f"{_stamp()} CMD {' '.join(args)}\n")
            handle.flush()
            proc = subprocess.Popen(
                args,
                cwd=str(REPO),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                startupinfo=_hidden_startupinfo(),
                creationflags=CHILD_FLAGS,
            )
            while proc.poll() is None:
                time.sleep(HEARTBEAT_S)
                log(f"heartbeat pid={proc.pid} {' '.join(args[-5:])}")
                try:
                    _sync_progress(profile, work)
                except OSError as exc:
                    log(f"progress sync skipped: {exc!r}")
            last_rc = int(proc.returncode)
        finally:
            handle.close()
        if last_rc == 0:
            log(f"OK {' '.join(args)}")
            return
        log(f"FAIL rc={last_rc} {' '.join(args)}")
        time.sleep(5)
    raise subprocess.CalledProcessError(last_rc, args)


def _sync_back(profile: str, work: Path) -> None:
    dest = DEST_ROOT / profile
    if dest.resolve() == (DEST_ROOT / "formal").resolve():
        raise RuntimeError("refusing to write formal/")
    log(f"SYNC {work} -> {dest}")
    _copy_tree(work, dest)
    if LOG_PATH.is_file():
        (dest / "logs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOG_PATH, dest / "logs" / "formal_pipeline.log")


def _should_skip(command: str, work: Path) -> bool:
    return command_is_fresh(work, command)


def run_profile(profile: str) -> None:
    work = _prepare_workdir(profile)
    log(f"==== {profile} pipeline start work={work} ====")
    for command in COMMANDS:
        if _should_skip(command, work):
            log(f"skip {command}; already complete for {profile}")
            continue
        retries = TRAIN_RETRIES if command == "train" else OTHER_RETRIES
        _run_python(
            [
                str(PYTHON),
                "-u",
                "-m",
                "src.experiments.masse_delayed_cue_lif.run",
                command,
                "--profile",
                profile,
                "--output-directory",
                str(work),
                "--device",
                "cuda",
            ],
            retries=retries,
            profile=profile,
            work=work,
        )
        _sync_back(profile, work)
    log(f"==== {profile} pipeline complete ====")


def _spawn_detached() -> int:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    launcher = PYTHONW if PYTHONW.is_file() else PYTHON
    proc = subprocess.Popen(
        [str(launcher), "-u", str(script)],
        cwd=str(REPO),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        startupinfo=_hidden_startupinfo(),
        creationflags=DETACH_FLAGS,
    )
    pid_path = WORK_ROOT / "supervisor.pid"
    pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps({"status": "detached", "pid": proc.pid, "log": str(LOG_PATH)}) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _ignore_console_ctrl()
    if "--detach" in args:
        return _spawn_detached()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    log("pair supervisor start")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    require_identical_trial_tables(
        *(DEST_ROOT / profile / "data" / "trials.csv" for profile in PROFILES)
    )
    for profile in PROFILES:
        if profile == "formal":
            raise RuntimeError("refusing formal profile")
        run_profile(profile)
    log("BOTH FORMAL PAIR PIPELINES COMPLETE")
    (WORK_ROOT / "COMPLETE").write_text(_stamp() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        log(f"PIPELINE FAILED: {exc!r}")
        raise
