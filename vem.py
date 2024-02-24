"""Richard's janky Python environment management tool."""

__author__ = "Richard Si"
__version__ = "2024.02.24a1"

import json
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Union

import click
import platformdirs
import questionary
from click import secho
from click_aliases import ClickAliasedGroup
from questionary import Choice

WINDOWS = platform.system() == "Windows"
BASE_PATH = Path(platformdirs.user_data_dir("vem", "ichard26"))
RECORD_PATH = BASE_PATH / "record.json"
ENV_STORE_PATH = BASE_PATH / "envs"


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default( o)


@dataclass(frozen=True)
class PythonInstall:
    label: str
    version: str
    location: Path
    flags: Sequence[str]

    @property
    def executable(self) -> Path:
        bin_dir = "Scripts" if WINDOWS else "bin"
        p = self.location / bin_dir / "python3"
        assert p.exists()
        return p

    @property
    def default(self) -> bool:
        return "default" in self.flags

    @property
    def external(self) -> bool:
        return "external" in self.flags


@dataclass(frozen=True)
class Environment:
    label: str
    project: Path
    python: PythonInstall
    location: Path
    flags: Sequence[str]

    created_at: datetime

    @property
    def default(self) -> bool:
        return "default" in self.flags

    @property
    def description(self) -> str:
        return self.label or self.location.name


def activate_path(env: Environment, shell: str) -> Path:
    bin_dir = "Scripts" if WINDOWS else "bin"
    if shell == "default":
        return env.location / bin_dir / "activate"
    elif shell == "fish":
        return env.location / bin_dir / "activate.fish"
    raise ValueError(f"unsupported shell: {shell}")


def load_record(path: Path = RECORD_PATH) -> tuple[list[Environment], dict[str, PythonInstall]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    pythons = {}
    for entry in data.get("python-installs", []):
        version = entry["version"]
        pythons[version] = PythonInstall(
            label=entry["label"],
            version=version,
            location=Path(entry["location"]),
            flags=entry["flags"],
        )
    envs = []
    for entry in data.get("environments", []):
        envs.append(Environment(
            label=entry["label"],
            project=Path(entry["project"]),
            location=Path(entry["location"]),
            python=pythons[entry["python-version"]],
            flags=entry["flags"],
            created_at=datetime.fromisoformat(entry["created_at"]),
        ))
    return envs, pythons


def save_record(
    environments: Sequence[Environment],
    pythons: dict[str, PythonInstall],
    path: Path = RECORD_PATH
) -> None:
    data = {}
    data["python-installs"] = [asdict(p) for p in pythons.values()]
    data["environments"] = [asdict(e) for e in environments]
    for e in data["environments"]:
        e["python-version"] = e["python"]["version"]
        del e["python"]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, cls=JSONEncoder, indent=2)


def default_python(pythons: dict[str, PythonInstall]) -> PythonInstall:
    default = [p for p in pythons.values() if p.default]
    assert len(default) == 1, "there should be only one default python install"
    return default[0]


def search_environments_at(location: Path, to_search: Sequence[Environment]) -> Sequence[Environment]:
    envs = [e for e in to_search if e.project == location]
    return sorted(envs, key=lambda e: e.default, reverse=True)


class SmartAliasedGroup(ClickAliasedGroup):
    """
    This subclass extends click-aliases' Group subclass to also support implicit
    abbreviated aliases. BSD-3 licensed.

    Source: https://github.com/pallets/click/blob/cab9483a30379f9b8e3ddb72d5a4e88f88d517b6/examples/aliases/aliases.py
    """

    def get_command(self, ctx, cmd_name: str):
        if result := super().get_command(ctx, cmd_name):
            return result
        # Fallback option: if we did not find an explicit alias we
        # allow automatic abbreviation of the command. "status" for
        # instance will match "st". We only allow that however if
        # there is only one command.
        matches = [
            x for x in self.list_commands(ctx) if x.lower().startswith(cmd_name.lower())
        ]
        if not matches:
            return None
        elif len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")

    def resolve_command(self, ctx, args):
        # always return the command's name, not the alias
        _, cmd, args = super().resolve_command(ctx, args)
        return cmd.name, cmd, args


@click.group(cls=SmartAliasedGroup)
def main() -> None:
    """Richard's janky Python environment management tool."""
    if not RECORD_PATH.exists():
        RECORD_PATH.parent.mkdir(exist_ok=True, parents=True)
        RECORD_PATH.write_text("{}\n", encoding="utf-8")
    if not ENV_STORE_PATH.exists():
        ENV_STORE_PATH.mkdir(exist_ok=True, parents=True)


@main.command("new")
@click.option("-l", "--label", help="Give the environment a descriptive label.", default="")
def command_env_new(label: str) -> None:
    """Create a new environment for the CWD."""
    all_envs, pythons = load_record()
    version = questionary.select(
        "What version of Python do you want?",
        choices=list(pythons.keys()),
        default=default_python(pythons).version,
    ).ask()
    if not version:
        sys.exit(1)
    python = pythons[version]

    cwd = Path.cwd()
    project_envs = search_environments_at(cwd, all_envs)
    # if any(e.python == python for e in project_envs):
    #     old_env = next(e for e in project_envs if e.python == python)
    #     secho(f"\n[!] A {python.version} environment already exists! ({old_env.location.name})", fg="yellow")
    #     sys.exit(1)

    make_default = not bool(project_envs)
    if make_default:
        secho("[+] No existing environments found, this will be the default :)", fg="magenta")
    else:
        if not label:
            label = questionary.text("Let's give the environment a label:").ask()

    timestamp = datetime.now()
    location = ENV_STORE_PATH / f"{re.sub('[^0-9a-zA-Z]', '-', cwd.name)}-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    location = location.resolve()
    cmd: list[Union[str, Path]] = [sys.executable, "-m", "virtualenv", location, "--python", python.executable]
    if label:
        cmd.extend(("--prompt", label))
    subprocess.run(cmd, check=True)
    all_envs.append(Environment(
        label=label,
        python=python,
        project=cwd,
        location=location,
        flags=["default"] if make_default else [],
        created_at=timestamp,
    ))
    save_record(all_envs, pythons)
    secho(f"[+] Created new {version} environment for {cwd} ~❀", fg="green")


@main.command("import")
@click.argument("root", type=click.Path(exists=True, resolve_path=True, path_type=Path))
@click.option("-l", "--label", help="Give the environment a descriptive label.", default="")
def command_env_import(root: Path, label: str) -> None:
    """Import a pre-existing environment for the CWD."""
    all_envs, pythons = load_record()
    executable = Path(root, "Scripts", "python3") if WINDOWS else Path(root, "bin", "python3")
    proc = subprocess.run(
        [executable, "-c", "import sys; print(sys.version.split(' ')[0])"],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    version = proc.stdout.strip()
    python = pythons[version]

    cwd = Path.cwd()
    project_envs = search_environments_at(cwd, all_envs)
    make_default = not bool(project_envs)
    if make_default:
        secho("[+] No existing environments found, this will be the default :)", fg="magenta")

    timestamp = datetime.now()
    all_envs.append(Environment(
        label=label,
        python=python,
        project=cwd,
        location=root,
        flags=["default", "external"] if make_default else ["external"],
        created_at=timestamp,
    ))
    save_record(all_envs, pythons)
    secho(f"[+] Imported {version} environment ({root.name}) for {cwd} ~❀", fg="green")


@main.command("list")
@click.option("-a", "--all", "list_all", is_flag=True, help="List *all* environments managed by vem.")
def command_env_list(list_all: bool) -> None:
    """List environments."""
    envs: Sequence[Environment]
    envs, _ = load_record()
    if list_all:
        envs_by_project = defaultdict(list)
        for e in envs:
            envs_by_project[e.project].append(e)
        for project, project_envs in envs_by_project.items():
            secho(project, fg="magenta", bold=True)
            for e in project_envs:
                secho(f" - {e.python.version}", nl=False)
                secho(f" ({e.description})", dim=True)
            print()
        secho(f"{len(envs)} environments are currently managed by vem.", bold=True)
    else:
        envs = search_environments_at(Path.cwd(), envs)
        secho(f"Found {len(envs)} environments for {Path.cwd()}\n")
        for e in envs:
            color = "magenta" if e.default else None
            secho(f"- {e.python.version}", nl=False, fg=color, bold=True)
            secho(f" ({e.description})", fg=color)


@main.command("activation-path")
@click.argument("shell", type=click.Choice(["default", "fish"]))
def command_env_activation_path(shell: str) -> None:
    """Return an environment's activation script path (STDERR)."""
    envs, _ = load_record()
    cwd = Path.cwd()
    project_envs = search_environments_at(cwd, envs)
    if len(project_envs) == 0:
        secho(f"[!] No environments found for {cwd}", fg="yellow")
        sys.exit(1)
    elif len(project_envs) == 1:
        print(activate_path(project_envs[0], shell), file=sys.stderr)
    else:
        choices = [
            Choice(f"{e.python.version} ({e.description})", value=e)
            for e in project_envs
        ]
        default_choice = next(c for c in choices if c.value.default)
        selected = questionary.select(
            "Activate which environment?", choices=choices, default=default_choice
        ).ask()
        if not selected:
            sys.exit(1)
        print(activate_path(selected, shell), file=sys.stderr)


@main.command("remove", aliases=["rm"])
def command_env_remove() -> None:
    """Remove environments."""
    envs, pythons = load_record()
    project_envs = search_environments_at(Path.cwd(), envs)
    if not project_envs:
        secho(f"[!] No environments to remove for {Path.cwd()}", fg="yellow")
        sys.exit(1)

    selected = questionary.checkbox(
        "Which environments do you want to prune?",
        choices=[
            Choice(f"{e.python.version} ({e.description})", value=e)
            for e in project_envs
        ],
    ).ask()
    for e in selected:
        shutil.rmtree(e.location, ignore_errors=True)
        project_envs.remove(e)
        envs.remove(e)
        secho(f"[-] Removed {e.python.version} environment ({e.description})", fg="red")

    if project_envs and all(not e.default for e in project_envs):
        selected = questionary.select(
            "Select a new default environment:",
            choices=[
                Choice(f"{e.python.version} ({e.description})", value=e)
                for e in project_envs
            ],
        ).ask()
        envs[envs.index(selected)] = replace(selected, flags=[*selected.flags, "default"])
    save_record(envs, pythons)


@main.group("python", cls=SmartAliasedGroup)
def group_python() -> None:
    """Manage Python installations."""


@group_python.command("add")
@click.argument("executable", type=click.Path(exists=True, path_type=Path))
@click.option("--default/--no-default", "mark_as_default", help="Mark this Python installation as the default.")
def command_python_add(executable: Path, mark_as_default: bool) -> None:
    envs, pythons = load_record()
    proc = subprocess.run(
        [executable, "-c", "import sys; print(sys.prefix); print(sys.version.split(' ')[0])"],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    location, version = proc.stdout.splitlines()
    if current_install := pythons.get(version):
        secho(f"[!] Python {version} is already registered (from {current_install.location})", fg="yellow")
        if not questionary.confirm("Do you want to replace the current installation?").ask():
            sys.exit(1)

    if mark_as_default and (default_install := default_python(pythons)):
        secho(f"[+] Undefaulting {default_install.version} (from {default_install.location})", fg="cyan")
    flags = ["external", "default"] if mark_as_default else ["external"]
    pythons[version] = PythonInstall(label="", version=version, location=Path(location), flags=flags)
    secho(f"[+] Registered Python {version} at {location}", fg="green")
    save_record(envs, pythons)


@group_python.command("list")
def command_python_list() -> None:
    _, pythons = load_record()
    for p in sorted(pythons.values(), key=lambda p: p.version):
        color = "magenta" if p.default else None
        secho(f"- {p.version}", nl=False, fg=color, bold=True)
        secho(f" via {p.location} ({', '.join(p.flags)})", fg=color)


@group_python.command("remove", aliases=["rm"])
def command_python_remove() -> None:
    pass


if __name__ == "__main__":
    main()
