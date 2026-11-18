# Sourced by the sbatch scripts (after `cd "$SLURM_SUBMIT_DIR"`): picks the interpreter, checks it
# imports this checkout's ghost (not a stale install), and unbuffers Python output for the Slurm logs.
# PYTHON=/path/to/python (or a command name on PATH) overrides the search; the override is still checked.

expected=$(cd "$SLURM_SUBMIT_DIR/src" && pwd -P)

# check_python PY: sets ghost_dir to where PY imports ghost from; returns 1 unless that is this checkout's src/.
# An env can hold an editable install of an old checkout and silently run old code, so require src/.
check_python() {
    local ghost_file
    ghost_file=$("$1" -c "import ghost; print(ghost.__file__)" 2>/dev/null) || { ghost_dir="(cannot import ghost)"; return 1; }
    ghost_dir=$(cd "$(dirname "$ghost_file")" && pwd -P)
    case "$ghost_dir/" in
        "$expected"/*) return 0 ;;
        *) return 1 ;;
    esac
}

if [ -n "${PYTHON:-}" ]; then
    resolved_py=$(command -v "$PYTHON") || {
        echo "error: PYTHON=$PYTHON is not an executable; set PYTHON=/path/to/env/bin/python" >&2
        exit 1
    }
    PYTHON=$resolved_py
    if ! check_python "$PYTHON"; then
        echo "error: $PYTHON gives ghost from $ghost_dir, not $expected (stale install?); 'pip install -e .' from this checkout in that env, or set PYTHON" >&2
        exit 1
    fi
else
    # `ghost` is environment.yaml's name; ghost_env is what Longleaf's working env is called.
    rejected=""
    for candidate in "$HOME/.conda/envs/ghost/bin/python" "$HOME/.conda/envs/ghost_env/bin/python"; do
        [ -x "$candidate" ] || continue
        if check_python "$candidate"; then PYTHON=$candidate; break; fi
        rejected="$rejected
  $candidate -> $ghost_dir"
    done
    if [ -z "${PYTHON:-}" ]; then
        if [ -n "$rejected" ]; then
            echo "error: no candidate interpreter imports ghost from $expected (stale install?); rejected:$rejected
'pip install -e .' from this checkout in one of them, or set PYTHON" >&2
        else
            echo "error: none of ~/.conda/envs/ghost/bin/python or ~/.conda/envs/ghost_env/bin/python exist; set PYTHON=/path/to/env/bin/python" >&2
        fi
        exit 1
    fi
fi
echo "python: $PYTHON"
echo "ghost:  $ghost_dir"

export PYTHONUNBUFFERED=1
