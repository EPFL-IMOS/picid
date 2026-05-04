#!/usr/bin/env bash
# Verifies that all install variants resolve without error and use the expected
# package sources (absent / local path / git).
# Run from the project root: bash scripts/test_install_variants.sh
set -euo pipefail

pass=0
fail=0

_assert() {
    # _assert <description> <pkg> <check-function>
    local label="$1" pkg="$2" fn="$3"
    local loc
    loc=$(pkg_location "$pkg")
    printf "    %-46s" "$label"
    if "$fn" "$pkg" 2>/dev/null; then
        echo "ok  ${loc:-(absent)}"
    else
        echo "FAIL  ${loc:-(absent)}"
        fail=$((fail + 1))
    fi
}

pkg_location() {
    # Prints the file path of the installed package, or empty string if absent.
    python - "$1" <<'EOF'
import sys, importlib.util
spec = importlib.util.find_spec(sys.argv[1])
print(spec.origin if spec else "")
EOF
}

assert_absent() {
    local pkg="$1"
    local loc
    loc=$(pkg_location "$pkg")
    [ -z "$loc" ]
}

assert_local() {
    local pkg="$1"
    local loc
    loc=$(pkg_location "$pkg")
    [[ "$loc" == *"local_packages"* ]]
}

assert_git() {
    local pkg="$1"
    local loc
    loc=$(pkg_location "$pkg")
    [[ "$loc" == *".venv"* ]]
}

run_variant() {
    local label="$1"; shift
    echo "$label"
    if ! uv sync "$@" --quiet 2>&1; then
        echo "  uv sync FAILED"
        fail=$((fail + 1))
        return
    fi
    pass=$((pass + 1))
}

local_requirements_present() {
    local path
    for path in \
        local_packages/TabPFN \
        local_packages/TabDPT \
        local_packages/carte-ai \
        local_packages/phmd
    do
        [ -d "$path" ] || return 1
    done
}

run_local_variant() {
    echo "local editable overlay"
    if ! local_requirements_present; then
        echo "  skipped: local_packages checkouts are missing"
        return
    fi
    if ! uv sync --quiet 2>&1; then
        echo "  uv sync FAILED"
        fail=$((fail + 1))
        return
    fi
    if ! uv pip install --quiet -r requirements-local.txt 2>&1; then
        echo "  uv pip install -r requirements-local.txt FAILED"
        fail=$((fail + 1))
        return
    fi
    pass=$((pass + 1))
}

# ── without foundation models ────────────────────────────────────────────────
run_variant "without foundation models" --no-group tabfm --no-group phmd
_assert "tabpfn absent"   tabpfn   assert_absent
_assert "tabdpt absent"   tabdpt   assert_absent
_assert "carte_ai absent" carte_ai assert_absent
_assert "phmd absent"     phmd     assert_absent

# ── phmd only (no foundation models) ────────────────────────────────────────
run_variant "phmd only (no foundation models)" --no-group tabfm
_assert "tabpfn absent"   tabpfn   assert_absent
_assert "tabdpt absent"   tabdpt   assert_absent
_assert "carte_ai absent" carte_ai assert_absent
_assert "phmd from git"   phmd     assert_git

# ── local editable overlay ───────────────────────────────────────────────────
run_local_variant
if local_requirements_present; then
    _assert "tabpfn from local"   tabpfn   assert_local
    _assert "tabdpt from local"   tabdpt   assert_local
    _assert "carte_ai from local" carte_ai assert_local
    _assert "phmd from local"     phmd     assert_local
fi

# ── default git install ──────────────────────────────────────────────────────
run_variant "default git install"  # no flags — uses default-groups
_assert "tabpfn from git"   tabpfn   assert_git
_assert "tabdpt from git"   tabdpt   assert_git
_assert "carte_ai from git" carte_ai assert_git
_assert "phmd from git"     phmd     assert_git

# ── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "$pass install variants passed, $fail checks failed"

uv sync --quiet  # restore default state

[ "$fail" -eq 0 ]
