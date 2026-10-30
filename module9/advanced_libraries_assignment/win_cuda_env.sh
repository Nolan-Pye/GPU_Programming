# Sourced by build.sh and run.sh. Does nothing on Linux.
#
# On Windows (Git Bash) the nvcc on PATH may come from a conda install
# that has no cuRAND. If so, find a conda environment that has it, pass
# its include/lib folders to nvcc (NVCC_EXTRA) and put its DLLs on PATH
# so ./savings_sim can load them.

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) ;;
    *) return 0 ;;
esac

nvcc_root="$(dirname "$(dirname "$(command -v nvcc)")")"
if [ -f "$nvcc_root/include/curand.h" ] || [ -n "${NVCC_EXTRA:-}" ]
then
    return 0
fi

for prefix in "${CONDA_PREFIX:-}" "$HOME"/anaconda3*/envs/* \
              "$HOME"/miniconda3*/envs/*; do
    lib="$prefix/Library"
    if [ -f "$lib/include/curand.h" ] && [ -d "$lib/lib/x64" ]; then
        lib="$(cygpath -m "$lib")"
        export NVCC_EXTRA="-I$lib/include -L$lib/lib/x64"
        export PATH="$(cygpath -u "$lib")/bin:$PATH"
        return 0
    fi
done
echo "warning: curand.h not found; set NVCC_EXTRA to its folders" >&2
