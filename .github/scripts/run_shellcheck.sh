#!/usr/bin/env bash
# Run shellcheck over every shell script in the repository.
#
# The s6-overlay service scripts have no file extension and a
# `#!/command/with-contenv bashio` shebang that shellcheck does not recognise,
# so they are found by shebang and rely on their own `# shellcheck shell=bash`
# directive.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

declare -a scripts=()
while IFS= read -r file; do
    case "${file}" in
        *.sh | *.bash)
            scripts+=("${file}")
            continue
            ;;
    esac
    # Match an interpreter we can actually check, including the bashio wrapper
    # used by the s6 run/finish scripts.
    if IFS= read -r shebang <"${file}" 2>/dev/null &&
        [[ "${shebang}" =~ ^#!.*(bash|bashio|/bin/sh|/usr/bin/env[[:space:]]+sh)([[:space:]]|$) ]]; then
        scripts+=("${file}")
    fi
done < <(git ls-files)

if [[ ${#scripts[@]} -eq 0 ]]; then
    echo 'No shell scripts found' >&2
    exit 1
fi

echo "Checking ${#scripts[@]} shell script(s):"
printf '  %s\n' "${scripts[@]}"
echo

# -x lets shellcheck follow `source` directives between scripts.
shellcheck --external-sources --color=always "${scripts[@]}"

echo 'shellcheck found no issues'
