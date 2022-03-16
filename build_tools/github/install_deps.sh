#!/usr/bin/env bash

set -e
set -x

python -m venv venv

if [[ "$RUNNER_OS" == "Linux" ]]; then
    sudo apt-get update
    sudo apt-get install texlive-full --fix-missing
elif [[ "$RUNNER_OS" == "macOS" ]]; then
    brew install --cask mactex
    eval "$(/usr/libexec/path_helper)"
    echo "$PATH" > "$GITHUB_PATH"
    echo "MANPATH=$MANPATH" >> "$GITHUB_ENV"
else
    echo "Windows platform is not supported."
    exit 1
fi

source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -i https://pypi.anaconda.org/ragonneau/simple cobyqa
