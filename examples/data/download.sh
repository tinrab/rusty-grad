#!/bin/bash

set -euo pipefail
current_path="$(realpath $0)"
current_dir="$(dirname $current_path)"

function mnist() {
	mkdir -p "$current_dir/mnist"
	cd "$current_dir/mnist"

	url="https://storage.googleapis.com/cvdf-datasets/mnist/"
	files=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")

	for file in "${files[@]}"; do
		wget -nc "$url$file"
	done

	gzip -d *.gz

	cd "$current_dir"
}

function help() {
	echo "Usage: $(basename "$0") [OPTIONS]

Commands:
  mnist          Download MNIST dataset
  help           Show help
"
}

if [[ $1 =~ ^(mnist|help)$ ]]; then
	"$@"
else
	help
	exit 1
fi
