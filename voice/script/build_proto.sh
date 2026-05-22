#!/usr/bin/env sh

# Requires protoc to be installed

this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/../src")"

protoc -I "${src_dir}/protos/" "--python_betterproto2_out=${src_dir}/voice_messages" "${src_dir}/protos/voice.proto"
