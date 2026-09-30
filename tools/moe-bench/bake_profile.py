#!/usr/bin/env python3
"""Bake the expert cache's usage profile into a GGUF (key moe_cache.expert_usage).

The fork saves per-expert lifetime activation counts at shutdown to
~/.cache/llama.cpp/moe-hot-<model>-<size>.bin and preloads the hottest experts at the next start.
A GGUF carrying the same table gives that hot start on a new machine, with no local profile yet.

  bake_profile.py model.gguf out.gguf [--profile ~/.cache/llama.cpp/moe-hot-....bin]

For a split model pass the first shard; only its metadata changes, the other shards stay as they
are (copy or link them next to the output with their names). The input file is never modified.
"""
import argparse, glob, os, re, struct, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gguf-py"))
import gguf  # noqa: E402
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata  # noqa: E402

KEY = "moe_cache.expert_usage"
MAGIC = 0x3448454d  # "MEH4"


def read_profile(path):
    with open(path, "rb") as f:
        magic, n_layers = struct.unpack("II", f.read(8))
        if magic != MAGIC:
            sys.exit(f"{path}: not an expert cache profile")
        layers = []
        for _ in range(n_layers):
            n, = struct.unpack("I", f.read(4))
            layers.append(list(struct.unpack(f"{n}I", f.read(4 * n))))
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--profile", help="profile file (default: the only matching ~/.cache/llama.cpp/moe-hot-*.bin)")
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.model)
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    n_layer = reader.get_field(f"{arch}.block_count").contents()
    # the cache's layers are the MoE layers in order: map them to model layer indices
    moe = sorted({int(m.group(1)) for t in reader.tensors if (m := re.match(r"blk\.(\d+)\.ffn_gate_exps\.weight", t.name))})
    if not moe:
        sys.exit("no MoE expert tensors in this file (for a split model pass the first shard)")

    path = a.profile
    if not path:
        cands = glob.glob(os.path.expanduser("~/.cache/llama.cpp/moe-hot-*.bin"))
        if len(cands) != 1:
            sys.exit("pass --profile (found %d profiles: %s)" % (len(cands), ", ".join(cands)))
        path = cands[0]
    prof = read_profile(path)
    if len(prof) != len(moe):
        sys.exit(f"profile has {len(prof)} layers, model has {len(moe)} MoE layers: other model?")
    n_expert = len(prof[0])
    table = [0] * (n_layer * n_expert)
    for il, counts in zip(moe, prof):
        table[il * n_expert:(il + 1) * n_expert] = counts

    writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
    new = {KEY: MetadataDetails(gguf.GGUFValueType.ARRAY, table, sub_type=gguf.GGUFValueType.UINT32)}
    copy_with_new_metadata(reader, writer, new, [KEY])
    print(f"baked {len(moe)} layers x {n_expert} experts from {path} into {a.out}")


if __name__ == "__main__":
    main()
