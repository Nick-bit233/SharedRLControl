# Frozen real-flight environment backup and restore

This document defines the non-destructive archive contract used for the
2026-08-07 real-flight freeze.  It is intentionally separate from normal
experiment operation: creating or checking an archive must never arm a vehicle,
start MAVROS, contact Nokov, or replay a flight.

## Canonical snapshot

The snapshot ID is `sharedrlcontrol-realflight-20260807`.  A complete snapshot
has three copies:

1. GitHub branches and tag:
   `real-px4-srlc-deploy`,
   `codex/collision-clearance-dual-channel`,
   `archive/real-flight-container-20260721`, and
   `archive/real-flight-20260807`.
2. A local directory named
   `sharedrlcontrol-realflight-20260807`.
3. One outer archive object at
   `aliyunoss:pjlab-bjpai-ipec/wanghaotian/SharedRLControl/real-flight-archive/2026-08-07/sharedrlcontrol-realflight-20260807.tar.zst`.

The local NTFS volume and the working tree are partitions on the same physical
NVMe device.  The local copy is useful for recovery mistakes, but it is not an
independent disk-failure copy.  After the local snapshot directory passes all
checks, it is wrapped as one `sharedrlcontrol-realflight-20260807.tar.zst`
object with the snapshot directory as its top-level member.  No individual
payload file or separate completion marker is uploaded to OSS.

`BACKUP_COMPLETE.json` is sealed inside that outer archive and records the
locally verified payload plus the required remote object address.  The OSS copy
is complete only after the outer object is streamed back in full and compared
with the local outer archive.  The post-upload object size, SHA-256, rclone
check result, and verification time are kept in the local acceptance record;
they cannot be inserted into the already immutable remote object.

The archive contains:

```text
MANIFEST.yaml
FILE_SHA256SUMS
SHA256SUMS
BACKUP_COMPLETE.json
RESTORE.md
code/SharedRLControl-all-refs.bundle
data/real-flight-results.tar.zst
data/training-dataset.tar.zst
data/models-maps-hardware-assets.tar.zst
docker/docker-images-linux-amd64.tar.zst
provenance/container-logs-and-environment.tar.zst
```

`FILE_SHA256SUMS` records hashes of source data before compression.
`SHA256SUMS` records the inner transport artifacts.  The SHA-256 of the outer
archive is recorded separately in the local acceptance record.
`MANIFEST.yaml` maps every source path to its archive path and records the
source size, SHA-256, UTC/CST times, companions, paper role, Git refs,
container/image IDs, and platform.

## Secrets and immutable source data

Never put an AccessKey, SecretKey, SSH key, GnuPG home, editor server, Copilot
state, or shell history in Git or in an archive.  Keep the rclone configuration
outside the repository with mode `0600`.  Record only the rclone version,
remote name, endpoint, provider, ACL, storage class, and the bucket's observed
versioning/SSE state.

Do not move, rename, deduplicate, or rewrite experiment outputs in the working
tree.  The archive staging layout may group runs by `YYYY/MM/DD/<run-id>/`, but
`MANIFEST.yaml` remains the authoritative source-to-archive mapping.

The original real-flight container and original source files are retained for
at least 30 days after both the local and OSS recovery checks pass.  Removing
them requires a separate decision.

## Git recovery

Verify the bundle before using it:

```bash
git bundle verify code/SharedRLControl-all-refs.bundle
git clone code/SharedRLControl-all-refs.bundle SharedRLControl-restored
git -C SharedRLControl-restored show-ref
git -C SharedRLControl-restored checkout archive/real-flight-20260807
```

The container archive branch starts at the pre-recorded-replay commit and
contains the two source files recovered from the active 2026-07-21 container.
It exists to preserve that exact flight environment; do not rebase it.

## Data recovery

First require the completion marker, verify all transport hashes, and test each
compressed tar stream:

```bash
test -f BACKUP_COMPLETE.json
sha256sum -c SHA256SUMS

for archive in data/*.tar.zst provenance/*.tar.zst; do
  zstd -t "$archive"
  tar -tf "$archive" >/dev/null
done
```

Follow the source-to-archive mappings in `MANIFEST.yaml` when restoring files.
Restore into a new directory, never over the live repository.  After
extraction, run `sha256sum -c FILE_SHA256SUMS` from the layout documented in
the snapshot's `RESTORE.md`.

The four fixed paper roles are:

- `20260721_111553`: `primary_example`
- `20260717_062438`: `success_candidate`
- `20260717_062851`: `direct_candidate`
- `20260717_060212`: `collision_candidate`

The primary example must include its source JSON/NPZ pair, trajectory report,
replay sidecar, PNG/SVG figures, and MP4.  Paper roles are metadata on the
canonical files; they are not separate paper-data copies.

## Docker recovery

The archive contains two `linux/amd64` images:

- `srlc_ros1_real:real-flight-20260721-archive-20260807`
- `srlc_ros1_real:workspace-20260807-<gitsha12>`

Load and inspect them:

```bash
zstd -dc docker/docker-images-linux-amd64.tar.zst | docker load
docker image inspect \
  srlc_ros1_real:real-flight-20260721-archive-20260807
docker image inspect \
  srlc_ros1_real:workspace-20260807-<gitsha12>
```

Select either image without editing the base Compose file by setting
`SRLC_IMAGE`.  Always combine the base file with the offline restore override
and use a new project name:

```bash
cd SharedRLControl-restored/ros1_real
export SRLC_IMAGE=srlc_ros1_real:workspace-20260807-<gitsha12>
export SRLC_CKPT_HOST_DIR=/absolute/path/to/restored/ros1/ckpts
export SRLC_MAP_HOST_DIR=/absolute/path/to/restored/ros1/real_maps
export SRLC_OUTPUT_HOST_DIR=/absolute/path/to/empty/restore-output

docker compose \
  -p srlc_archive_restore \
  -f docker-compose.real.yml \
  -f docker-compose.restore.yml \
  config

docker compose \
  -p srlc_archive_restore \
  -f docker-compose.real.yml \
  -f docker-compose.restore.yml \
  up -d --no-build
```

The merged service must report `network_mode: none`, and its command must remain
`sleep infinity`.  It must not start a launch file automatically.  Validate
the loaded environment without any network:

```bash
docker compose \
  -p srlc_archive_restore \
  -f docker-compose.real.yml \
  -f docker-compose.restore.yml \
  exec -T real_runtime bash -lc '
    rospack find srlc_real &&
    rospack find mavros &&
    rospack find mavros_msgs &&
    rospack find nokov_uav &&
    python3 -m pip check &&
    cd /root/catkin_ws &&
    catkin_make -DCATKIN_ENABLE_TESTING=ON run_tests_srlc_real &&
    catkin_test_results build/test_results
  '
```

Stop only the isolated recovery project when finished:

```bash
docker compose \
  -p srlc_archive_restore \
  -f docker-compose.real.yml \
  -f docker-compose.restore.yml \
  down
```

Do not bring down, rename, or remove the original real-flight container as part
of a recovery drill.

## OSS write and acceptance rules

The rclone remote is `aliyunoss`, using Alibaba's S3-compatible public endpoint
`oss-cn-beijing.aliyuncs.com`, private ACL, and `STANDARD` storage class.  The
bucket is `pjlab-bjpai-ipec`; the object prefix is
`wanghaotian/SharedRLControl/real-flight-archive/2026-08-07/`.
Do not change bucket versioning, server-side encryption, or lifecycle policy
during backup.  Record the observed state, including any missing protection,
in `MANIFEST.yaml`.

Create the outer archive only after the local snapshot directory is finalized:

```bash
tar --zstd -cf sharedrlcontrol-realflight-20260807.tar.zst \
  sharedrlcontrol-realflight-20260807
zstd -t sharedrlcontrol-realflight-20260807.tar.zst
tar -tf sharedrlcontrol-realflight-20260807.tar.zst >/dev/null
sha256sum sharedrlcontrol-realflight-20260807.tar.zst
```

Upload exactly that one file with immutable/checksum semantics; never use
`sync` and never copy the unpacked directory:

```bash
rclone copyto --immutable --checksum \
  sharedrlcontrol-realflight-20260807.tar.zst \
  aliyunoss:pjlab-bjpai-ipec/wanghaotian/SharedRLControl/real-flight-archive/2026-08-07/sharedrlcontrol-realflight-20260807.tar.zst
```

The target prefix must contain exactly one object for this snapshot.  Compare
its size with the local outer archive, then run a filtered
`rclone check --download` so the complete object is streamed back and compared.
The local acceptance record must contain the outer SHA-256, verified object
count (`1`), verified byte count, verification timestamp, and rclone version.
