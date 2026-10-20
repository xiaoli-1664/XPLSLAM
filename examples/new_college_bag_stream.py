#!/usr/bin/env python3
"""Stream synchronized NewCollege compressed camera frames from a ROS1 bag."""

from __future__ import annotations

import argparse
import struct
import sys
from collections import deque
from pathlib import Path
from typing import BinaryIO

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError as error:
    raise SystemExit(
        "Install the bag reader with: python3 -m pip install -r examples/requirements.txt"
    ) from error


MAGIC = b"XPLNCB01"
CAMERA_COUNT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream four synchronized NewCollege compressed-image topics."
    )
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument(
        "--topic",
        action="append",
        required=True,
        help="Compressed-image topic in output camera order; specify four times.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--sync-tolerance-ns",
        type=int,
        default=20_000,
        help="Maximum timestamp span within a synchronized frame.",
    )
    args = parser.parse_args()

    if len(args.topic) != CAMERA_COUNT or len(set(args.topic)) != CAMERA_COUNT:
        parser.error("--topic must be specified exactly four times and be unique")
    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.step <= 0:
        parser.error("--step must be > 0")
    if args.sync_tolerance_ns < 0:
        parser.error("--sync-tolerance-ns must be >= 0")
    if not args.bag.is_file():
        parser.error(f"bag does not exist: {args.bag}")
    return args


def image_timestamp_ns(message: object) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def compressed_payload(message: object, topic: str) -> bytes:
    image_format = str(message.format).lower()
    payload = memoryview(message.data).tobytes()
    if not any(codec in image_format for codec in ("jpeg", "jpg", "png")):
        raise RuntimeError(
            f"{topic}: expected JPEG/PNG CompressedImage, got {message.format}"
        )
    if not payload:
        raise RuntimeError(f"{topic}: compressed image payload is empty")
    return payload


def write_frame(
    output: BinaryIO, timestamp_ns: int, topics: list[str], images: dict[str, object]
) -> None:
    output.write(struct.pack("<q", timestamp_ns))
    for topic in topics:
        payload = compressed_payload(images[topic], topic)
        output.write(struct.pack("<I", len(payload)))
        output.write(payload)


def stream(args: argparse.Namespace) -> int:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    output = sys.stdout.buffer

    complete_index = 0
    emitted = 0
    discarded_images = 0
    queues: dict[str, deque[tuple[int, object]]] = {
        topic: deque() for topic in args.topic
    }
    latest_by_topic: dict[str, int] = {}
    last_complete_timestamp = -1
    stop_requested = False

    with Reader(args.bag) as reader:
        topics = set(args.topic)
        available = {connection.topic for connection in reader.connections}
        missing = sorted(topics - available)
        if missing:
            raise RuntimeError("bag is missing image topics: " + ", ".join(missing))
        # A topic may have multiple connections after a publisher reconnects.
        connections = [c for c in reader.connections if c.topic in topics]
        for connection in connections:
            if connection.msgtype != "sensor_msgs/msg/CompressedImage":
                raise RuntimeError(
                    f"{connection.topic}: expected sensor_msgs/CompressedImage, "
                    f"got {connection.msgtype}"
                )
        output.write(MAGIC)
        output.write(struct.pack("<I", CAMERA_COUNT))

        for connection, _, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            timestamp_ns = image_timestamp_ns(message)
            previous_timestamp = latest_by_topic.get(connection.topic, -1)
            if timestamp_ns <= previous_timestamp:
                raise RuntimeError(
                    f"{connection.topic}: non-increasing image timestamp "
                    f"{timestamp_ns} after {previous_timestamp}"
                )
            latest_by_topic[connection.topic] = timestamp_ns
            queues[connection.topic].append((timestamp_ns, message))

            while all(queues[topic] for topic in args.topic):
                head_timestamps = {
                    topic: queues[topic][0][0] for topic in args.topic
                }
                earliest = min(head_timestamps.values())
                latest = max(head_timestamps.values())

                if latest - earliest > args.sync_tolerance_ns:
                    for topic in args.topic:
                        if head_timestamps[topic] == earliest:
                            queues[topic].popleft()
                            discarded_images += 1
                    continue

                images = {
                    topic: queues[topic].popleft()[1] for topic in args.topic
                }
                frame_timestamp = head_timestamps[args.topic[0]]
                if frame_timestamp <= last_complete_timestamp:
                    raise RuntimeError(
                        "synchronized frames are not in increasing timestamp order: "
                        f"{frame_timestamp} after {last_complete_timestamp}"
                    )
                last_complete_timestamp = frame_timestamp

                selected = (
                    complete_index >= args.start
                    and (complete_index - args.start) % args.step == 0
                )
                if selected:
                    write_frame(output, frame_timestamp, args.topic, images)
                    emitted += 1
                complete_index += 1

                if args.max_frames > 0 and emitted >= args.max_frames:
                    stop_requested = True
                    break
            if stop_requested:
                break

    buffered_images = sum(len(queue) for queue in queues.values())
    output.flush()
    print(
        f"NewCollege stream complete: synchronized={complete_index}, "
        f"emitted={emitted}, discarded_images={discarded_images}, "
        f"buffered_images={buffered_images}",
        file=sys.stderr,
    )
    if emitted == 0:
        raise RuntimeError("no synchronized frames selected; check topics, start, and tolerance")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return stream(args)
    except BrokenPipeError:
        return 0
    except Exception as error:
        print(f"NewCollege bag stream failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
